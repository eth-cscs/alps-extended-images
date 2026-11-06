"""GPU-free checks of the applied Megatron fixes and prebuilt CPU BFD helper."""
import argparse
import ast
import copy
import ctypes
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path("/opt/megatron-lm")


def source_function(relative, name, namespace, owner=None):
    tree = ast.parse((SOURCE / relative).read_text())
    body = tree.body
    if owner:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == owner).body
    node = copy.deepcopy(next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name))
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE / relative), "exec"), namespace)
    return namespace[name]


class MegatronRegressions(unittest.TestCase):
    def test_router_backend_selection_is_rocm_local_and_explicit(self):
        torch = SimpleNamespace(version=SimpleNamespace(hip="test"), float64=object())
        namespace = {"os": os, "torch": torch, "te_general_gemm": object()}
        select = source_function("megatron/core/transformer/moe/moe_utils.py", "_use_te_router_gemm", namespace)
        key = "APERTUS_ROCM_ROUTER_GEMM_BACKEND"
        with patch.dict(os.environ):
            os.environ.pop(key, None)
            self.assertFalse(select(object()))
            torch.version.hip = None
            self.assertTrue(select(object()))
            torch.version.hip = "test"
            for value, expected in (("aten", False), ("te", True)):
                os.environ[key] = value
                self.assertEqual(select(object()), expected)
            self.assertFalse(select(torch.float64))
            namespace["te_general_gemm"] = None
            self.assertFalse(select(object()))
            for value in ("", "auto", "rocblas"):
                os.environ[key] = value
                with self.subTest(value=value), self.assertRaises(ValueError):
                    select(object())
        tree = ast.parse((SOURCE / "megatron/core/transformer/moe/moe_utils.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RouterGatingLinearFunction")
        for name in ("forward", "backward"):
            method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertTrue(any(isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
                                and node.test.attr == "use_te_router_gemm" for node in ast.walk(method)))

    def test_dispatch_materializes_before_recording_the_event(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                calls = []
                class Tensor:
                    def __init__(self, name, dense=False, dtype="bf16"):
                        self.name, self.dense, self.dtype = name, dense, dtype
                    def contiguous(self):
                        calls.append(self.name + ".contiguous")
                        return Tensor(self.name, True, self.dtype)
                    def float(self):
                        calls.append(self.name + ".float")
                        return Tensor(self.name, self.dense, "fp32")
                def capture():
                    self.assertEqual(calls, ["output.contiguous", "probability.float", "probability.contiguous"])
                    calls.append("event")
                    return "event"
                def combine(x, handle, **kwargs):
                    self.assertTrue(x.dense)
                    self.assertTrue(kwargs["topk_weights"].dense)
                    self.assertEqual(kwargs["topk_weights"].dtype, "fp32")
                    self.assertEqual(kwargs["previous_event"], "event" if asynchronous else None)
                    self.assertEqual(kwargs["async_finish"], asynchronous)
                    self.assertEqual(kwargs["allocate_on_comm_stream"], asynchronous)
                    calls.append("combine")
                    return "dx", "dp", SimpleNamespace(current_stream_wait=lambda: calls.append("wait"))
                namespace = {"get_buffer": lambda group, size: SimpleNamespace(combine=combine),
                             "get_hidden_bytes": lambda tensor: 128, "EventHandle": capture,
                             "EventOverlap": lambda event: event}
                backward = source_function("megatron/core/transformer/moe/fused_a2a.py", "backward", namespace, "FusedDispatch")
                ctx = SimpleNamespace(group="group", handle="handle", async_finish=asynchronous,
                                      allocate_on_comm_stream=asynchronous)
                result = backward(ctx, Tensor("output"), None, Tensor("probability"), None, None)
                self.assertEqual(result, ("dx", None, "dp", None, None, None, None))
                self.assertEqual(calls[-1], "wait" if asynchronous else "combine")

    def test_bfd_is_prebuilt_fresh_and_numerically_correct(self):
        directory = SOURCE / "megatron/core/datasets"
        library = directory / "libbfd_pack.so"
        self.assertTrue(library.is_file(), "BFD must not compile at rank startup")
        self.assertGreaterEqual(library.stat().st_mtime_ns, (directory / "bfd_pack.cpp").stat().st_mtime_ns)
        function = ctypes.CDLL(str(library)).bfd_pack
        integer = ctypes.POINTER(ctypes.c_int)
        function.argtypes = [integer, ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, integer, integer, integer, integer]
        function.restype = None
        positions = (ctypes.c_int * 4)(0, 1, 2, 3)
        lengths = (ctypes.c_long * 4)(4, 3, 2, 1)
        documents = (ctypes.c_int * 4)(10, 11, 12, 13)
        for cap, expected, boundaries in ((0, [10, 13, 11, 12], [0, 2, 4]),
                                           (1, [10, 11, 12, 13], [0, 1, 2, 3, 4])):
            output = (ctypes.c_int * 4)()
            offsets = (ctypes.c_int * 5)()
            bins = ctypes.c_int()
            function(positions, lengths, 4, 5, cap, documents, output, offsets, ctypes.byref(bins))
            self.assertEqual(list(output), expected)
            self.assertEqual(list(offsets)[:bins.value + 1], boundaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    args = parser.parse_args()
    SOURCE = args.source_root.resolve(strict=True)
    unittest.main(argv=[__file__], verbosity=2)
