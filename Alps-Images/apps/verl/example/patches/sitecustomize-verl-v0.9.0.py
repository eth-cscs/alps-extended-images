"""
Hybrid-rollout OOM fallback for verl v0.9.0, V1 separate-async trainer.

Injected at Python startup via PYTHONPATH — does not modify verl source files.
Distinct from sitecustomize.py (written against v0.8.0; class/module paths there
are not verified against v0.9.0) so that file is left alone. This file must be
staged as the process's actual "sitecustomize.py" on sys.path to be auto-loaded —
see the training script for how it's copied into place per node.

Why: PPOTrainer._setup() (verl/trainer/ppo/v1/trainer_base.py, v0.9.0) always
builds hybrid rollout replicas on top of the training worker group
(trainer_world_size / rollout_world_size replicas, e.g. 288/32 = 9 here) *in
addition to* the standalone rollout — actor_rollout_ref.hybrid_engine is not
consulted anywhere in the V1 path, so this cannot be disabled from config. Those
replicas are instantiated at gpu_memory_utilization=0.75 on the training GPUs
during trainer.init(), before the first on_sample_end() ever runs, and
free_cache_engine=false (required for TP=32 SGLang stability) makes their
sleep() a no-op — so they hold ~71 GiB per training GPU for the life of the run.
trainer.init() then needs ~15 GiB back on those same GPUs to stage Megatron
params for the NCCL export to the standalone rollout, and OOMs (run 3121001,
repeated unpatched in run 3125195, and repeated a third time in run 3129805
because a script-relative path bug meant this file never even got copied into
place).

Patches applied via a sys.meta_path find_spec hook that wraps each target
module's real loader — the patch code runs only once the target module is
actually imported by something that needs it (verl itself, in the worker/actor
processes that construct the trainer or the rollout replicas), the same as
before this file existed. This is NOT the same as an earlier version of this
file, which used the legacy find_module/load_module protocol: that hook style
is deprecated since Python 3.4, and its importlib compatibility shim no longer
calls find_module/load_module at all on the Python 3.11/3.12 in this image
(confirmed by a local repro where find_module was never invoked for an ordinary
import) — so that version silently never patched anything, in any run, ever.
A later revision fixed *that* by importing both target modules eagerly right
here instead, which does work, but eagerly forces transformers/torch/verl's
whole worker-side import chain into every single Python process that inherits
this PYTHONPATH — including `ray start` itself and Ray's own internal daemon
processes, not just verl's own worker/actor processes. That is suspected (not
yet proven) of interfering with Ray cluster/actor formation on the Apertus
benchmark script, which uses an analogous eager-import sitecustomize patch for
a different bug and started stalling at Ray actor creation only after that
patch started actually executing (it never stalled there across many runs
while the patch was still a no-op). This version restores the original lazy,
import-on-demand behavior, using the modern find_spec/Loader.exec_module hook
(verified locally to actually fire, unlike find_module) instead of either of
the previous two approaches.

  - LLMServerManager._initialize_llm_servers (verl.workers.rollout.llm_server):
    no-ops when called in hybrid mode (worker_group is not None) instead of
    launching replicas via init_hybrid(). Standalone-mode calls (worker_group
    is None) are unaffected.
  - PPOTrainerSeparateAsync.on_init_end (verl.trainer.ppo.v1.trainer_separate_async):
    drops the self.checkpoint_manager.update_weights(...) call. That manager's
    backend is forced to "naive" (trainer_base.py), which pushes weights into
    each worker's colocated hybrid engine directly rather than going through
    the (now empty) replica list — with hybrid replicas disabled above, that
    colocated engine is never created, so the call would push into nothing.
    self.standalone_checkpoint_manager.update_weights(...), the actual sync to
    the standalone rollout, is left untouched.

separate-async never actually needs the hybrid engine: get_llm_client() is
overridden in PPOTrainerSeparateAsync to always route through the standalone
rollout, so the hybrid replicas exist only to be immediately put to sleep.

  - list_of_dict_to_tensordict (verl.utils.tensordict_utils): replaces the
    dense-vs-nested heuristic with an unconditional delegation to the same
    file's own nested_tensor_from_tensor_list. Found via run 3134772/3136766:
    training crashed twice with AttributeError: 'Tensor' object has no
    attribute 'offsets' (once at engine_workers.py:294, once at
    padding.py:119, after bumping ppo_mini_batch_size only reduced rather
    than eliminated the crash). Root cause: this function decides
    nested-vs-stacked per field by checking `all(item.shape ==
    val_list[0].shape for item in val_list)` — true whenever items
    coincidentally share a shape, which is guaranteed for a length-1 list
    (verl calls this per rollout output, so len(list_of_dicts) is often 1)
    and common otherwise whenever an undertrained model saturates
    max_response_length across a rollout group, silently producing a dense
    Tensor for fields callers assume are nested (input_ids, prompts,
    responses, position_ids). Confirmed via the pinned TransferQueue==0.1.7
    wheel (verl/requirements.txt) that the read side already does this
    correctly — transfer_queue/storage/managers/simple_storage_manager.py's
    _pack_field_values always tries torch.nested.as_nested_tensor first for
    non-scalar tensor lists, never taking a shape-equality shortcut — and
    that nested_tensor_from_tensor_list, elsewhere in this very same verl
    file (used for chunking/dispatch), already builds nested tensors
    unconditionally with no such shortcut either. list_of_dict_to_tensordict
    is the one outlier; the patch just brings it in line with both.

  - no_padding_2_padding (verl.workers.utils.padding) -- DIAGNOSTIC ONLY, no
    fix. Run 3137775 confirmed the list_of_dict_to_tensordict patch above
    fired everywhere (its confirmation line appeared ~283 times) yet the
    IDENTICAL offsets AttributeError recurred at the SAME call site
    (padding.py:119) as run 3136766 -- so that patch, while a real fix for
    its own bug, was not the (sole) cause of this one. Traced as far as
    static analysis + a local torch/tensordict repro can go: ruled out
    shape-coincidence in TransferQueue's own read-side reconstruction
    (torch.nested.as_nested_tensor never silently densifies same-length
    inputs -- confirmed empirically) and ruled out the leading-dim-of-1
    write-time storage quirk in simple_storage_manager.py's
    _select_by_positions (also stays properly nested with a working
    .offsets() -- confirmed empirically). Found instead, empirically, that
    torch.nested.as_nested_tensor(..., layout=torch.jagged) raises on a
    dtype mismatch across items, and TransferQueue's _pack_field_values
    catches exactly that and falls back to layout=torch.strided, which
    reports is_nested=True and type() == torch.Tensor but has NO .offsets()
    -- reproducing the crash's exact error message character for character
    in a local repro. TransferQueue's own code already logs a warning on
    this fallback, but that string does not appear in either prior log
    (grepped both) -- inconclusive, since Python `logging` module output is
    not reliably captured the way `print(..., flush=True)` is in Ray
    workers, so this could mean either the fallback isn't happening or the
    log just didn't reach stdout. This wrapper prints, via `print(...,
    flush=True)` (bypasses that uncertainty), the nested-state of `tensor`,
    `prompts`, `responses`, and `attention_mask` immediately before every
    no_padding_2_padding call, so the next run's log gives a definitive
    answer (which field, which of the three states) instead of another
    guess. Remove once the real mechanism is confirmed and the actual fix
    (wherever it turns out to belong -- verl or the external TransferQueue
    package) is written.
"""

import sys
import importlib.abc
import importlib.util


def _patch_llm_server(mod) -> None:
    _orig_init_servers = mod.LLMServerManager._initialize_llm_servers

    async def _initialize_llm_servers(self, start_rank: int = None):
        if self.worker_group is not None:
            self.rollout_replicas = []
            self.server_handles = []
            self.server_addresses = []
            print(
                "[sitecustomize-verl-v0.9.0] LLMServerManager: hybrid replicas "
                "disabled (worker_group is set) — skipping init_hybrid()",
                flush=True,
            )
            return
        await _orig_init_servers(self, start_rank=start_rank)

    mod.LLMServerManager._initialize_llm_servers = _initialize_llm_servers


def _patch_trainer_separate_async(mod) -> None:
    def _on_init_end(self):
        self.standalone_checkpoint_manager.update_weights(self.global_steps)
        print(
            "[sitecustomize-verl-v0.9.0] on_init_end: skipped "
            "self.checkpoint_manager.update_weights (naive backend, no hybrid "
            "replicas to push into)",
            flush=True,
        )

    mod.PPOTrainerSeparateAsync.on_init_end = _on_init_end


def _patch_tensordict_utils(mod) -> None:
    torch = mod.torch
    NonTensorStack = mod.NonTensorStack

    def _list_of_dict_to_tensordict(list_of_dicts):
        assert mod.parse_version(mod.tensordict.__version__) >= mod.parse_version("0.10"), (
            "Storing non-tensor data in TensorDict at least requires tensordict version 0.10"
        )
        assert len(list_of_dicts) > 0

        keys = list_of_dicts[0].keys()
        dict_of_lists = {key: [d[key] for d in list_of_dicts] for key in keys}
        batch_size = len(list_of_dicts)

        def _pack(val_list):
            if not val_list or not all(isinstance(item, torch.Tensor) for item in val_list):
                return NonTensorStack(*val_list)
            # Scalar tensors have no dimension to make ragged along.
            if all(item.dim() == 0 for item in val_list):
                return torch.stack(val_list)
            # Always nested — never guess dense-vs-ragged from shape equality.
            return mod.nested_tensor_from_tensor_list(val_list)

        final_data = {key: _pack(val_list) for key, val_list in dict_of_lists.items()}
        return mod.TensorDict(final_data, batch_size=[batch_size])

    mod.list_of_dict_to_tensordict = _list_of_dict_to_tensordict
    print(
        "[sitecustomize-verl-v0.9.0] tensordict_utils: patched "
        "list_of_dict_to_tensordict to always build nested tensors for "
        "non-scalar tensor fields instead of guessing from shape equality",
        flush=True,
    )


def _describe_tensor_field(name: str, t) -> str:
    """Distinguish the three possible nested-vs-plain states a field can be in.

    A real jagged NestedTensor (type() != Tensor, is_nested=True) has .offsets().
    A degraded strided "nested" tensor (type() == Tensor, is_nested=True) does
    NOT have .offsets() -- torch.nested.as_nested_tensor(..., layout=jagged)
    silently falls back to this when jagged construction raises (e.g. a dtype
    mismatch across items), and this reproduces the exact observed crash:
    AttributeError: 'Tensor' object has no attribute 'offsets'. A genuinely
    dense/plain tensor (is_nested=False) is a third, distinct possibility.
    """
    try:
        is_nested = bool(getattr(t, "is_nested", False))
        type_name = type(t).__name__
        dtype = getattr(t, "dtype", None)
    except Exception as e:
        return f"{name}: <diagnostic inspection failed: {e!r}>"

    if is_nested and type_name != "Tensor":
        return f"{name}: OK nested(jagged) type={type_name} dtype={dtype}"
    if is_nested and type_name == "Tensor":
        # .shape itself raises on a degraded strided nested tensor (confirmed
        # locally: "NestedTensorImpl doesn't support sizes") -- don't let that
        # obscure the classification, which is already conclusive without it.
        return f"{name}: DEGRADED nested(strided, NO .offsets()!) type={type_name} dtype={dtype}"
    try:
        shape = tuple(t.shape)
    except Exception:
        shape = "<shape unavailable>"
    return f"{name}: DENSE (not nested at all) type={type_name} dtype={dtype} shape={shape}"


def _patch_padding_diagnostics(mod) -> None:
    """Diagnostic-only wrapper (see run 3134772/3136766/3137775 in CLAUDE.md):
    the offsets AttributeError has now recurred three times at this exact
    function despite two prior fixes (mini-batch-size bump, then the
    list_of_dict_to_tensordict write-path patch). This does not attempt a
    fourth fix -- it prints the nested-state of every field this function
    reads, right before it would crash, so the next run's log tells us
    definitively which of the three states above is occurring and for which
    field, instead of guessing again.
    """
    _orig = mod.no_padding_2_padding

    def _no_padding_2_padding(tensor, data):
        try:
            parts = [_describe_tensor_field("tensor_arg", tensor)]
            for key in ("prompts", "responses", "attention_mask"):
                if key in data.keys():
                    parts.append(_describe_tensor_field(key, data[key]))
            print(
                "[sitecustomize-verl-v0.9.0] DIAGNOSTIC no_padding_2_padding: " + " | ".join(parts),
                flush=True,
            )
        except Exception as e:
            print(
                f"[sitecustomize-verl-v0.9.0] DIAGNOSTIC no_padding_2_padding: inspection itself failed: {e!r}",
                flush=True,
            )
        return _orig(tensor, data)

    mod.no_padding_2_padding = _no_padding_2_padding
    print(
        "[sitecustomize-verl-v0.9.0] padding: installed diagnostic wrapper around "
        "no_padding_2_padding (prints nested-state of tensor/prompts/responses "
        "before every call)",
        flush=True,
    )


_TARGETS = {
    "verl.workers.rollout.llm_server": _patch_llm_server,
    "verl.trainer.ppo.v1.trainer_separate_async": _patch_trainer_separate_async,
    "verl.utils.tensordict_utils": _patch_tensordict_utils,
    "verl.workers.utils.padding": _patch_padding_diagnostics,
}
_PATCHED: set = set()


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, fullname, orig_loader):
        self._fullname = fullname
        self._orig_loader = orig_loader

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self._orig_loader.exec_module(module)
        if self._fullname not in _PATCHED:
            _PATCHED.add(self._fullname)
            _TARGETS[self._fullname](module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS or fullname in _PATCHED:
            return None
        sys.meta_path.remove(self)
        try:
            real_spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if real_spec is None or real_spec.loader is None:
            return None
        real_spec.loader = _PatchLoader(fullname, real_spec.loader)
        return real_spec


sys.meta_path.insert(0, _PatchFinder())
