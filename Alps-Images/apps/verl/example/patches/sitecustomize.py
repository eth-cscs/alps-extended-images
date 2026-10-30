"""
Debug instrumentation for verl fully-async training.

Injected at Python startup via PYTHONPATH — does not modify verl source files.
Remove the patches/ directory from PYTHONPATH to disable all patches.

Patches applied:
  - FullyAsyncRollouter._process_single_sample_streaming: log dispatch + elapsed time per sample
  - FullyAsyncRollouter._streaming_generation_main: 30-second heartbeat
  - SGLangHttpServer.abort_all_requests: log before/after pause_generation
  - SGLangHttpServer.resume_generation: log before/after continue_generation
  - CheckpointEngineManager.abort_replicas: log before/after
  - CheckpointEngineManager.build_process_group: log before/after
  - CheckpointEngineManager.resume_generation_replicas: log before/after
"""
import sys

_PATCHED: set = set()


class _VerlDebugPatcher:
    """sys.meta_path hook: patches specific verl modules immediately after they load."""

    _TARGETS = frozenset({
        "verl.experimental.fully_async_policy.fully_async_rollouter",
        "verl.workers.rollout.sglang_rollout.async_sglang_server",
        "verl.checkpoint_engine.base",
    })

    def find_module(self, fullname, path=None):
        if fullname in self._TARGETS and fullname not in _PATCHED:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        # Remove self temporarily to avoid recursion during the real import.
        sys.meta_path[:] = [m for m in sys.meta_path if m is not self]
        try:
            import importlib
            mod = importlib.import_module(fullname)
        finally:
            sys.meta_path.append(self)
        _PATCHED.add(fullname)
        _apply_patches(fullname, mod)
        return mod


def _apply_patches(name: str, mod) -> None:
    import asyncio as _aio
    import time as _t

    # ── Rollouter ──────────────────────────────────────────────────────────────
    if name == "verl.experimental.fully_async_policy.fully_async_rollouter":
        cls = mod.FullyAsyncRollouter

        # 1. Time the SGLang round-trip for each sample.
        _orig_proc = cls._process_single_sample_streaming

        async def _proc(self, rollout_sample):
            sid = rollout_sample.sample_id
            t0 = _t.time()
            print(
                f"[DEBUG][Rollouter] Dispatching sample {sid} "
                f"at {_t.strftime('%H:%M:%S')} active_tasks={len(self.active_tasks)}",
                flush=True,
            )
            try:
                await _orig_proc(self, rollout_sample)
            finally:
                print(
                    f"[DEBUG][Rollouter] Sample {sid} completed in {_t.time() - t0:.1f}s",
                    flush=True,
                )

        cls._process_single_sample_streaming = _proc

        # 2. Heartbeat every 30s so we can see the rollouter is still alive.
        _orig_main = cls._streaming_generation_main

        async def _main(self):
            async def _heartbeat():
                t0 = _t.time()
                while True:
                    await _aio.sleep(30)
                    print(
                        f"[DEBUG][Rollouter][heartbeat] "
                        f"elapsed={_t.time() - t0:.0f}s "
                        f"active_tasks={len(self.active_tasks)} "
                        f"generated={self.total_generated_samples} "
                        f"processed={self.processed_sample_count} "
                        f"paused={self.paused}",
                        flush=True,
                    )

            hb = _aio.ensure_future(_heartbeat())
            try:
                await _orig_main(self)
            finally:
                hb.cancel()
                await _aio.gather(hb, return_exceptions=True)

        cls._streaming_generation_main = _main

    # ── SGLang HTTP server ─────────────────────────────────────────────────────
    elif name == "verl.workers.rollout.sglang_rollout.async_sglang_server":
        cls = mod.SGLangHttpServer

        _orig_abort = cls.abort_all_requests

        async def _abort(self):
            if self.node_rank == 0:
                print(
                    f"[DEBUG][SGLang] abort_all_requests → pause_generation {_t.strftime('%H:%M:%S')}",
                    flush=True,
                )
            await _orig_abort(self)
            if self.node_rank == 0:
                print(
                    f"[DEBUG][SGLang] abort_all_requests ← pause_generation returned {_t.strftime('%H:%M:%S')}",
                    flush=True,
                )

        cls.abort_all_requests = _abort

        _orig_resume = cls.resume_generation

        async def _resume(self):
            if self.node_rank == 0:
                print(
                    f"[DEBUG][SGLang] resume_generation → continue_generation {_t.strftime('%H:%M:%S')}",
                    flush=True,
                )
            await _orig_resume(self)
            if self.node_rank == 0:
                print(
                    f"[DEBUG][SGLang] resume_generation ← continue_generation returned {_t.strftime('%H:%M:%S')}",
                    flush=True,
                )

        cls.resume_generation = _resume

    # ── Checkpoint engine manager ──────────────────────────────────────────────
    elif name == "verl.checkpoint_engine.base":
        from verl.utils.ray_utils import auto_await as _auto_await

        cls = mod.CheckpointEngineManager

        _orig_abort_r = cls.abort_replicas

        @_auto_await
        async def _abort_replicas(self):
            print(f"[DEBUG][CheckpointEngine] step 1: abort_replicas → {_t.strftime('%H:%M:%S')}", flush=True)
            await _orig_abort_r(self)
            print(f"[DEBUG][CheckpointEngine] step 1: abort_replicas ← {_t.strftime('%H:%M:%S')}", flush=True)

        cls.abort_replicas = _abort_replicas

        _orig_build_pg = cls.build_process_group

        def _build_process_group(self, rollout):
            print(f"[DEBUG][CheckpointEngine] step 4: build_process_group → {_t.strftime('%H:%M:%S')}", flush=True)
            result = _orig_build_pg(self, rollout)
            print(f"[DEBUG][CheckpointEngine] step 4: build_process_group ← {_t.strftime('%H:%M:%S')}", flush=True)
            return result

        cls.build_process_group = _build_process_group

        _orig_resume_gen_r = cls.resume_generation_replicas

        @_auto_await
        async def _resume_generation_replicas(self):
            print(f"[DEBUG][CheckpointEngine] step 8: resume_generation_replicas → {_t.strftime('%H:%M:%S')}", flush=True)
            await _orig_resume_gen_r(self)
            print(f"[DEBUG][CheckpointEngine] step 8: resume_generation_replicas ← {_t.strftime('%H:%M:%S')}", flush=True)

        cls.resume_generation_replicas = _resume_generation_replicas


sys.meta_path.append(_VerlDebugPatcher())
