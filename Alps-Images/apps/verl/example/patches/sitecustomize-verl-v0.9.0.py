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
repeated unpatched in run 3125195).

separate-async never actually needs the hybrid engine: get_llm_client() is
overridden in PPOTrainerSeparateAsync to always route through the standalone
rollout, so the hybrid replicas exist only to be immediately put to sleep.

Patches applied:
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
"""
import sys

_PATCHED: set = set()


class _HybridRolloutOomPatcher:
    """sys.meta_path hook: patches specific verl modules immediately after they load."""

    _TARGETS = frozenset({
        "verl.workers.rollout.llm_server",
        "verl.trainer.ppo.v1.trainer_separate_async",
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
    # ── LLMServerManager: skip hybrid replica launch ────────────────────────────
    if name == "verl.workers.rollout.llm_server":
        cls = mod.LLMServerManager
        _orig_init_servers = cls._initialize_llm_servers

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

        cls._initialize_llm_servers = _initialize_llm_servers

    # ── PPOTrainerSeparateAsync: drop the hybrid-engine weight push at init ─────
    elif name == "verl.trainer.ppo.v1.trainer_separate_async":
        cls = mod.PPOTrainerSeparateAsync

        def _on_init_end(self):
            self.standalone_checkpoint_manager.update_weights(self.global_steps)
            print(
                "[sitecustomize-verl-v0.9.0] on_init_end: skipped "
                "self.checkpoint_manager.update_weights (naive backend, no hybrid "
                "replicas to push into)",
                flush=True,
            )

        cls.on_init_end = _on_init_end


sys.meta_path.append(_HybridRolloutOomPatcher())
