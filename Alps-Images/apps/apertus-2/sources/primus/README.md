# Primus Apertus2 provenance and acceptance

These are independent reference-derived candidates, not replacements for `megatron-rocm` or claims of full-model acceptance.
Primus 26.3 is the reference-generation baseline; 26.8 uses a newer compute stack with a version-gated Turbo API adapter.
Both use our RCCL and Alps network stack, retaining P2P off and all other Alps communication defaults.

## Reproducible inputs

- Megatron: exact committed tree `d1277d86e0c1925f8438fd7630d86d05ab398dcd`, Git tree `34b2f06da4a9ad5488320e8447075735059799ec`.
- Archive: `vendor/megatron-source.tar.gz`, SHA-256 `0ae108a350f69b4d4ce3e95ad75e6d7eb083edc773407144d76b9e5ddb11f9f5`, with per-file and symlink identity in `vendor/manifest.json`.
- The archive contains the committed reference source and licenses, not uncommitted edits, scratch binaries, external caches or recursive `_research` submodule contents.
- Emerging-Optimizers: `NVIDIA-NeMo/Emerging-Optimizers` commit `1f29ffd2acbf047f3dd3efdd8e9469d9305a8f77`.
- FLA/fla-core: PyPI0.5.2 wheels with hashes in `requirements.lock`, replacing only `ops/kda/gate.py` with reference SHA-256 `b350410cc605d1b3fac082d1e8384d7e9897e3aaa36c00f3d01d8d7fd48320ae`.
- `reference.env`: explicitly selected settings for the bundled reference source and its image-local tests.
- `configs/`: captured grouped-GEMM tile settings, FLA autotune tables and hipBLASLt tuning; the opt-in configuration selects historical FLA/hipBLASLt tables only for exact Torch 2.10/Triton 3.6.

## Runtime selection

The image installs no B2 Python startup hook and does not patch Triton's autotuner.
Autotuning failures retain the library's native failure behavior.
The bundled reference source, compatibility adapter, storage extension and tuning tables support self-contained reference tests; they are not required to activate ordinary SwissAI training.
Experiments using an external Megatron checkout own their compute-path settings.
For the bundled reference, opt in explicitly after loading the base environment:

```bash
source /opt/alps/env/source-alps-env.sh
source /opt/alps/apertus2-primus/reference.env
```

The fragment uses `defvar`, preserving explicit overrides, including empty values, and leaves network policy to the base image.
Image-local CI uses `tests/primus/with-reference-env.sh` to make this selection visible in its commands.
Existing published images retain their original behavior; this change takes effect only in rebuilt images.

## Installation contracts

`install-source.py` checks the archive digest, member paths, per-file checksums and internal symlinks before source acceptance.
The installer uses `--no-deps` and base constraints to avoid implicit compute upgrades.
The source's storage-sharing C++ extension and dataset helpers are rebuilt for the image ABI.
The app deliberately does not import the old native UCCL/TE rebuild policy or Imanol's communication overrides.
The two app targets conservatively hash the complete app `sources/` tree, selected tests/Containerfile/profile/compatibility patches and canonical base reference.
Primus26.3 retains the exact reference source without compatibility patches.
The26.8 upstream API probe confirmed that Turbo0.5.1 moved its offset helper and removed older private CU/gfx950/Origami helpers.
Its version-gated app-local patch changes only `beverin_turbo.py`, `beverin_gg.py` and `beverin_vk.py`: use the relocated offset/CU helpers, retain the newer vendor Origami selector, and explicitly constrain the reference pointer kernels to this profile's `gfx942` target rather than calling removed gfx950 knobs.
It neither modifies vendor Turbo files nor disables the pointer-GEMM/fused-wgrad paths.
`source-identity.json` records applied patch hashes and expected post-patch file identities, while the original archive remains unchanged.
The adapter is CPU source-validated and passes standalone reference pointer-GEMM forward/dgrad/fused-wgrad checks on the upstream26.8 compute stack with system-memory AQL rings.
Both Primus bases set `DEBUG_CLR_AQL_DEV_QUEUE=0` as a late-layer MI300 compatibility default: vendor26.8 otherwise segfaulted on its first simple Torch GPU fill, before reference kernels.
The older26.3 runtime is unaffected by this newer knob and passes the same numerical control.
This does not change vendor compute binaries, `GPU_USE_DEVICE_QUEUE`, RCCL or Alps network defaults.
New-image stack checks, multi-node communication and training/restart acceptance are still required.

## Acceptance gates

1. Compile each base and verify before/after vendor compute fingerprints.
2. Verify actual RCCL/plugin/libfabric/CXI loading and a single-GPU numerical smoke.
3. Export and independently verify complete SquashFS data readability and checksum, then pass base environment and multi-node collective tests.
4. Build the matching app from its canonical base only after base validation.
5. Pass exact source/metadata checks, reference private Turbo API checks, storage lifetime checks and AITER forward/backward smoke.
6. Pass fresh-process save2/load2/save4 with finite updates, zero skipped/NaN counts, actual per-rank optimizer ownership and successful native optimizer/RNG restoration.
7. Establish full-depth two-update/checkpoint and full-scale restart before the separate26-update performance comparison.

The declared CI training layouts are two-node EP8/PP1 and four-node EP4/PP2 with four consecutive expert ranks per node, matching T6's topology contract.
The tests keep writable JIT caches in executable node-local tmpfs and durable evidence on shared storage.
Consult image-specific CI records and retained run artifacts for outcomes; recipe presence and CPU checks do not establish GPU acceptance.
