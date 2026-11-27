# Run Chonk and Megachonk with vLLM

These scripts start vLLM on Alps GH200 nodes (four GPUs per node).
You do not need Spellbook, Ray, a source checkout, or extra patches.
`environment.toml` selects the CI-tested vLLM [`v0.31.0+Apertus2.1`](https://github.com/swiss-ai/vllm/releases/tag/v0.31.0%2BApertus2.1) image by its content-hashed tag.

## Layouts

Run the commands from the model directory.

| Model | Layout | Command | Nodes / GPUs | Use |
| --- | --- | --- | --- | --- |
| Chonk | TP1 × PP4 | `PP=4 sbatch --nodes=1 serve.sh` | 1 / 4 | Fastest Chonk layout |
| Chonk | TP1 × DP16, EP16 | `sbatch serve.sh` | 4 / 16 | Default; slower per GPU |
| Megachonk | TP4 × PP4 | `TP=4 PP=4 sbatch --nodes=4 serve.sh` | 4 / 16 | Best interactive layout |
| Megachonk | TP4 × PP8 | `TP=4 PP=8 sbatch serve.sh` | 8 / 32 | Lowest TPOT at 64 requests on real text |
| Megachonk | TP1 × PP8 | `TP=1 PP=8 sbatch --nodes=2 serve.sh` | 2 / 8 | Batch work only; needs FP8 weights |
| Megachonk | TP2 × DP16, EP32 | `sbatch serve.sh` | 8 / 32 | Default; slower per GPU |

`PP` greater than 1 starts one engine across all nodes.
Node 0 serves the API.
`TP × PP` must equal the GPU count of the job.
Chonk has seven KV heads, so it always uses TP1.

PP layouts use 256 sequences, 2,048 batched tokens, and GPU memory utilization 0.90 (Chonk: 0.88).
At 0.95, Megachonk TP4 × PP4 and TP4 × PP8 ran out of GPU memory.
DP layouts use 8 sequences per engine, 512 batched tokens, and 0.95.
All layouts use `allgather_reducescatter`, Triton experts, CUDA graphs, and NUMA binding.
To change the memory value, set `GPU_MEMORY_UTILIZATION`.

## Checkpoints

The scripts use shared BF16 checkpoints by default.
To use another checkpoint, set `MODEL`:

```bash
MODEL=/path/to/megachonk-fp8 TP=4 PP=4 sbatch --nodes=4 serve.sh
```

The fastest layouts use block-FP8 checkpoints from [swiss-ai/hfconverter#8](https://github.com/swiss-ai/hfconverter/pull/8) (open).
Routed experts, shared experts, and dense MLPs are FP8 with 128 × 128 blocks.
vLLM reads the quantization from the checkpoint, so keep `FP8=0`.

A block-FP8 checkpoint has the same speed as online FP8.
For Megachonk, online FP8 ran out of memory during loading.
Megachonk TP1 × PP8 needs about 70 GiB of FP8 weights per GPU, so BF16 does not fit.

## Start a server

```bash
mkdir -p /iopsstor/scratch/cscs/$USER/tmp
cd chonk  # Or megachonk.
sbatch --account=infra01 --partition=normal serve.sh
```

If you submit from another directory, set `CONTAINER_EDF` to the full path of `environment.toml`.

The context limit includes input and output tokens.
The default is 65,536 tokens for Chonk and 32,768 for Megachonk:

```bash
MAX_MODEL_LEN=32768 sbatch serve.sh
```

In a DP layout, `sbatch --nodes=N` changes the node count, and the script calculates DP.
For example, `sbatch --nodes=1 serve.sh` starts Chonk TP1 × DP4.

Megachonk DP needed about eight minutes to start.
The server URL is in `server-url.txt`.

## Online FP8 and EPLB

These options are for BF16 checkpoints.
Both default to `0`.

```bash
FP8=1 sbatch serve.sh                     # Routed experts in FP8.
FP8=1 FP8_SCOPE=mlp sbatch serve.sh       # Also shared experts and dense MLPs.
FP8=1 MOE_BACKEND=deep_gemm sbatch serve.sh
EPLB=1 sbatch serve.sh                    # Chonk DP layout only.
```

Online FP8 uses 128 × 128 blocks and CPU-staged loading.
`FP8_SCOPE=mlp` keeps attention, KDA, latent projections, embeddings, and the output head in BF16.
`EPLB=1` balances the experts across the EP GPUs; it does not add replicas.
With vLLM 0.28, all three Chonk FP8 variants passed serving and short numerical checks against BF16 on one node.
For Megachonk, use a block-FP8 checkpoint.

## Send a request

Replace `HEAD_IP` with the address in the server URL.

```bash
curl http://HEAD_IP:8000/health

curl http://HEAD_IP:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chonk","prompt":"The capital of Switzerland is","max_tokens":32,"temperature":0}'
```

Use `"model":"megachonk"` for Megachonk.
The API listens on the cluster network.
For outside access, use an authenticated SSH tunnel, not a public proxy.

## Speed

Measured on GH200 in November 2026 with `vllm-apertus2-v031-cuda:alps7-dev-2890e3d6`, an earlier build of the same vLLM source.
The default image adds Mooncake CXI and UCCL `1c9e39bb`; it was not benchmarked separately.
The benchmark also applied three vLLM runtime patches that are not in the image: two for DeepEP with DBO, and one NUMA change for online FP8 loading.

Prompts are random tokens, and values are the mean of three passes, unless noted.
"FP8" is a block-FP8 checkpoint.
**Throughput is per GPU, over all GPUs of the deployment.**

| Test | Input / output tokens | Requests in flight | Values |
| --- | --- | ---: | --- |
| c1 | 1,024 / 128 | 1 | Mean time per output token (TPOT) |
| c64 | 1,024 / 128 | 64 | Tokens/s per GPU, mean TPOT |
| prefill-16k | 16,384 / 32 | 8 | Input tokens/s per GPU, mean time to first token (TTFT) |
| decode-1024 | 256 / 1,024 | 8 | Mean TPOT |

### Megachonk

| Layout | Weights | GPUs | c1 TPOT | c64 tok/s/GPU | c64 TPOT | prefill tok/s/GPU | prefill TTFT | decode TPOT | KV tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4 × PP4 | FP8 | 16 | **16.0 ms** | 742 | **43.1 ms** | 1,699 | 1.52 s | 23.1 ms | 4.4 M |
| TP1 × PP8 | FP8 | 8 | 26.6 ms | **769** | 84.7 ms | **2,297** | 1.91 s | — | 0.32 M |
| TP4 × PP4 | BF16 | 16 | 18.6 ms | 575 | 56.7 ms | 1,436 | 1.59 s | — | — |
| TP4 × PP8 | BF16 | 32 | 18.4 ms | 349 | 46.9 ms | 1,156 | 1.20 s | — | — |

- FP8 gives 29% more c64 throughput than BF16.
- TP1 × PP8 has the most throughput per GPU, but 1.7 times the TPOT.
  Use it for batch work.
- TP1 × PP16 and TP1 × PP32 (BF16) had about twice the TPOT of TP4.

**Real text, FP8, first of three passes.**
Prompts come from seven public sources, cut to exact token counts without a chat template.
Against random tokens, throughput is 9–11% lower and TPOT is 12–16% higher; prefill is the same.

| Layout | GPUs | c1 TPOT | c64 tok/s/GPU | c64 TPOT | prefill tok/s/GPU | prefill TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4 × PP4 | 16 | — | 659 | 49.6 ms | — | — |
| TP4 × PP8 | 32 | 16.1 ms | 424 | **38.4 ms** | 1,385 | 0.91 s |
| TP1 × PP8 | 8 | — | 699 | 94.4 ms | — | — |

### Chonk

| Layout | Weights | GPUs | c1 TPOT | c64 tok/s/GPU | c64 TPOT | prefill tok/s/GPU | prefill TTFT | decode TPOT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP1 × PP4 | FP8 | 4 | **9.0 ms** | **5,385** | 23.9 ms | **13,017** | **0.83 s** | 14.7 ms |
| TP1 × DP4, EP4 | BF16 | 4 | 10.7 ms | 4,686 | 23.2 ms | 9,657 | 1.68 s | — |
| TP1 × DP16, EP16 | BF16 | 16 | 17.7 ms | 1,093 | 28.8 ms | 2,536 | 1.81 s | — |
| TP1 × DP16, EP16 | FP8, DeepGEMM | 16 | 16.7 ms | 1,056 | — | 2,468 | — | — |

- TP1 × PP4 is the fastest in every test.
  Its TTFT at c1 is 107 ms.
- Four nodes give about one fifth of the one-node throughput per GPU, because of network all-to-all.

### Expert parallelism over the network

The scripts do not set up these layouts.
The runs used UCCL DeepEP (`1c9e39bb`) and tested prefill and decode separately.
The PP layouts above are still faster per GPU.

- Megachonk TP2 × DP16, EP32, decode only: DeepEP low-latency is faster from 64 requests in flight (TPOT 70.4 vs 76.4 ms; 87.9 vs 150.4 ms at 256).
  Below 64, it is 9–12 ms per token slower, because SSSGLU is not fused into DeepGEMM.
- Megachonk prefill only: DeepEP high-throughput gives 1.3–1.7 times the throughput of `allgather_reducescatter`.
  With DBO, it gives up to 1.75 times at 32 or more requests, but 10–27% less at 8.
- Chonk TP1 × DP16, EP16, prefill only: high-throughput gives 1.3–1.7 times, and up to 1.76 times with DBO.
  Chonk cannot use low-latency mode, because the latent size 1,792 is not supported.

Settings for these runs:

- DeepEP needs `--moe-backend deep_gemm`.
- High-throughput mode has no CUDA graphs.
  Do not use it for decode (Chonk TPOT 104 ms vs 17 ms).
- UCCL high-throughput configs (`UCCL_EP_DISPATCH_CONFIG` / `UCCL_EP_COMBINE_CONFIG`): Chonk `24,8,512,32,512` / `24,4,1024,24,512`; Megachonk `20,8,512,16,1024` / `20,2,512,16,512`.
  At EP32, a combine `rdma_buffer` of 1,024 overflows the 1 GiB DeepEP buffer (`CXI CQ error: NO_SPACE`).
- With DeepGEMM and 4,096-token steps, use GPU memory utilization 0.80.
- DBO needs vLLM [#60735](https://github.com/vllm-project/vllm/pull/60735) and [#48659](https://github.com/vllm-project/vllm/pull/48659) (open).
  Without #48659, DBO with FlashAttention can hang (`batch_size must be equal to batch_size_k`).

### Earlier long-context runs

October 2026, DP layouts, BF16, vLLM 0.28.
32 requests in flight, 128 output tokens each.
**Values are tokens/s per request.**

| Model | GPUs | Input tokens | Prefill (input tokens / median TTFT) | Decode (1,000 / median TPOT in ms) |
| --- | ---: | ---: | ---: | ---: |
| Chonk | 16 | 32,512 | 1,957 | 14.2 |
| Megachonk | 32 | 32,512 | 747 | 5.9 |
| Chonk | 16 | 65,280 | 470 | 16.4 |

The prefill value includes queue time.
Chonk near 64k had a median TTFT of 138.8 s and a median TPOT of 60.8 ms.

## Logs and stop

The Slurm log is `/iopsstor/scratch/cscs/$USER/tmp/vllm-<model>-<job>.out`.
The `vllm-<model>-<job>/` directory next to it has `server.log` and `server-url.txt`.
Caches use node-local `/tmp`; the job does not write to HOME.

The server stops after two hours.
To stop it earlier, use `scancel JOB_ID`.
