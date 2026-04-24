# Batch Size Notes — PyTorch vs llama.cpp

**Status:** Design notes. No batching code is implemented in the current harness;
this document captures the reasoning so the next iteration has a starting point.

## Why batch size matters for this comparison

The two engines we care about have very different sweet spots. A single batch
size will flatter one of them.

### llama.cpp / GGUF
- Optimized for **single-stream, latency-sensitive** inference.
- Kernels and KV-cache layout assume `batch = 1` is the common case.
- Supports batching via `llama-batched` / `n_parallel`, but it is not where it
  shines.

### PyTorch + HF transformers (GPTQ, Marlin, AWQ kernels)
- Gains significantly from batching.
- GPU matmuls become **compute-bound** instead of memory-bound.
- Throughput scales nearly linearly until VRAM saturates (typically KV-cache).

## Which batch size to use

| Goal | Right batch size |
|---|---|
| "Which is faster for a single user / chat app / local assistant" | `batch = 1` |
| "Which is faster for a serving backend" | sweep `[1, 2, 4, 8, 16, 32]`, stop at OOM or plateau |
| "Apples-to-apples engine comparison" | report **both** batch=1 latency and batch=N throughput on the same hardware |

## Recommendation for this repo

Add a `batch_sizes = [1, 4, 8, 16]` sweep. For a 7B model in 4-bit on a
consumer GPU (~12–24 GB), 16 is usually the ceiling before KV-cache OOM at
`max_new_tokens = 128`. Report two metrics:

- **batch=1** — interactive latency (TTFT, TPOT)
- **batch=max** — server throughput (aggregate tokens/sec)

This is the only way to honestly characterize llama.cpp's weakness (it won't
scale much past 4–8) and PyTorch's strength (GPTQ kernels often 4–8x throughput
by batch=16).

## Reporting rules

- **TTFT becomes meaningless in batched mode** — all requests in the batch get
  their first token at the same wall-clock moment. In batched runs, drop TTFT
  and report:
  - Mean per-request latency (end-to-end)
  - Aggregate throughput (total new tokens / total wall time)
  - Peak memory
- Keep TTFT/TPOT only for the `batch = 1` runs.

## Out of scope (for this iteration)

- Actual batching implementation in `benchmark_gguf.py` / `benchmark_quantize.py`.
- Padding strategy for mixed-length prompts.
- Request scheduling / continuous batching (vLLM-style).

These are explicitly deferred to a future PR.
