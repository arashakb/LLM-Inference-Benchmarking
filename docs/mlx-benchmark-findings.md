# MLX Benchmark Findings: Apple Silicon Gets Its Proper Backend

**Date:** 2026-04-16
**Run dir:** `outputs/benchmark_mlx/20260416-095352/`

## Context

The previous MPS run (`docs/mps-gguf-findings.md`) showed GGUF_TORCH is
~13× slower than FP16 on Apple Silicon — a pure-PyTorch fallback that
doesn't represent real quantized inference on Mac. This run uses
**MLX** (Apple's native ML framework) via `mlx-lm`, which provides Metal-
accelerated inference with unified memory. Both the FP16 baseline and
the 4-bit quantized model run on the same MLX stack, giving a clean
apples-to-apples quantization comparison.

## Environment

| Field | Value |
|---|---|
| Device | Apple Silicon (MPS/Metal via MLX) |
| MLX version | 0.31.1 |
| mlx-lm version | 0.31.2 |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Quantized model | mlx-community/Qwen2.5-0.5B-Instruct-4bit |
| Prompts | 10 × 128 max new tokens, greedy decode |
| Warmup | 2 runs |

## Results

| Metric | FP16 (MLX) | MLX 4-bit | Delta |
|---|---|---|---|
| Throughput (tok/s) | 177.96 | **334.77** | **+88.1%** |
| TTFT (ms) | 79.97 | 74.95 | -6.3% |
| TPOT (ms) | 4.98 | **2.42** | **-51.4%** |
| Latency / prompt (ms) | 656.89 | **382.35** | -41.8% |
| Peak memory (MB) | 998.7 | **298.1** | **-70.1%** |
| Speedup | — | **1.88×** | — |
| Memory saved | — | **70.1%** | — |

No `tpot_skipped` warnings. Clean run, no crashes.

## Key takeaways

### 1. MLX is the right backend for Apple Silicon

The FP16 MLX run alone (177.96 tok/s) is **3.5× faster** than the
FP16 PyTorch-on-MPS run from `benchmark_gguf.py` (50.15 tok/s). MLX's
native Metal kernels and unified memory architecture deliver a step
change in throughput on the same hardware without any quantization.

### 2. 4-bit quantization delivers real gains on MLX

Unlike GGUF_TORCH (which was *slower* and used *more* memory than FP16),
MLX 4-bit quantization delivers exactly what you'd expect:

- **1.88× throughput** — nearly double the tokens per second
- **70% memory reduction** — model footprint drops from ~1 GB to ~300 MB
- **TPOT halved** (4.98 → 2.42 ms) — decode-bound workloads benefit most
- **TTFT nearly identical** — prefill is not bottlenecked by weight size
  at this model scale

### 3. Comparison across all Mac backends

| Backend | Throughput (tok/s) | TPOT (ms) | Memory (MB) | Notes |
|---|---|---|---|---|
| GGUF_TORCH (PyTorch) | 3.91 | 255.87 | 1,117 | Unusable; pure-PyTorch fallback |
| FP16 PyTorch on MPS | 50.15 | 19.61 | 1,069 | HuggingFace eager attention |
| FP16 MLX | 177.96 | 4.98 | 999 | 3.5× faster than HF MPS |
| **MLX 4-bit** | **334.77** | **2.42** | **298** | Best overall; 1.88× vs MLX FP16 |

### 4. What this means for the project

For Apple Silicon development and testing, `benchmark_mlx.py` with
`mlx-lm` is the representative benchmark. The GGUF_TORCH numbers from
`benchmark_gguf.py` should be treated as smoke-test-only on Mac (as
noted in `docs/mps-gguf-findings.md`).

For the 7B model comparison on CUDA, `benchmark_gguf.py` with
GGUF_TRITON remains the correct path. MLX is Apple-only.

## Script and dependencies

- Script: `benchmark_mlx.py`
- Shared utilities: `bench_utils.benchmark_mlx_model()` (MLX counterpart
  of `benchmark()`)
- Dependencies: `pip install mlx-lm` (pulls `mlx` and `mlx-metal`)
- Models: downloaded from HuggingFace Hub on first run, cached in
  `~/.cache/huggingface/`
