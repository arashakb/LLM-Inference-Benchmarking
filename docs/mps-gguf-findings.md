# MPS + GGUF_TORCH: Not a Fair PyTorch-vs-llama.cpp Stand-In

**Context:** First end-to-end run of `benchmark_gguf.py` on Apple Silicon
(MPS) after wiring up MPS device detection, eager-attention workaround,
`.to(device)` loader, and the `BACKEND.GGUF_TRITON → BACKEND.GGUF_TORCH`
fallback. Run dir:
`outputs/benchmark_gguf/20260410-160835/`.

## Run numbers

Model: `Qwen/Qwen2.5-0.5B-Instruct` (tiny substitute for local smoke
testing; the real comparison is meant to run on the 7B sibling).
Device: `mps`. 10 prompts × 128 new tokens. 2 warmup runs. Greedy decode.

| Metric | FP16 | GGUF Q4_K_M | Delta |
|---|---|---|---|
| Throughput (tok/s) | **50.18** | **3.92** | **−92.2%** |
| TTFT (ms) | 50.51 | 282.48 | +459.2% |
| TPOT (ms) | 19.64 | 255.14 | **+1199%** |
| Latency / prompt (ms) | 2,449 | 32,685 | +1234.6% |
| Peak memory (MB) | 1,069.0 | 1,117.1 | +4.5% |
| Speedup | — | **0.08×** | — |
| Memory saved | — | **−4.5%** | — |

Wall times of each phase:

- FP16 benchmark: ~31 s
- GGUF quantization (cached from a previous run): skipped
- GGUF benchmark: **~6 min 34 s**

## Why the quantized path is ~13× slower

`BACKEND.GGUF_TORCH` is a pure-PyTorch GGUF kernel
(`gptqmodel.nn_modules.qlinear.gguf.GGUFTorchQuantLinear`). It advertises
`SUPPORTS_DEVICES=[DEVICE.ALL]` and `SUPPORTS_PLATFORM=[PLATFORM.ALL]`,
which is why we fell back to it on MPS after the Triton path raised
`ModuleNotFoundError: GGUFTritonKernel requires triton`. But "works on
all devices" is not the same as "fast on all devices."

On every forward pass, for every linear layer, the torch kernel:

1. Allocates a fresh scratch tensor.
2. Unpacks the 4-bit packed weights into fp16/fp32.
3. Runs the matmul.
4. Releases the dequantized copy.

On CUDA, `GGUF_TRITON` fuses this whole sequence into inlined Triton
kernels so the dequantization cost nearly vanishes. On MPS there is no
equivalent backend and the eager torch path pays allocation + copy +
reshape + matmul every single token — and it does it per layer, so a
0.5 B model with ~24 transformer blocks pays the dequant tax dozens of
times per generated token.

Net effect:

- Decode time goes from **19.64 ms/token → 255.14 ms/token** (+1199 %).
- Prefill (TTFT) goes from 50 ms → 282 ms — smaller relative hit
  because prefill amortizes the dequant across many tokens at once.
- Memory actually *grows* by 4.5 %: the packed weights stay resident
  *and* dequant scratch buffers are allocated repeatedly on top.

## Why this breaks the "PyTorch vs llama.cpp" comparison on Mac

The project goal in `docs/batch-size-notes.md` is an honest comparison
between the PyTorch + HF stack and llama.cpp as two different engines.
On CUDA that reduces to:

- FP16 via `transformers` + SDPA/Flash-Attention → representative of
  PyTorch server workloads.
- Q4_K_M via `gptqmodel` + `GGUF_TRITON` → representative of the fused
  quantized path you'd actually deploy.

On Apple Silicon none of that holds:

- `GGUF_TRITON` requires Triton (CUDA-only).
- `GGUF_TORCH` is a functional fallback but is not what anyone would
  deploy — its per-token cost makes any serving decision you derive
  from it misleading.
- `GGUF_CPP_CPU` requires `llama-cpp-python` and runs on CPU, not MPS.
- There is no `GGUF_METAL` backend in gptqmodel today.

So the comparison we can do on this box is **FP16 PyTorch on MPS vs.
PyTorch-implemented GGUF dequant on MPS**, and that is not the same
thing as "PyTorch vs llama.cpp on Apple Silicon." llama.cpp has its
own Metal backend (`ggml-metal`) with hand-written Metal shaders, and
it is roughly an order of magnitude faster per token than
`GGUFTorchQuantLinear`. Any cross-engine conclusion drawn from this
benchmark on a Mac would be a conclusion about **gptqmodel's torch
fallback**, not about llama.cpp.

This is the same point `docs/batch-size-notes.md` was making in the
abstract: PyTorch and llama.cpp have different native turf. The MPS
run made it concrete — the numbers above show *how* far off it is when
you stay inside the PyTorch ecosystem and try to read them as an engine
comparison.

## What a fair Mac comparison would look like

Two changes would make the on-Mac run representative of the project's
stated goal:

1. **Swap the quant engine on non-CUDA devices.** Use
   `llama-cpp-python` built with `CMAKE_ARGS="-DLLAMA_METAL=on"` for the
   GGUF path, keeping `gptqmodel` only for the CUDA + Triton path. This
   is a genuinely different package — not a different `BACKEND` string
   within gptqmodel — and it needs its own loader / tokenizer /
   `generate()` wrapper.
2. **Report the device-specific backend in the result JSON and in
   `print_comparison`,** so anyone reading the numbers later can see
   whether they came from `GGUF_TRITON`, `GGUF_TORCH`, or `llama.cpp
   Metal`. Without that label, FP16 vs GGUF comparisons risk being
   taken at face value across different engines.

Neither of those is in scope for the current refinement task — they
are follow-up work.

## What *is* still valid from this run

Even though the GGUF numbers don't support a cross-engine conclusion,
several things the run *does* verify remain useful:

- MPS device detection and the `pick_device()` helper work.
- The `attn_implementation="eager"` workaround keeps `model.generate()`
  stable on MPS.
- The `.to(device)` fallback (instead of `device_map=`) avoids the
  14 GiB single-buffer allocator crash on 7B models.
- `GPTQModel.load(...)` followed by `quant_model.model.to(device)`
  correctly places embeddings / layernorms / `lm_head` on MPS, fixing
  the "Placeholder storage has not been allocated" error.
- `tpot_skipped` never triggered (10/10 prompts generated ≥ 2 tokens).
- JSON dump and file-logging land cleanly; tracebacks from any future
  crash will appear inline in `run.log` via `sys.excepthook`.
- The FP16 numbers (50.18 tok/s on Qwen2.5-0.5B, MPS, eager attention)
  are a clean baseline and are directly comparable to other PyTorch
  FP16 runs on the same hardware.

The action item is: **do not use these GGUF numbers as an engine
comparison.** Treat them as a smoke-test confirmation that the pipeline
runs end-to-end on Apple Silicon and nothing more.
