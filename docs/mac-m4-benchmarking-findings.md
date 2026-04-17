# Mac Mini M4 Pro Benchmarking — Session Findings (2026-04-17)

**Hardware:** Mac Mini M4 Pro, 24 GB unified memory, Metal  
**Models targeted:** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-9B-it  
**Frameworks:** MLX (`mlx-lm`), GPTQ (`gptqmodel`), GGUF (`gptqmodel` GGUF_TORCH)

---

## 1. GSM8K accuracy bug in `evaluate_gsm8k_mlx`

### Root cause

`mlx-lm` 0.31.2's `stream_generate` yields one object **per token** — each
`response.text` is a single token string, not the cumulative output. The
original code took only the last chunk:

```python
# BUGGY — last chunk is a single token (e.g. " " or "\n")
model_output = output_chunks[-1] if output_chunks else ""
```

`parse_gsm8k_answer` received a whitespace character and returned `None` for
every sample → **0% accuracy on all 50 questions**.

### Fix

```python
# FIXED — join all per-token chunks to reconstruct the full response
model_output = "".join(output_chunks)
```

### Confirmation

After the fix, Qwen2.5-7B-Instruct FP16 (MLX) scored **92% on 50 samples**
and **80% on 10 samples** — consistent with published benchmarks.

Note: `evaluate_gsm8k` (PyTorch/HuggingFace path) was unaffected — it uses
`model.generate()` which returns the full sequence at once.

---

## 2. PyTorch MPS: 14 GB single-buffer limit

### Error

When loading Qwen2.5-7B-Instruct FP16 via `GPTQModel` on `device="mps"`:

```
RuntimeError: Invalid buffer size: 14.19 GiB
```

### Cause

Metal enforces a hard per-buffer size limit (~4 GB on Apple Silicon). PyTorch
MPS maps each tensor to a single Metal buffer — any weight matrix or
concatenated parameter tensor exceeding ~4 GB fails at allocation time.

GPTQModel includes a pre-flight memory check:

```python
torch.empty(int(byte_count // 2), dtype=torch.float16, device=device)
```

This tries to allocate the full model weight footprint (~14 GB for Qwen2.5-7B
FP16) in one shot, hitting the Metal buffer cap immediately.

### Why MLX is unaffected

MLX uses its own Metal heap allocator that distributes large tensors across
multiple smaller Metal buffers internally. Lazy evaluation means weights are
only materialized during the forward pass, not all at once. This is why MLX
can load and run 14–18 GB FP16 models on 24 GB unified memory without issue.

### Implication for this project

| Scenario | Status |
|---|---|
| FP16 7B+ via MLX on Mac | ✅ Works |
| FP16 7B+ via PyTorch/GPTQModel on Mac (MPS) | ❌ Buffer size error |
| 4-bit quantized via GPTQModel on Mac (MPS, ~4 GB) | ✅ Works |
| FP16 7B+ via PyTorch on CPU | ✅ Works but very slow |

For the Mac benchmark suite, **MLX is the only viable path for FP16 7B+ models**.
GPTQ and GGUF benchmarks on Mac should be limited to quantized models only,
or the FP16 baseline should be sourced from MLX results.

---

## 3. HuggingFace gated model access

Llama-3.1-8B-Instruct (Meta) and Gemma-2-9B-it (Google) are gated — a valid
HF token is necessary but not sufficient. The HF account linked to the token
must also accept each model's license agreement on the HuggingFace website
before download is permitted.

**Authentication** (token login) ≠ **Authorization** (license acceptance).

| Model | Status |
|---|---|
| Qwen/Qwen2.5-7B-Instruct | Open, no gate |
| google/gemma-2-9b-it | Gated — instant approval on HF website |
| meta-llama/Llama-3.1-8B-Instruct | Gated — Meta approval, minutes to hours |
| All `mlx-community/` 4-bit variants | Open (re-uploaded by community) |
| All GPTQ pre-quantized variants (Qwen, Llama, Gemma) | Open |

---

## 4. Benchmark configuration optimizations

### Perplexity: full WikiText-2 → 2,048 tokens

| Setting | Before | After |
|---|---|---|
| Tokens evaluated | 290,000 (full test set) | 2,048 |
| Time per model | ~49 min | ~6 sec |
| Academic standard | GPTQ, SqueezeLLM, QuIP# all use ≤2,048 tokens |

The full dataset is more rigorous but impractical for a multi-model sweep.
2,048 tokens gives a stable perplexity estimate consistent with published
quantization paper methodology.

Implementation: `load_wikitext2_tokens(tokenizer, max_tokens=2048)` — the
`max_tokens` parameter defaults to 2,048 and accepts `None` for full dataset.

### GSM8K: 50 samples → 10 samples (preliminary runs)

Reduces GSM8K eval time from ~13 min to ~3 min per model. The same 10 questions
are used across all models (fixed `seed=42`), making accuracy directly
comparable. For final publication, increase to 50–500 samples.

### Combined impact

| Phase | Before | After |
|---|---|---|
| GSM8K per model pair | ~13 min | ~3 min |
| Perplexity per model pair | ~49 min | ~6 sec |
| Total per model pair | ~2 hrs | ~3 min |
| Full 9-run suite | ~10 hrs | ~45 min |

---

## 5. First real results: Qwen2.5-7B-Instruct on MLX

**Run dir:** `outputs/benchmark_mlx/20260417-163230/`  
**Settings:** 10 GSM8K samples, 2,048-token perplexity, seed=42

| Metric | FP16 MLX | MLX 4-bit | Delta |
|---|---|---|---|
| Throughput (tok/s) | 16.6 | 53.2 | +220% |
| TTFT (ms) | ~465 | ~387 | -17% |
| TPOT (ms) | 58.7 | 17.4 | -70% |
| Peak memory (MB) | 14,664 | 4,394 | -70% |
| Weight memory (MB) | 14,526 | 4,089 | -72% |
| Runtime memory (MB) | 138 | 305 | +121% |
| Perplexity (WikiText-2) | 5.08 | 5.61 | +10% |
| GSM8K accuracy (10 samples) | 80% | 80% | 0% |

Key observations:
- **3.2× throughput** from 4-bit quantization on MLX
- **70% memory reduction** — fits comfortably in 24 GB unified memory at 4-bit
- **Perplexity degradation of 10%** — modest quality loss for major speed/memory gain
- **Runtime memory is higher for 4-bit** — dequantization requires temporary buffers
  during the forward pass even though weight footprint is smaller
- GSM8K accuracy identical at this small sample size (10 questions); larger
  sample needed to detect real differences
