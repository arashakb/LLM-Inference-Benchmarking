# 2-Bit Quantization Plan

## Pre-quantized 2-bit models available on HuggingFace

### GPTQ 2-bit

| Model | Model ID | Quantizer |
|-------|----------|-----------|
| Qwen2.5-7B-Instruct | `kaitchup/Qwen2.5-7B-Instruct-gptqmodel-2bit` | kaitchup |
| Llama-3.1-8B-Instruct | `kaitchup/Llama-3.1-8B-Instruct-gptqmodel-2bit` | kaitchup |
| Gemma-2-9B-it | `irish-quant/google-gemma-2-9b-it-2bit` | irish-quant |

### GGUF Q2_K

| Model | Model ID | Notes |
|-------|----------|-------|
| Qwen2.5-7B-Instruct | `bartowski/Qwen2.5-7B-Instruct-GGUF` | Q2_K file inside multi-quant repo |
| Llama-3.1-8B-Instruct | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | Q2_K file inside multi-quant repo |
| Gemma-2-9B-it | `bartowski/gemma-2-9b-it-GGUF` | Q2_K file inside multi-quant repo |

### MLX 2-bit

No vanilla 2-bit models on `mlx-community` for our target models.
Must self-quantize (see below).

---

## Self-quantization commands

### MLX 2-bit (Apple Silicon — works)

```bash
# Qwen2.5-7B
mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct \
  --q-bits 2 --q-group-size 64 \
  -o models/Qwen2.5-7B-Instruct-mlx-2bit

# Llama-3.1-8B (requires HF auth for gated model)
mlx_lm.convert --hf-path meta-llama/Llama-3.1-8B-Instruct \
  --q-bits 2 --q-group-size 64 \
  -o models/Llama-3.1-8B-Instruct-mlx-2bit

# Gemma-2-9B (requires HF auth for gated model)
mlx_lm.convert --hf-path google/gemma-2-9b-it \
  --q-bits 2 --q-group-size 64 \
  -o models/Gemma-2-9B-it-mlx-2bit
```

The benchmark script (`benchmark_mlx.py` index 3-5) auto-quantizes if the
local model directory doesn't exist.

### GPTQ 2-bit (CUDA only)

```python
from gptqmodel import GPTQModel
from gptqmodel.quantization import QuantizeConfig

qconfig = QuantizeConfig(bits=2, group_size=128)
model = GPTQModel.load("Qwen/Qwen2.5-7B-Instruct", qconfig)
model.quantize(calibration_dataset, tokenizer=tokenizer)
model.save("models/Qwen2.5-7B-Instruct-GPTQ-2bit")
```

Requires CUDA — FP16 model loading fails on MPS (buffer size limit).

### GGUF Q2_K (CUDA only)

```python
from gptqmodel import GPTQModel
from gptqmodel.quantization import GGUFConfig

qconfig = GGUFConfig(bits=2, format="q2_k")
model = GPTQModel.load("Qwen/Qwen2.5-7B-Instruct", qconfig)
model.quantize(calibration=None, tokenizer=tokenizer, backend=BACKEND.GGUF_TRITON)
model.save("models/Qwen2.5-7B-Instruct-GGUF-Q2_K")
```

Requires CUDA — both FP16 loading and GGUF_TRITON backend need CUDA.

---

## Platform compatibility matrix

| Method | Mac (MLX) | Mac (MPS/PyTorch) | CUDA |
|--------|:---------:|:-----------------:|:----:|
| MLX 2-bit self-quantize | ✅ | N/A | N/A |
| MLX 2-bit inference | ✅ | N/A | N/A |
| GPTQ 2-bit pre-quantized inference | N/A | ❌ (Placeholder error) | ✅ |
| GPTQ 2-bit self-quantize | N/A | ❌ (FP16 buffer limit) | ✅ |
| GGUF Q2_K inference (GGUF_TORCH) | N/A | ❌ (format unsupported) | ✅ |
| GGUF Q2_K inference (native llama.cpp) | ✅ (slow) | N/A | ✅ |
| GGUF Q2_K self-quantize | N/A | ❌ (FP16 buffer limit) | ✅ |

---

## Benchmark run commands

### Mac (MLX only)

```bash
python3 benchmark_mlx.py 3   # Qwen2.5-7B 2-bit
python3 benchmark_mlx.py 4   # Llama-3.1-8B 2-bit
python3 benchmark_mlx.py 5   # Gemma-2-9B 2-bit
```

### CUDA (all frameworks)

```bash
# GPTQ 2-bit
python3 benchmark_quantize.py 3   # Qwen2.5-7B
python3 benchmark_quantize.py 4   # Llama-3.1-8B
python3 benchmark_quantize.py 5   # Gemma-2-9B

# GGUF Q2_K
python3 benchmark_gguf.py 3      # Qwen2.5-7B
python3 benchmark_gguf.py 4      # Llama-3.1-8B
python3 benchmark_gguf.py 5      # Gemma-2-9B
```

---

## Expected quality impact

2-bit quantization significantly degrades model quality compared to 4-bit.
Expect:

- **Perplexity**: +30-100% increase over FP16 (vs +10% for 4-bit)
- **GSM8K accuracy**: likely 20-50% drop from FP16
- **Memory**: ~50% smaller than 4-bit (~2 GB for 7B models)
- **Speed**: faster than 4-bit on MLX (less data to move), but quality
  tradeoff makes it unsuitable for most production use cases

The benchmark results will quantify these tradeoffs for our specific models.
