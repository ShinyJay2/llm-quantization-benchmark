# DIY LLM Quantization

From-scratch implementation of weight quantization for LLMs.

## What's implemented

| Method | Description |
|--------|-------------|
| Absmax (per-tensor) | Symmetric quantization: scale = max(abs(w)) / (2^(b-1)-1) |
| Absmax (per-channel) | Same, but one scale per output channel (row) |
| Zero-point | Asymmetric quantization: maps [min, max] → [0, 2^b-1] |

Each method is tested at **8-bit** and **4-bit**.

## Metrics

- **Perplexity** (wikitext-2) — accuracy degradation
- **Tokens/sec** — generation speed
- **Peak GPU memory** — memory footprint
- **Weight MSE** — quantization error vs FP16

## Usage

```bash
# Full benchmark (perplexity + latency)
python quantize.py

# Quick run (skip perplexity, just latency + generation)
python quantize.py --skip-perplexity

# Use a different model
python quantize.py --model Qwen/Qwen2.5-0.5B

# Choose GPU
python quantize.py --device cuda:1
```
