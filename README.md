<p align="center">
  <h1 align="center">LLM Quantization Benchmark</h1>
  <p align="center">
    <strong>How much can you compress a 14B-parameter LLM before it breaks?</strong>
  </p>
  <p align="center">
    BF16 vs NF4 4-bit vs QLoRA on <code>Qwen3-14B</code>
  </p>
</p>

---

## TL;DR

> **NF4 quantization compresses Qwen3-14B to 33% of its original size with only +0.51 perplexity increase.** GPU memory drops from 28 GB to 9.5 GB — turning a multi-GPU model into a single-GPU model.

---

## Results

### Benchmark Summary

| Configuration | Param Size | GPU Memory | Size (% of BF16) | Perplexity | PPL Change | Tokens/s | Latency |
|:---|---:|---:|:---:|---:|---:|---:|---:|
| **BF16 (baseline)** | 28,168 MB | 28,168 MB | 100% | 13.95 | — | 22.6 | 4.43s |
| **NF4 4-bit (BnB)** | 9,268 MB | 9,503 MB | **33%** | 14.46 | +0.51 | 17.0 | 5.89s |
| **QLoRA (NF4+LoRA)** | 12,359 MB | 12,663 MB | 44% | 10.17 | -3.78* | 10.1 | 9.37s |

> \* QLoRA perplexity improvement reflects domain adaptation to the evaluation set (WikiText-2), not general model improvement. See [Notes](#notes--caveats).

### Size vs Performance Tradeoff

```
Perplexity
    15 ┤
       │                                                    ● BF16 (baseline)
    14 ┤  ● NF4 4-bit  ·····························
       │                    67% size reduction, +0.51 PPL
    13 ┤
       │
    12 ┤
       │
    11 ┤
       │          ● QLoRA (NF4+LoRA)
    10 ┤
       └──────┬──────┬──────┬──────┬──────┬──────┬──────
             10     12     15     18     20     25     28
                              Model Size (GB)
```

### Memory Reduction

```
BF16 (baseline)  ████████████████████████████████████████  28.2 GB (100%)
NF4 4-bit        █████████████                             9.3 GB  (33%)
QLoRA            █████████████████                        12.4 GB  (44%)
```

---

## Hardware & Environment

| Component | Specification |
|:---|:---|
| **GPU** | NVIDIA L40S (48 GB VRAM, Ada Lovelace) |
| **GPU Count** | 3x L40S (benchmark ran on a single GPU) |
| **CUDA** | 12.1 |
| **PyTorch** | 2.5.1+cu121 |
| **Transformers** | 5.3.0 |
| **BitsAndBytes** | 0.49.2 |
| **PEFT** | 0.14.0 |
| **TRL** | 0.29.0 |
| **Python** | 3.10.12 |
| **OS** | Ubuntu (Linux 6.5.0-44-generic) |

### GPU Specs — NVIDIA L40S

| Spec | Value |
|:---|:---|
| Architecture | Ada Lovelace |
| VRAM | 48 GB GDDR6 with ECC |
| Memory Bandwidth | 864 GB/s |
| FP16 Performance | 181.05 TFLOPS |
| INT8 Performance | 362.05 TOPS |
| TDP | 350W |

---

## What's Compared

### 1. BF16 Baseline
Full-precision model loaded in `bfloat16`. No quantization. This is the reference point.

### 2. NF4 4-bit (BitsAndBytes)
Post-training quantization using **NormalFloat 4-bit** (NF4) data type from the [QLoRA paper](https://arxiv.org/abs/2305.14314). NF4 is information-theoretically optimal for normally-distributed weights. We also enable **double quantization** (quantizing the quantization constants) for additional savings.

- No calibration data needed
- No fine-tuning
- Uses `bitsandbytes` library for on-the-fly dequantization

### 3. QLoRA (NF4 + LoRA Fine-tuning)
NF4 quantized model with **Low-Rank Adaptation** (LoRA) adapters fine-tuned on top.

| QLoRA Hyperparameter | Value |
|:---|:---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | 64.2M / 14.8B (0.43%) |
| Training steps | 100 |
| Batch size | 4 (× 2 gradient accumulation) |
| Learning rate | 2e-4 |
| Optimizer | paged_adamw_8bit |
| Dataset | WikiText-2 train (1000 samples) |

---

## Evaluation

| Metric | Method |
|:---|:---|
| **Perplexity** | Sliding window on WikiText-2 test set (4,096 tokens, window=512, stride=256) |
| **Throughput** | Average tokens/sec over 3 generation runs (100 tokens each) |
| **Latency** | Average wall-clock time for 100-token generation |
| **Memory** | `torch.cuda.memory_allocated()` after model loading |

---

## Sample Generations

**Prompt:** *"The future of artificial intelligence is"*

**BF16 (baseline):**
> The future of artificial intelligence is a topic of both promise and peril, a landscape where the potential for transformative advancements is matched only by the risks they pose. As we stand on the brink of a new era, the question of whether AI will lead to ut or dystopia is not merely a matter of speculation but a critical that demands careful consideration...

**NF4 4-bit:**
> The future of artificial intelligence is a topic of both promise and peril, a landscape where the potential for transformative advancements is matched only by the risks they pose. As we stand on the brink of a new era, the question of whether AI will be our salvation or our downfall is not merely a matter of technological capability but a profound ethical and societal challenge...

**QLoRA (NF4+LoRA):**
> The future of artificial intelligence is a topic of both excitement and uncertainty. As AI continues to evolve, it is likely to become more integrated into our daily and work. This could lead to significant changes in various industries, including healthcare, education, and transportation...

---

## Usage

```bash
# Full benchmark (BF16 + NF4 + QLoRA)
CUDA_VISIBLE_DEVICES=2 python quantize.py

# Skip QLoRA fine-tuning (faster, just BF16 vs NF4)
CUDA_VISIBLE_DEVICES=2 python quantize.py --skip-qlora

# Use a different model
CUDA_VISIBLE_DEVICES=0 python quantize.py --model Qwen/Qwen2.5-3B-Instruct

# More fine-tuning steps
CUDA_VISIBLE_DEVICES=1 python quantize.py --qlora-steps 500

# Generate PDF report from results
python generate_report.py
```

### Requirements

```
torch>=2.5
transformers>=5.0
bitsandbytes>=0.46.1
peft>=0.14
trl>=0.29
datasets
matplotlib
```

---

## Project Structure

```
.
├── quantize.py                 # Main benchmark script
├── generate_report.py          # PDF report generator
├── results.json                # Raw benchmark results
├── quantization_report.pdf     # Generated PDF report
└── README.md
```

---

## Notes & Caveats

1. **QLoRA perplexity is misleading.** We fine-tuned on WikiText-2 train and evaluated on WikiText-2 test. The -3.78 PPL improvement reflects domain adaptation, not general model improvement. A fair evaluation would use a held-out corpus (e.g., C4, LAMBADA).

2. **NF4 inference is slower than BF16**, not faster. BitsAndBytes uses software dequantization — weights are stored in 4-bit but dequantized to FP16 on every forward pass. Hardware-native quantization (e.g., GPTQ with Marlin kernels, or AWQ with TinyGemm) would show actual speed gains.

3. **NF4 is optimal for normal distributions.** LLM weights are approximately normally distributed, which is why NF4 (NormalFloat) achieves remarkably low quantization error compared to uniform INT4.

4. **QLoRA adds ~3 GB** on top of the NF4 base due to the FP16 LoRA adapter weights (rank=16 across all linear layers, 0.43% of total parameters).

---

## References

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
- [LLM.int8(): 8-bit Matrix Multiplication](https://arxiv.org/abs/2208.07339) (Dettmers et al., 2022)
- [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)

---

<p align="center">
  <sub>Built with Claude Code</sub>
</p>
