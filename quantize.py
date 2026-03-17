"""
QLoRA Quantization Benchmark
==============================
Compares FP16 baseline vs BnB NF4 quantization vs QLoRA (NF4 + LoRA fine-tune)
on Qwen3-8B. Measures perplexity, latency, memory, and model size.

Flow:
  1. FP16 baseline        → eval perplexity & latency
  2. NF4 4-bit (BnB)      → eval perplexity & latency (quantization-only degradation)
  3. QLoRA fine-tuned      → fine-tune on small dataset, then eval (recovery after fine-tuning)
"""

import os
import time
import json
import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

# ──────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, text, max_length=512, stride=256):
    """Sliding-window perplexity."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls, n_tokens = [], 0
    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[:, begin:end]
        target = chunk.clone()
        if begin > 0:
            target[:, :-(end - begin - stride)] = -100
        loss = model(chunk, labels=target).loss
        n_valid = (target != -100).sum().item()
        nlls.append(loss.float() * n_valid)
        n_tokens += n_valid
        if end == seq_len:
            break

    return torch.exp(torch.stack(nlls).sum() / n_tokens).item()


@torch.no_grad()
def measure_latency(model, tokenizer, prompt="The future of artificial intelligence is",
                    max_new_tokens=100, n_runs=3):
    """Average generation latency & tokens/sec."""
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # warmup
    model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    n_gen = out.shape[1] - inputs["input_ids"].shape[1]
    avg = sum(times) / len(times)
    return {"latency_s": round(avg, 3), "tokens_per_sec": round(n_gen / avg, 1)}


def get_model_memory(model):
    """Model parameter memory in MB."""
    return sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 ** 2)


def get_gpu_memory_used(device):
    """Current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated(device) / (1024 ** 2)


def sample_generate(model, tokenizer, prompt="The future of artificial intelligence is"):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────

def benchmark(model, tokenizer, eval_text, label, max_length=512):
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {label}")
    print(f"{'='*60}")

    device = next(model.parameters()).device
    result = {"config": label}

    # Size
    result["param_memory_mb"] = round(get_model_memory(model), 1)
    result["gpu_memory_mb"] = round(get_gpu_memory_used(device), 1)
    print(f"  Param memory: {result['param_memory_mb']} MB")
    print(f"  GPU memory:   {result['gpu_memory_mb']} MB")

    # Perplexity
    print("  Evaluating perplexity...")
    ppl = evaluate_perplexity(model, tokenizer, eval_text, max_length=max_length)
    result["perplexity"] = round(ppl, 2)
    print(f"  Perplexity:   {ppl:.2f}")

    # Latency
    print("  Measuring latency...")
    lat = measure_latency(model, tokenizer)
    result.update(lat)
    print(f"  Latency:      {lat['latency_s']}s ({lat['tokens_per_sec']} tok/s)")

    # Sample
    sample = sample_generate(model, tokenizer)
    result["sample"] = sample
    print(f"  Sample:       {sample[:120]}...")

    return result


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QLoRA Quantization Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-eval-tokens", type=int, default=4096)
    parser.add_argument("--qlora-steps", type=int, default=100, help="QLoRA fine-tuning steps")
    parser.add_argument("--skip-qlora", action="store_true", help="Skip QLoRA fine-tuning")
    args = parser.parse_args()

    device = args.device
    print(f"\n{'#'*60}")
    print(f"  QLoRA Quantization Benchmark")
    print(f"  Model:  {args.model}")
    print(f"  Device: {device}")
    print(f"{'#'*60}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Eval dataset ---
    print("\nLoading eval dataset (wikitext-2)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join(ds["text"])
    toks = tokenizer(eval_text, return_tensors="pt")["input_ids"]
    n = min(args.max_eval_tokens, toks.shape[1])
    eval_text = tokenizer.decode(toks[0, :n], skip_special_tokens=True)
    print(f"  Using {n} tokens for evaluation")

    results = []

    # ═══════════════════════════════════════════════════════════
    # Config 1: FP16 baseline
    # ═══════════════════════════════════════════════════════════
    print("\n[1/3] Loading baseline model...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=device, trust_remote_code=True,
    )
    baseline_dtype = next(model_fp16.parameters()).dtype
    results.append(benchmark(model_fp16, tokenizer, eval_text, f"{baseline_dtype} (baseline)"))
    del model_fp16
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # Config 2: NF4 quantization (no fine-tuning)
    # ═══════════════════════════════════════════════════════════
    print("\n[2/3] Loading NF4 4-bit model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # nested quantization
    )
    model_nf4 = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map=device, trust_remote_code=True,
    )
    results.append(benchmark(model_nf4, tokenizer, eval_text, "NF4 4-bit (BnB)"))

    # ═══════════════════════════════════════════════════════════
    # Config 3: QLoRA (NF4 + LoRA fine-tune)
    # ═══════════════════════════════════════════════════════════
    if not args.skip_qlora:
        print(f"\n[3/3] QLoRA fine-tuning ({args.qlora_steps} steps)...")

        # Prepare for k-bit training
        model_qlora = prepare_model_for_kbit_training(model_nf4)

        # LoRA config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model_qlora = get_peft_model(model_qlora, lora_config)

        trainable, total = model_qlora.get_nb_trainable_parameters()
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # Training dataset (small subset of wikitext for demo)
        train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        train_ds = train_ds.filter(lambda x: len(x["text"]) > 50)
        train_ds = train_ds.select(range(min(1000, len(train_ds))))

        training_args = TrainingArguments(
            output_dir="./qlora_output",
            num_train_epochs=1,
            max_steps=args.qlora_steps,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            optim="paged_adamw_8bit",
        )

        trainer = SFTTrainer(
            model=model_qlora,
            train_dataset=train_ds,
            args=training_args,
            processing_class=tokenizer,
        )

        trainer.train()
        print("  Fine-tuning complete.")

        results.append(benchmark(model_qlora, tokenizer, eval_text, f"QLoRA (NF4+LoRA, {args.qlora_steps} steps)"))

        del model_qlora
    else:
        print("\n[3/3] Skipping QLoRA (--skip-qlora)")

    del model_nf4
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'#'*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'#'*80}")
    print(f"{'Config':<35} {'Param MB':>10} {'GPU MB':>10} {'PPL':>8} {'Tok/s':>8}")
    print(f"{'-'*80}")
    baseline_ppl = results[0]["perplexity"]
    baseline_mem = results[0]["param_memory_mb"]
    for r in results:
        size_ratio = r["param_memory_mb"] / baseline_mem * 100
        ppl_delta = r["perplexity"] - baseline_ppl
        delta_str = f"({ppl_delta:+.2f})" if r["config"] != "FP16 (baseline)" else ""
        print(f"{r['config']:<35} {r['param_memory_mb']:>8.1f} MB {r['gpu_memory_mb']:>8.1f} MB "
              f"{r['perplexity']:>7.2f} {r['tokens_per_sec']:>7.1f}")
        if r["config"] != "FP16 (baseline)":
            print(f"{'':35} {'size':>10}: {size_ratio:.0f}% of FP16   PPL change: {ppl_delta:+.2f}")
    print(f"{'#'*80}\n")

    # Save
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved to results.json")


if __name__ == "__main__":
    main()
