"""Generate a PDF report for the QLoRA quantization benchmark results."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ── Load results ──
with open("results.json") as f:
    results = json.load(f)

configs = [r["config"] for r in results]
short_names = ["BF16\n(baseline)", "NF4\n4-bit", "QLoRA\n(NF4+LoRA)"]
colors = ["#2196F3", "#FF9800", "#4CAF50"]

param_mb = [r["param_memory_mb"] for r in results]
gpu_mb = [r["gpu_memory_mb"] for r in results]
ppl = [r["perplexity"] for r in results]
tps = [r["tokens_per_sec"] for r in results]
latency = [r["latency_s"] for r in results]
samples = [r["sample"] for r in results]

baseline_mb = param_mb[0]
size_pct = [m / baseline_mb * 100 for m in param_mb]

# ── PDF Report ──
with PdfPages("quantization_report.pdf") as pdf:

    # ====== PAGE 1: Title + Summary ======
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    # Title block
    fig.text(0.5, 0.88, "LLM Quantization Benchmark Report",
             ha="center", fontsize=24, fontweight="bold", color="#1a1a2e")
    fig.text(0.5, 0.83, "Qwen3-14B  |  NF4 Quantization + QLoRA",
             ha="center", fontsize=14, color="#555555")
    fig.text(0.5, 0.79, "March 2026",
             ha="center", fontsize=11, color="#888888")

    # Divider
    line = plt.Line2D([0.1, 0.9], [0.76, 0.76], color="#cccccc", linewidth=1.5)
    fig.add_artist(line)

    # Overview
    overview = """Overview

This report evaluates the impact of post-training quantization (PTQ) on the Qwen3-14B language model.
We compare three configurations:

  1. BF16 Baseline — Full-precision model (28.2 GB)
  2. NF4 4-bit (BitsAndBytes) — NormalFloat 4-bit quantization with double quantization (9.3 GB)
  3. QLoRA — NF4 quantization + Low-Rank Adaptation fine-tuning on WikiText-2 (12.4 GB)

Hardware: NVIDIA L40S (48 GB VRAM)
Evaluation: Perplexity on WikiText-2 test set (4096 tokens, sliding window)"""

    fig.text(0.1, 0.72, overview, fontsize=11, verticalalignment="top",
             fontfamily="monospace", linespacing=1.6, color="#333333")

    # Key findings box
    fig.text(0.1, 0.38, "Key Findings", fontsize=14, fontweight="bold", color="#1a1a2e")

    findings = [
        ("NF4 4-bit quantization reduces model size to 33% of BF16 with only +0.51 perplexity increase.",
         "#FF9800"),
        ("GPU memory drops from 28.2 GB to 9.5 GB — enabling deployment on much smaller hardware.",
         "#FF9800"),
        ("QLoRA fine-tuning (100 steps) reduced perplexity to 10.17, but this reflects domain adaptation\n"
         "    to the evaluation set (WikiText-2), not general model improvement.",
         "#4CAF50"),
        ("Inference speed decreases with quantization due to on-the-fly dequantization overhead\n"
         "    in the BitsAndBytes library (not hardware-native INT4).",
         "#F44336"),
    ]
    y = 0.34
    for text, color in findings:
        fig.text(0.12, y, "●", fontsize=12, color=color, verticalalignment="top")
        fig.text(0.14, y, text, fontsize=10, verticalalignment="top",
                 color="#333333", linespacing=1.5)
        y -= 0.07

    # Method box
    fig.text(0.1, 0.08, "Method: Post-Training Quantization (PTQ) — no calibration data, "
             "round-to-nearest with NormalFloat data type",
             fontsize=9, color="#888888", style="italic")

    pdf.savefig(fig)
    plt.close()

    # ====== PAGE 2: Charts ======
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.95, "Benchmark Results", ha="center", fontsize=18,
             fontweight="bold", color="#1a1a2e")

    gs = gridspec.GridSpec(2, 2, left=0.1, right=0.95, top=0.88, bottom=0.08,
                           hspace=0.35, wspace=0.3)

    # Chart 1: Model Size
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(short_names, [m / 1024 for m in param_mb], color=colors, edgecolor="white", width=0.6)
    ax1.set_ylabel("Parameter Memory (GB)", fontsize=10)
    ax1.set_title("Model Size", fontsize=12, fontweight="bold", pad=10)
    ax1.spines[["top", "right"]].set_visible(False)
    for bar, pct in zip(bars1, size_pct):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{pct:.0f}%", ha="center", fontsize=10, fontweight="bold", color="#555")

    # Chart 2: Perplexity
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(short_names, ppl, color=colors, edgecolor="white", width=0.6)
    ax2.set_ylabel("Perplexity (↓ better)", fontsize=10)
    ax2.set_title("Perplexity (WikiText-2)", fontsize=12, fontweight="bold", pad=10)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylim(0, max(ppl) * 1.2)
    for bar, val in zip(bars2, ppl):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{val:.2f}", ha="center", fontsize=10, fontweight="bold", color="#555")

    # Chart 3: Tokens/sec
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(short_names, tps, color=colors, edgecolor="white", width=0.6)
    ax3.set_ylabel("Tokens / second (↑ better)", fontsize=10)
    ax3.set_title("Generation Throughput", fontsize=12, fontweight="bold", pad=10)
    ax3.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars3, tps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", fontsize=10, fontweight="bold", color="#555")

    # Chart 4: GPU Memory
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.bar(short_names, [m / 1024 for m in gpu_mb], color=colors, edgecolor="white", width=0.6)
    ax4.set_ylabel("GPU Memory (GB)", fontsize=10)
    ax4.set_title("Peak GPU Memory Usage", fontsize=12, fontweight="bold", pad=10)
    ax4.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars4, [m / 1024 for m in gpu_mb]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f} GB", ha="center", fontsize=10, fontweight="bold", color="#555")

    pdf.savefig(fig)
    plt.close()

    # ====== PAGE 3: Size vs Performance Tradeoff + Table ======
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.95, "Size–Performance Tradeoff", ha="center", fontsize=18,
             fontweight="bold", color="#1a1a2e")

    # Scatter: size vs perplexity
    ax = fig.add_axes([0.12, 0.52, 0.76, 0.38])
    for i, (x, y, name, c) in enumerate(zip(
            [m / 1024 for m in param_mb], ppl, short_names, colors)):
        ax.scatter(x, y, s=300, c=c, edgecolors="white", linewidth=2, zorder=5)
        ax.annotate(name.replace("\n", " "), (x, y),
                    textcoords="offset points", xytext=(15, 10),
                    fontsize=10, fontweight="bold", color=c)
    ax.set_xlabel("Model Size (GB)", fontsize=11)
    ax.set_ylabel("Perplexity", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.3)

    # Arrow annotation
    ax.annotate("", xy=(param_mb[1]/1024, ppl[1]),
                xytext=(param_mb[0]/1024, ppl[0]),
                arrowprops=dict(arrowstyle="->", color="#999", lw=1.5, ls="--"))
    mid_x = (param_mb[0]/1024 + param_mb[1]/1024) / 2
    mid_y = (ppl[0] + ppl[1]) / 2
    ax.text(mid_x + 1, mid_y, "67% size reduction\n+0.51 PPL", fontsize=9,
            color="#777", ha="left", style="italic")

    # Summary table
    fig.text(0.5, 0.44, "Detailed Results", ha="center", fontsize=14,
             fontweight="bold", color="#1a1a2e")

    col_labels = ["Configuration", "Param Size\n(MB)", "GPU Mem\n(MB)",
                  "Size\n(% of BF16)", "Perplexity", "PPL\nChange",
                  "Tokens/s", "Latency\n(s)"]
    table_data = []
    for i, r in enumerate(results):
        delta = r['perplexity'] - ppl[0]
        ppl_change = f"{delta:+.2f}" if i > 0 else "—"
        # Shorten config names for table
        config_name = (r["config"]
                       .replace("torch.bfloat16 (baseline)", "BF16 (baseline)")
                       .replace("QLoRA (NF4+LoRA, 100 steps)", "QLoRA (NF4+LoRA)"))
        table_data.append([
            config_name,
            f"{r['param_memory_mb']:,.0f}",
            f"{r['gpu_memory_mb']:,.0f}",
            f"{size_pct[i]:.0f}%",
            f"{r['perplexity']:.2f}",
            ppl_change,
            f"{r['tokens_per_sec']:.1f}",
            f"{r['latency_s']:.3f}",
        ])

    ax_table = fig.add_axes([0.05, 0.1, 0.9, 0.3])
    ax_table.axis("off")
    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                           loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(results)):
        bg = "#f5f5f5" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(bg)

    pdf.savefig(fig)
    plt.close()

    # ====== PAGE 4: Sample Outputs ======
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.95, "Sample Generations", ha="center", fontsize=18,
             fontweight="bold", color="#1a1a2e")

    prompt = 'Prompt: "The future of artificial intelligence is"'
    fig.text(0.08, 0.89, prompt, fontsize=11, fontweight="bold", color="#1a1a2e")

    y = 0.85
    for i, r in enumerate(results):
        fig.text(0.08, y, short_names[i].replace("\n", " "),
                 fontsize=11, fontweight="bold", color=colors[i])
        y -= 0.03
        # Wrap sample text
        sample = r["sample"]
        wrapped = []
        while len(sample) > 100:
            idx = sample[:100].rfind(" ")
            if idx == -1:
                idx = 100
            wrapped.append(sample[:idx])
            sample = sample[idx:].strip()
        wrapped.append(sample)
        for line in wrapped:
            fig.text(0.10, y, line, fontsize=9, color="#444444", fontfamily="serif")
            y -= 0.025
        y -= 0.03

    pdf.savefig(fig)
    plt.close()

    # ====== PAGE 5: Notes ======
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.95, "Notes & Caveats", ha="center", fontsize=18,
             fontweight="bold", color="#1a1a2e")

    notes = [
        "1. Perplexity was evaluated on WikiText-2 test set (4,096 tokens, sliding window, stride=256).",
        "2. QLoRA was fine-tuned on WikiText-2 train set — the improved perplexity reflects domain\n"
        "    adaptation to the evaluation distribution, NOT general model improvement.",
        "3. For a fair evaluation, QLoRA should be evaluated on a held-out corpus (e.g., C4, LAMBADA).",
        "4. BitsAndBytes NF4 uses software dequantization, causing slower inference vs. native INT4.\n"
        "    Hardware-native quantization (e.g., GPTQ with Marlin kernels) would show speed gains.",
        "5. NF4 (NormalFloat 4-bit) is information-theoretically optimal for normally-distributed\n"
        "    weights, which is why the perplexity degradation is remarkably small (+0.51).",
        "6. QLoRA adapters add ~3 GB on top of NF4 (LoRA rank=16 on all linear layers, 0.43% trainable).",
    ]
    y = 0.85
    for note in notes:
        n_lines = note.count("\n") + 1
        fig.text(0.10, y, note, fontsize=11, color="#444444", linespacing=1.6)
        y -= 0.035 * n_lines + 0.02

    # Footer
    fig.text(0.5, 0.03, "Generated with matplotlib  |  Model: Qwen/Qwen3-14B  |  GPU: NVIDIA L40S 48GB",
             ha="center", fontsize=8, color="#aaaaaa")

    pdf.savefig(fig)
    plt.close()

print("Report saved to quantization_report.pdf")
