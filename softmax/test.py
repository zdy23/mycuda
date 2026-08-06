#!/usr/bin/env python3
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(DIR, "main.cu")
SINGLE = os.path.join(DIR, "main_single")
FUSED = os.path.join(DIR, "main_fused")

NS = [4096, 65536, 262144, 1000000, 10000000, 50000000]
ARCH = os.environ.get("CUDA_ARCH", "sm_86")
COLORS = ["#7CFC00", "#008B8B", "#32CD32", "#40E0D0"]

BINS = [
    ("single-block", SINGLE, []),
    ("multi-fused", FUSED, ["-DMULTI_FUSED"]),
]


def compile_bins():
    for name, out, flags in BINS:
        cmd = ["nvcc", "-O3", f"-arch={ARCH}", *flags, SRC, "-o", out]
        print(" ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            sys.exit(1)


def parse_gpu_ms(text: str) -> float:
    m = re.search(r"GPU softmax time:\s*([\d.]+)\s*ms", text)
    if not m:
        raise RuntimeError(f"parse fail:\n{text}")
    return float(m.group(1))


def parse_cpu_ms(text: str) -> float:
    m = re.search(r"CPU softmax time:\s*([\d.]+)\s*ms", text)
    if not m:
        raise RuntimeError(f"parse fail:\n{text}")
    return float(m.group(1))


def run_one(bin_path: str, n: int) -> str:
    r = subprocess.run([bin_path, str(n)], capture_output=True, text=True, cwd=DIR)
    if r.returncode not in (0, 1):
        print(r.stderr or r.stdout, file=sys.stderr)
        sys.exit(1)
    return r.stdout


def bench():
    gpu = {name: [] for name, _, _ in BINS}
    cpu_ms = []
    for n in NS:
        print(f"=== N={n} ===")
        line = []
        for name, path, _ in BINS:
            out = run_one(path, n)
            g = parse_gpu_ms(out)
            gpu[name].append(g)
            line.append(f"{name}={g:.4f}")
            if name == "single-block":
                cpu_ms.append(parse_cpu_ms(out))
        print("  " + "  ".join(line) + f"  cpu={cpu_ms[-1]:.4f}")
    return gpu, cpu_ms


def plot(gpu, cpu_ms, out_path):
    labels = [str(n) for n in NS]
    x = np.arange(len(labels))
    series = [(n, gpu[n], COLORS[i]) for i, (n, _, _) in enumerate(BINS)]
    series.append(("CPU", cpu_ms, COLORS[3]))
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (name, vals, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width, label=name, color=color,
            edgecolor="white", linewidth=0.5,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.set_xlabel("N")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper center", ncol=len(series), frameon=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    plt.close(fig)


def main():
    skip_compile = "--no-compile" in sys.argv
    if not skip_compile:
        compile_bins()
    else:
        for _, path, _ in BINS:
            if not os.path.isfile(path):
                print(f"missing {path}", file=sys.stderr)
                sys.exit(1)

    gpu, cpu_ms = bench()
    out = os.path.join(DIR, "softmax_bench.png")
    plot(gpu, cpu_ms, out)


if __name__ == "__main__":
    main()
