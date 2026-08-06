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
MULTI = os.path.join(DIR, "main_multi")
AUTO = os.path.join(DIR, "main_auto")

# (M, N, d) — N 受 smem 限制：N*4 + 32 ≲ 48KB → N ≲ 12k
SHAPES = [
    (128, 128, 64),
    (256, 256, 64),
    (512, 512, 64),
    (1024, 1024, 64),
    (2048, 2048, 64),
    (4096, 4096, 64),
    (512, 512, 128),
    (1024, 1024, 128),
]

ARCH = os.environ.get("CUDA_ARCH", "sm_86")
COLORS = ["#7CFC00", "#008B8B", "#32CD32", "#40E0D0"]

BINS = [
    ("single", SINGLE, ["-DFORCE_SINGLE"]),
    ("multi", MULTI, ["-DFORCE_MULTI"]),
    ("auto", AUTO, []),
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
    m = re.search(r"GPU attn time:\s*([\d.]+)\s*ms", text)
    if not m:
        raise RuntimeError(f"parse fail:\n{text}")
    return float(m.group(1))


def parse_cpu_ms(text: str) -> float:
    m = re.search(r"CPU attn time:\s*([\d.]+)\s*ms", text)
    if not m:
        raise RuntimeError(f"parse fail:\n{text}")
    return float(m.group(1))


def parse_err(text: str) -> float:
    m = re.search(r"max\|gpu-cpu\|=\s*([0-9.eE+-]+)", text)
    if not m:
        raise RuntimeError(f"parse err fail:\n{text}")
    return float(m.group(1))


def run_one(bin_path: str, M: int, N: int, d: int) -> str:
    r = subprocess.run(
        [bin_path, str(M), str(N), str(d)],
        capture_output=True,
        text=True,
        cwd=DIR,
    )
    if r.returncode not in (0, 1):
        print(r.stderr or r.stdout, file=sys.stderr)
        sys.exit(1)
    return r.stdout


def bench():
    gpu = {name: [] for name, _, _ in BINS}
    cpu_ms = []
    errs = {name: [] for name, _, _ in BINS}
    labels = []

    for M, N, d in SHAPES:
        label = f"{M}x{N}d{d}"
        labels.append(label)
        print(f"=== {label} ===")
        line = []
        for name, path, _ in BINS:
            out = run_one(path, M, N, d)
            g = parse_gpu_ms(out)
            e = parse_err(out)
            gpu[name].append(g)
            errs[name].append(e)
            line.append(f"{name}={g:.4f}(err={e:.1e})")
            if name == "single":
                cpu_ms.append(parse_cpu_ms(out))
        print("  " + "  ".join(line) + f"  cpu={cpu_ms[-1]:.4f}")
    return labels, gpu, cpu_ms, errs


def plot(labels, gpu, cpu_ms, out_path):
    x = np.arange(len(labels))
    series = [(n, gpu[n], COLORS[i]) for i, (n, _, _) in enumerate(BINS)]
    # CPU 大 shape 常 skip（0ms），仍画出来对照
    series.append(("CPU", cpu_ms, COLORS[3]))
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, (name, vals, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for b, v in zip(bars, vals):
            if v <= 0:
                continue
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.set_xlabel("shape (M x N d)")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
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

    labels, gpu, cpu_ms, errs = bench()
    out = os.path.join(DIR, "attn_bench.png")
    plot(labels, gpu, cpu_ms, out)

    # 正确性汇总
    print("\n=== max|gpu-cpu| ===")
    for name, _, _ in BINS:
        worst = max(errs[name])
        print(f"  {name}: worst={worst:.3e}")


if __name__ == "__main__":
    main()
