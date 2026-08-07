#!/usr/bin/env python3
"""
softmax_attn 参数扫点 + 性能画图。

用法:
    python3 bench.py                    # 用默认参数空间跑全部
    python3 bench.py --M 512 1024 2048  # 自定义 M 值
    python3 bench.py --no-rebuild       # 跳过编译

输出:
    bench_results.json   原始结果
   *.png          性能对比图
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "solve.cu"
MAIN = HERE / "main.cu"
EXE = HERE / "attn_bench"
RESULTS = HERE / "bench_results.json"

def rebuild():
    print(">> 编译", EXE.name)
    cmd = [
        "nvcc", "-O2", "-std=c++17",
        "-arch=sm_86",          # 改成你的架构, sm_70/sm_80/sm_89 ...
        "-Xptxas", "-v",
        str(SRC), str(MAIN), "-o", str(EXE),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        sys.exit(f"编译失败 (exit {r.returncode})")
    print(f"   编译完成 ({time.time()-t0:.1f}s)")

def run_one(M, N, d, ts, tpt, tpb, verify=0, warmup=3, repeat=10):
    cmd = [str(EXE), str(M), str(N), str(d), str(ts), str(tpt), str(tpb),
           str(verify), str(warmup), str(repeat)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"ms": float("inf"), "err": "\"timeout\""}
    if r.returncode != 0:
        return {"ms": float("inf"), "err": "\"crash\""}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ms": float("inf"), "err": "\"parse\""}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[10000])
    ap.add_argument("--N", type=int, nargs="+", default=[10000])
    ap.add_argument("--d", type=int, nargs="+", default=[128])
    ap.add_argument("--ts", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--tpt", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--tpb", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-rebuild", action="store_true")
    args = ap.parse_args()

    if not args.no_rebuild and EXE.exists():
        EXE.unlink()
    if not args.no_rebuild or not EXE.exists():
        rebuild()

    results = []
    # 每组 (M,N,d) 固定，扫 ts × tpt × tpb
    for M in args.M:
        for N in args.N:
            for d in args.d:
                for ts in args.ts:
                    for tpt in args.tpt:
                        for tpb in args.tpb:
                            row = run_one(M, N, d, ts, tpt, tpb,
                                          verify=0, warmup=args.warmup,
                                          repeat=args.repeat)
                            row.update(M=M, N=N, d=d, ts=ts, tpt=tpt, tpb=tpb)
                            results.append(row)
                            tag = (f"M{M} N{N} d{d} | "
                                   f"ts{ts} tpt{tpt} tpb{tpb}")
                            print(f"   {tag:42s} -> {row['ms']:7.3f} ms")

    RESULTS.write_text(json.dumps(results, indent=2))
    print(f">> 写入 {RESULTS.name}")

    try:
        plot(results)
    except Exception as e:
        print(f">> 画图失败 ({e})，跳过")

def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 所有参数组合放一张图: x = 配置全标, y = ms
    rows = sorted(results, key=lambda r: (r["M"], r["N"], r["d"],
                                           r["ts"], r["tpt"], r["tpb"]))
    labels = [f"M{r['M']} N{r['N']} d{r['d']}\nts{r['ts']} tpt{r['tpt']} tpb{r['tpb']}"
              for r in rows]
    vals = [r["ms"] if r["ms"] < 1e29 else float("nan") for r in rows]

    n = len(rows)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n))
    bars = ax.bar(range(n), vals, color=colors, edgecolor="#222", linewidth=.4)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7)
    ax.set_ylabel("time (ms, best of N)")
    ax.set_title("softmax attention — all configurations")
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(axis="y", linestyle=":", alpha=.5)

    # 标注最快
    finite = [(i, v) for i, v in enumerate(vals) if v == v and v < 1e29]
    if finite:
        best_i, best_v = min(finite, key=lambda p: p[1])
        ax.bar(best_i, best_v, color="#e53935", edgecolor="#222", linewidth=.6)
        ax.annotate(f"best\n{best_v:.3f} ms", xy=(best_i, best_v),
                    xytext=(best_i, best_v + max(vals)*0.03),
                    ha="center", fontsize=8, color="#b71c1c",
                    arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=.8))
    fig.tight_layout()
    fn = HERE / "bench_all.png"
    fig.savefig(fn, dpi=140)
    print(f">> 图: {fn.name}")

if __name__ == "__main__":
    main()