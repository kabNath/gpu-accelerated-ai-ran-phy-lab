#!/usr/bin/env python3
"""Generate the three README figures from the committed result files.

    python benchmarks/make_figures.py

Reads  results/sionna_bler_curves.json  and  results/cuda_benchmark.json,
writes PNGs into results/figures/. No GPU needed — plots committed data.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless (WSL/CI safe)
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 11})


def fig_bler():
    path = os.path.join(RES, "sionna_bler_curves.json")
    if not os.path.exists(path):
        print("skip BLER: results/sionna_bler_curves.json not found")
        return
    d = json.load(open(path))
    snr = np.asarray(d["snr_dbs"], float)
    curves = d["curves"]
    keys = sorted(curves, key=lambda k: int(k))
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(keys)))
    plt.figure(figsize=(7, 4.5))
    for c, k in zip(cmap, keys):
        bler = np.asarray(curves[k], float)
        bler[bler <= 0] = np.nan               # log can't show 0 (no errors observed)
        plt.semilogy(snr, bler, marker="o", ms=3, lw=1.4, color=c, label=f"MCS {k}")
    plt.axhline(0.1, ls="--", lw=1, color="0.4")
    plt.text(snr.min(), 0.12, "10% BLER target", fontsize=9, color="0.4")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Block Error Rate")
    plt.title("Sionna 5G TDL link — BLER vs SNR per MCS")
    plt.ylim(1e-4, 1.2)
    plt.legend(ncol=2, fontsize=8, loc="lower left")
    plt.tight_layout()
    out = os.path.join(FIG, "sionna_bler_curves.png")
    plt.savefig(out); plt.close()
    print("wrote", out)


def _bench():
    return json.load(open(os.path.join(RES, "cuda_benchmark.json")))


def fig_speedup():
    b = _bench()
    sizes = b["sizes"]; cpu = b["cpu_ms"]; gpu = b["gpu_ms"]; sp = b["speedup"]
    x = np.arange(len(sizes)); w = 0.38
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - w/2, cpu, w, label="CPU (NumPy)", color="#888")
    plt.bar(x + w/2, gpu, w, label="GPU (CUDA, custom kernel + cuSolver/cuBLAS)",
            color="#76b900")                    # NVIDIA green
    plt.yscale("log")
    plt.ylim(top=max(cpu) * 8)                  # headroom for the speedup labels
    for i, s in enumerate(sp):
        plt.text(x[i], max(cpu[i], gpu[i]) * 1.7, f"{s:.0f}×",
                 ha="center", fontweight="bold")
    plt.xticks(x, [f"N={n}" for n in sizes])
    plt.ylabel("latency per batch (ms, log scale)")
    plt.title(f"MMSE channel estimation — CPU vs GPU ({b['device']})")
    plt.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIG, "mmse_speedup_rtx4090.png")
    plt.savefig(out); plt.close()
    print("wrote", out)


def fig_error():
    b = _bench()
    sizes = b["sizes"]; err = b["rel_err"]
    x = np.arange(len(sizes))
    plt.figure(figsize=(7, 4.5))
    plt.bar(x, err, color="#2a9d8f", width=0.5)
    plt.yscale("log")
    plt.axhline(1e-3, ls="--", color="crimson", label="test tolerance (1e-3)")
    for i, e in enumerate(err):
        plt.text(x[i], e * 1.15, f"{e:.1e}", ha="center", fontsize=9)
    plt.xticks(x, [f"N={n}" for n in sizes])
    plt.ylabel("relative error vs CPU reference (log scale)")
    plt.title(f"GPU MMSE numerical accuracy ({b['dtype']}) vs CPU")
    plt.ylim(1e-5, 2e-3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIG, "mmse_relative_error.png")
    plt.savefig(out); plt.close()
    print("wrote", out)


if __name__ == "__main__":
    fig_bler()
    fig_speedup()
    fig_error()
    print("done -> results/figures/")
