# -*- coding: utf-8 -*-
"""Figures 3-4: NMI / ARI vs. mixing parameter mu on LFR benchmarks."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
df = pd.read_csv(os.path.join(RES, "lfr_results.csv"))

ORDER = ["Louvain", "Leiden", "Infomap", "Greedy", "LPA", "Spectral", "N2V+KM", "PHSCD"]
STYLE = {"Louvain": ("o", "#7f7f7f"), "Leiden": ("s", "#1f77b4"), "Infomap": ("^", "#2ca02c"),
         "Greedy": ("D", "#9467bd"), "LPA": ("v", "#8c564b"), "Spectral": ("<", "#e377c2"),
         "N2V+KM": (">", "#17becf"), "PHSCD": ("*", "#d62728")}

for metric, fignum in [("NMI", 3), ("ARI", 4)]:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m in ORDER:
        sub = df[df.method == m].groupby("mu")[metric].agg(["mean", "std"]).reset_index()
        mk, col = STYLE[m]
        lw, ms = (2.6, 11) if m == "PHSCD" else (1.4, 7)
        ax.errorbar(sub["mu"], sub["mean"], yerr=sub["std"], marker=mk, color=col,
                    lw=lw, ms=ms, capsize=3, label=m, alpha=0.95 if m == "PHSCD" else 0.8)
    ax.set_xlabel(r"Mixing parameter $\mu$", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    plt.title(f"Fig. {fignum}  {metric} on LFR benchmarks with varying mixing parameter", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"fig{fignum}_lfr_{metric.lower()}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG, f"fig{fignum}_lfr_{metric.lower()}.pdf"), bbox_inches="tight")
    plt.close()
print("saved fig3, fig4")
# quick summary at high mu
print(df[df.mu >= 0.5].groupby("method")[["NMI", "ARI"]].mean().round(4))
