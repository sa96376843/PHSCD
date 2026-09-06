# -*- coding: utf-8 -*-
"""Figure 10: scalability — runtime vs. network size (log-log)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
df = pd.read_csv(os.path.join(RES, "runtime_results.csv"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
STYLE = {"Louvain": ("o", "#7f7f7f"), "Leiden": ("s", "#1f77b4"), "Infomap": ("^", "#2ca02c"),
         "Spectral": ("<", "#e377c2"), "N2V+KM": (">", "#17becf"), "PHSCD": ("*", "#d62728")}
ax = axes[0]
for m, (mk, col) in STYLE.items():
    sub = df[df.method == m]
    ax.plot(sub["n"], sub["runtime"], marker=mk, color=col, lw=2 if m == "PHSCD" else 1.4,
            ms=10 if m == "PHSCD" else 7, label=m)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Number of nodes $n$"); ax.set_ylabel("Runtime (s)")
ax.set_title("(a) Runtime vs. network size (LFR, $\\mu$=0.4)")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)

ax = axes[1]
for m, (mk, col) in STYLE.items():
    sub = df[df.method == m]
    ax.plot(sub["n"], sub["NMI"], marker=mk, color=col, lw=2 if m == "PHSCD" else 1.4,
            ms=10 if m == "PHSCD" else 7, label=m)
ax.set_xlabel("Number of nodes $n$"); ax.set_ylabel("NMI")
ax.set_title("(b) Accuracy vs. network size (LFR, $\\mu$=0.4)")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.suptitle("Fig. 10  Scalability of PHSCD and baselines on LFR benchmarks", y=1.03, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig10_runtime.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG, "fig10_runtime.pdf"), bbox_inches="tight")
print("saved fig10")
