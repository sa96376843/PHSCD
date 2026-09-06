# -*- coding: utf-8 -*-
"""Figure 6: ablation study (grouped bar chart of NMI / ARI)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
df = pd.read_csv(os.path.join(RES, "ablation_results.csv"))

datasets_order = ["Karate", "Dolphins", "Polbooks", "Football", "Polblogs", "LFR(mu=0.4)"]
variants = ["PHSCD (full)", "w/o persistence", "w/o seeds", "w/o refinement", "w/o bias (p=q=1)"]
colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, metric in zip(axes, ["NMI", "ARI"]):
    x = np.arange(len(datasets_order)); w = 0.16
    for i, v in enumerate(variants):
        vals = [df[(df.dataset == d) & (df.variant == v)][metric].iloc[0] for d in datasets_order]
        ax.bar(x + (i - 2) * w, vals, w, label=v, color=colors[i], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(datasets_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric, fontsize=12); ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"(a) Ablation on {metric}" if metric == "NMI" else f"(b) Ablation on {metric}", fontsize=11)
axes[0].legend(fontsize=8.5, ncol=2)
fig.suptitle("Fig. 6  Ablation study of PHSCD components on real and synthetic networks", y=1.03, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig6_ablation.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG, "fig6_ablation.pdf"), bbox_inches="tight")
print("saved fig6")
