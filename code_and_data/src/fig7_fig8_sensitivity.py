# -*- coding: utf-8 -*-
"""Figure 7: (p,q) sensitivity heatmaps. Figure 8: seed-ratio sensitivity."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
RES = os.path.join(os.path.dirname(__file__), "..", "results")

# ---- fig 7: p,q heatmaps ----
df = pd.read_csv(os.path.join(RES, "sensitivity_pq.csv"))
PQ = [0.25, 0.5, 1.0, 2.0, 4.0]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, name, ttl in zip(axes, ["Polbooks", "Football"], ["(a) Polbooks", "(b) Football"]):
    M = np.zeros((5, 5))
    for i, p in enumerate(PQ):
        for j, q in enumerate(PQ):
            M[i, j] = df[(df.dataset == name) & (df.p == p) & (df.q == q)]["NMI"].iloc[0]
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.55 else "black", fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels(PQ)
    ax.set_yticks(range(5)); ax.set_yticklabels(PQ)
    ax.set_xlabel("$q$"); ax.set_ylabel("$p$"); ax.set_title(ttl, fontsize=11)
fig.colorbar(im, ax=axes, shrink=0.85, label="NMI")
fig.suptitle("Fig. 7  Sensitivity of PHSCD to the random-walk bias parameters $(p,q)$", y=1.04, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig7_pq.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG, "fig7_pq.pdf"), bbox_inches="tight")
plt.close()

# ---- fig 8: seed ratio ----
df2 = pd.read_csv(os.path.join(RES, "sensitivity_seedratio.csv"))
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for name, col in zip(["Polbooks", "Football", "Polblogs"], ["#1f77b4", "#2ca02c", "#d62728"]):
    sub = df2[df2.dataset == name]
    ax.errorbar(sub["ratio"] * 100, sub["NMI"], yerr=sub["NMI_std"], marker="o",
                color=col, lw=1.8, capsize=3, label=name)
ax.set_xlabel("Seed ratio (%)", fontsize=12); ax.set_ylabel("NMI", fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=10)
plt.title("Fig. 8  Effect of the labelled-seed ratio on detection accuracy", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig8_seedratio.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG, "fig8_seedratio.pdf"), bbox_inches="tight")
print("saved fig7, fig8")
