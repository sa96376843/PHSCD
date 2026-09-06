# -*- coding: utf-8 -*-
"""Figure 1: PHSCD framework diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)

fig, ax = plt.subplots(figsize=(13.5, 4.6))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 4.6); ax.axis("off")

C = dict(ec="#2c3e50", lw=1.4)
def box(x, y, w, h, title, lines, fc="#eaf2fb"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, **C))
    ax.text(x + w/2, y + h - 0.34, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#1a3a5c")
    for i, ln in enumerate(lines):
        ax.text(x + w/2, y + h - 0.72 - i * 0.32, ln, ha="center", va="center", fontsize=8.6, color="#333")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.5, color="#2c3e50"))

# Stage boxes
box(0.15, 1.55, 2.15, 1.75, "Input", ["Graph G=(V,E)", "Seed set S (ratio rho)",
                                      "Labels of seeds"], fc="#fdebd0")
box(2.85, 1.55, 2.55, 1.75, "Stage 1: Biased Random-Walk",
    ["2nd-order walks (p, q)", "Skip-gram + neg. sampling", "Embeddings Z in R^d"])
box(5.95, 1.55, 2.55, 1.75, "Stage 2: Persistent Homology",
    ["Cosine-sim. filtration", "0-dim merge tree (MST)", "Persistence gap -> k"], fc="#e8f8f0")
box(9.05, 1.55, 2.15, 1.75, "Stage 3: Seed-Guided",
    ["Centroids from seeds", "Constrained k-means", "(seeds fixed)"], fc="#e8f0fe")
box(11.3, 0.55, 2.05, 1.75, "Stage 4: Refinement",
    ["Margin tau-filter", "Local modularity gain", "Boundary reassignment"], fc="#f5e8f8")
box(11.3, 2.75, 2.05, 1.15, "Output", ["Partition C*", "Estimated k"], fc="#fdebd0")

arrow(2.30, 2.42, 2.85, 2.42)
arrow(5.40, 2.42, 5.95, 2.42)
arrow(8.50, 2.42, 9.05, 2.42)
arrow(11.20, 2.05, 11.30, 1.45)
arrow(12.32, 2.30, 12.32, 2.75)
# seed side input to stage 3
ax.add_patch(FancyArrowPatch((1.2, 1.55), (10.1, 1.55), arrowstyle="-|>",
                             mutation_scale=14, lw=1.2, ls="--", color="#b9770e",
                             connectionstyle="arc3,rad=0.25"))
ax.text(5.6, 0.62, "prior labels flow to Stage 3 (semi-supervised guidance)",
        fontsize=8.5, color="#b9770e", ha="center", style="italic")

plt.title("Fig. 1  The overall framework of the proposed PHSCD algorithm", fontsize=12, pad=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_framework.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG, "fig1_framework.pdf"), bbox_inches="tight")
print("saved fig1")
