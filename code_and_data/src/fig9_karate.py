# -*- coding: utf-8 -*-
"""Figure 9: detected communities on the Karate network (PHSCD vs. Louvain)."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, networkx as nx
import datasets, baselines as BL
from phscd import phscd

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
G, y = datasets.load_karate()
rng = np.random.default_rng(0)
seeds, labs = [], []
for c in np.unique(y):
    idx = np.where(y == c)[0]
    ch = rng.choice(idx, size=max(1, int(0.1 * len(idx))), replace=False)
    seeds += list(ch); labs += [c] * len(ch)
a, khat, _ = phscd(G, seeds=seeds, labels=labs, p=1.0, q=2.0, num_walks=10, walk_length=40, seed=0)
louv = BL.run_louvain(G, seed=0)
print("Karate PHSCD k:", khat, "NMI:", round(BL.normalized_mutual_info_score(y, a), 4),
      "| Louvain NMI:", round(BL.normalized_mutual_info_score(y, louv), 4))
pos = nx.spring_layout(G, seed=42)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, lab, ttl in [(axes[0], y, "(a) Ground truth"),
                     (axes[1], a, f"(b) PHSCD ($\\hat{{k}}$={khat}, NMI={BL.normalized_mutual_info_score(y, a):.3f})"),
                     (axes[2], louv, f"(c) Louvain (NMI={BL.normalized_mutual_info_score(y, louv):.3f})")]:
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, width=0.9)
    for c, col in zip(np.unique(lab), ["#2166ac", "#b2182b", "#1b7837", "#762a83"]):
        m = np.array(lab) == c
        nx.draw_networkx_nodes(G, pos, nodelist=list(np.where(m)[0]), ax=ax,
                               node_color=col, node_size=220, alpha=0.92, linewidths=0.6, edgecolors="white")
    nx.draw_networkx_nodes(G, pos, nodelist=seeds, ax=ax, node_color="none",
                           node_size=340, node_shape="s", linewidths=1.6, edgecolors="black")
    ax.set_title(ttl, fontsize=11); ax.axis("off")
fig.suptitle("Fig. 9  Community assignments on Zachary's Karate Club (squares = seed nodes)", y=1.03, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig9_karate.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG, "fig9_karate.pdf"), bbox_inches="tight")
print("saved fig9")
