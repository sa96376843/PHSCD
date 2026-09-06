# -*- coding: utf-8 -*-
"""Figure 5: t-SNE visualisation of PHSCD embeddings on Football (truth vs. detected)."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import datasets, baselines as BL
from phscd import phscd, node2vec_embed

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
G, y = datasets.load_gml("football")
rng = np.random.default_rng(0)
seeds, labs = [], []
for c in np.unique(y):
    idx = np.where(y == c)[0]
    ch = rng.choice(idx, size=max(1, int(0.1 * len(idx))), replace=False)
    seeds += list(ch); labs += [c] * len(ch)

a, khat, Z = phscd(G, seeds=seeds, labels=labs, p=1.0, q=2.0,
                   num_walks=10, walk_length=40, seed=0)
print("Football PHSCD k_hat:", khat, "NMI:", round(BL.normalized_mutual_info_score(y, a), 4))
ts = TSNE(n_components=2, random_state=0, init="pca", perplexity=20).fit_transform(Z)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
cmap = plt.get_cmap("tab20")
for ax, lab, ttl in [(axes[0], y, "(a) Ground truth (12 conferences)"),
                     (axes[1], a, f"(b) PHSCD detection ($\\hat{{k}}$={khat})")]:
    for c in np.unique(lab):
        m = lab == c
        ax.scatter(ts[m, 0], ts[m, 1], s=38, color=cmap(c % 20), alpha=0.85,
                   edgecolors="white", linewidths=0.4)
    ax.scatter(ts[seeds, 0], ts[seeds, 1], s=90, facecolors="none",
               edgecolors="black", linewidths=1.4, marker="s", label="seeds" if ax is axes[0] else None)
    ax.set_title(ttl, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
fig.suptitle("Fig. 5  t-SNE projection of the learned node embeddings (Football network)", y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig5_tsne.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG, "fig5_tsne.pdf"), bbox_inches="tight")
print("saved fig5")
