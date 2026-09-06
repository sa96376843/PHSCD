# -*- coding: utf-8 -*-
"""Figure 2: persistence-based estimation of k on the Football network."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import datasets
from phscd import node2vec_embed, persistence_candidates, seeded_kmeans, _nx_modularity
from sklearn.cluster import KMeans

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
G, y = datasets.load_gml("football")
nodes, Z = node2vec_embed(G, dim=64, num_walks=10, walk_length=40, p=1.0, q=2.0, seed=0)
cand, w = persistence_candidates(Z, k_max=12)
desc = np.sort(w)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ks_range = np.arange(2, len(desc[:14]) + 2)
ax.stem(ks_range, desc[:14], basefmt=" ")
gaps = desc[:13] - desc[1:14]
for i, kk in enumerate(ks_range[:-1]):
    ax.annotate("", xy=(kk + 1, desc[i + 1]), xytext=(kk, desc[i]),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.0))
ax.set_xlabel("Number of clusters $k$ (cut rank)")
ax.set_ylabel("Merge distance $w$")
ax.set_title("(a) 0-dim persistence: descending MST merge distances")

ax = axes[1]
Ks = sorted(set(cand) | {12})
Qs, NMIs = [], []
rng = np.random.default_rng(0)
seeds, labs = [], []
for c in np.unique(y):
    idx = np.where(y == c)[0]
    ch = rng.choice(idx, size=max(1, int(0.1 * len(idx))), replace=False)
    seeds += list(ch); labs += [c] * len(ch)
for k in Ks:
    a = seeded_kmeans(Z, seeds, labs, k, seed=0)
    Qs.append(_nx_modularity(G, a))
    NMIs.append(__import__("sklearn.metrics", fromlist=["normalized_mutual_info_score"])
                .normalized_mutual_info_score(y, a))
ax.plot(Ks, Qs, "o-", color="#1a5276", label="Modularity $Q$")
ax.set_xlabel("Candidate $k$"); ax.set_ylabel("Modularity $Q$", color="#1a5276")
ax.tick_params(axis="y", labelcolor="#1a5276")
ax2 = ax.twinx()
ax2.plot(Ks, NMIs, "s--", color="#b03a2e", label="NMI")
ax2.set_ylabel("NMI", color="#b03a2e"); ax2.tick_params(axis="y", labelcolor="#b03a2e")
bestk = Ks[int(np.argmax(Qs))]
ax.axvline(bestk, color="gray", ls=":", lw=1.2)
ax.text(bestk + 0.15, min(Qs), f"$\\hat{{k}}$={bestk}", fontsize=10)
ax.set_title("(b) Modularity selection over persistence candidates")
fig.suptitle("Fig. 2  Persistent-homology-guided estimation of the community number (Football)", y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_persistence.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG, "fig2_persistence.pdf"), bbox_inches="tight")
print("saved fig2; candidates:", cand, "best k:", bestk, "Q:", [round(q,3) for q in Qs], "NMI:", [round(x,3) for x in NMIs])
