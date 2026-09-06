# -*- coding: utf-8 -*-
"""Experiment 4 (cached): sensitivity to (p,q) and to the seed ratio."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import datasets, baselines as BL
from phscd import phscd, node2vec_embed

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEED_RATIO = 0.10

def seeds_for(y, ratio, rng):
    seeds, labs = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        ch = rng.choice(idx, size=max(1, int(ratio * len(idx))), replace=False)
        seeds += list(ch); labs += [c] * len(ch)
    return seeds, labs

ds = datasets.real_datasets()
PQ = [0.25, 0.5, 1.0, 2.0, 4.0]

# --- p,q grid: one embedding per (p,q,rep) ---
rows = []
for name in ["Polbooks", "Football"]:
    G, y = ds[name]
    k_true = len(np.unique(y))
    seeds, labs = seeds_for(y, SEED_RATIO, np.random.default_rng(0))
    for p in PQ:
        for q in PQ:
            accs = []
            for rep in [0, 1, 2]:
                _, Z = node2vec_embed(G, dim=64, num_walks=8, walk_length=40, p=p, q=q, seed=rep)
                a, _, _ = phscd(G, seeds=seeds, labels=labs, k_true=k_true, seed=rep, Z=Z)
                accs.append(BL.normalized_mutual_info_score(y, a))
            rows.append(dict(dataset=name, p=p, q=q, NMI=float(np.mean(accs))))
            print(name, "p", p, "q", q, "NMI", round(np.mean(accs), 4), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sensitivity_pq.csv"), index=False)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "sensitivity_pq.csv"), index=False)

# --- seed ratio: one embedding per rep, reused across ratios ---
rows2 = []
for name in ["Polbooks", "Football", "Polblogs"]:
    G, y = ds[name]
    k_true = len(np.unique(y))
    for rep in [0, 1, 2]:
        _, Z = node2vec_embed(G, dim=64, num_walks=8, walk_length=40, p=1.0, q=2.0, seed=rep)
        for ratio in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
            seeds, labs = seeds_for(y, ratio, np.random.default_rng(rep))
            a, _, _ = phscd(G, seeds=seeds, labels=labs, k_true=k_true, seed=rep,
                            Z=Z, use_seeds=ratio > 0)
            rows2.append(dict(dataset=name, ratio=ratio, rep=rep,
                              NMI=BL.normalized_mutual_info_score(y, a)))
            print(name, rep, "ratio", ratio, "NMI", round(rows2[-1]["NMI"], 4), flush=True)
    pd.DataFrame(rows2).to_csv(os.path.join(OUT, "sensitivity_seedratio_raw.csv"), index=False)
df2 = pd.DataFrame(rows2)
agg = df2.groupby(["dataset", "ratio"])["NMI"].agg(["mean", "std"]).reset_index()
agg.columns = ["dataset", "ratio", "NMI", "NMI_std"]
agg.to_csv(os.path.join(OUT, "sensitivity_seedratio.csv"), index=False)
print("DONE")
