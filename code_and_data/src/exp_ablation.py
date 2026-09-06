# -*- coding: utf-8 -*-
"""Experiment 3 (cached): ablation study of PHSCD components.
Embeddings are computed once per (dataset, rep, bias-config) and reused."""
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

VARIANTS = {
    "PHSCD (full)":      dict(use_persistence=True,  use_seeds=True,  use_refine=True),
    "w/o persistence":   dict(use_persistence=False, use_seeds=True,  use_refine=True),
    "w/o seeds":         dict(use_persistence=True,  use_seeds=False, use_refine=True),
    "w/o refinement":    dict(use_persistence=True,  use_seeds=True,  use_refine=False),
    "w/o bias (p=q=1)":  dict(use_persistence=True,  use_seeds=True,  use_refine=True),
}

CASES = {}
ds = datasets.real_datasets()
for nm in ["Karate", "Dolphins", "Polbooks", "Football", "Polblogs"]:
    CASES[nm] = ds[nm]
CASES["LFR(mu=0.4)"] = datasets.lfr_benchmark(0.4, n=1000, seed=0)

rows = []
for name, (G, y) in CASES.items():
    k_true = len(np.unique(y))
    for rep in [0, 1, 2]:
        seeds, labs = seeds_for(y, SEED_RATIO, np.random.default_rng(rep))
        # two embeddings per rep: biased (p=1,q=2) and unbiased (p=q=1)
        _, Z_b = node2vec_embed(G, dim=64, num_walks=10, walk_length=40, p=1.0, q=2.0, seed=rep)
        _, Z_u = node2vec_embed(G, dim=64, num_walks=10, walk_length=40, p=1.0, q=1.0, seed=rep)
        for vname, cfg in VARIANTS.items():
            Z = Z_u if "bias" in vname else Z_b
            a, khat, _ = phscd(G, seeds=seeds, labels=labs, k_true=k_true,
                               seed=rep, Z=Z, **cfg)
            ev = BL.evaluate_all(G, y, a)
            rows.append(dict(dataset=name, variant=vname, rep=rep, k_true=k_true, k_hat=khat,
                             **{kk: float(ev[kk]) for kk in ["NMI", "ARI", "F1", "Purity", "Q"]}))
            print(name, rep, vname, "NMI", round(ev["NMI"], 4), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ablation_results.csv"), index=False)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "ablation_results.csv"), index=False)
print("DONE")
