# -*- coding: utf-8 -*-
"""Experiment 2: full evaluation on seven real-world networks."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import datasets, baselines as BL
from phscd import phscd

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
SEED_RATIO = 0.10
ds = datasets.real_datasets()
rows = []

def seeds_for(y, ratio, rng):
    seeds, labs = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        ch = rng.choice(idx, size=max(1, int(ratio * len(idx))), replace=False)
        seeds += list(ch); labs += [c] * len(ch)
    return seeds, labs

for name, (G, y) in ds.items():
    print("=" * 20, name, flush=True)
    k_true = len(np.unique(y)) if y is not None else None
    methods = {"Louvain": BL.run_louvain, "Leiden": BL.run_leiden, "Infomap": BL.run_infomap,
               "Greedy": BL.run_greedy, "LPA": BL.run_lpa}
    for mname, fn in methods.items():
        t0 = time.time(); a = fn(G, seed=0); t = time.time() - t0
        r = dict(dataset=name, method=mname, runtime=t)
        if y is not None:
            r.update(BL.evaluate_all(G, y, a))
        else:
            r["Q"] = BL.modularity(G, a)
        rows.append(r); print(mname, r, flush=True)
    if y is not None:
        t0 = time.time(); a = BL.run_spectral(G, k_true, seed=0); t = time.time() - t0
        r = dict(dataset=name, method="Spectral", runtime=t); r.update(BL.evaluate_all(G, y, a))
        rows.append(r); print("Spectral", r, flush=True)
        t0 = time.time(); a = BL.run_n2v_kmeans(G, k_true, seed=0, num_walks=10, walk_length=40)
        t = time.time() - t0
        r = dict(dataset=name, method="N2V+KM", runtime=t); r.update(BL.evaluate_all(G, y, a))
        rows.append(r); print("N2V+KM", r, flush=True)
        # PHSCD: average over 3 seed draws
        accs = []
        for rep in [0, 1, 2]:
            seeds, labs = seeds_for(y, SEED_RATIO, np.random.default_rng(rep))
            t0 = time.time()
            a, khat, _ = phscd(G, seeds=seeds, labels=labs, k_true=k_true, p=1.0, q=2.0,
                               num_walks=10, walk_length=40, seed=rep)
            t = time.time() - t0
            ev = BL.evaluate_all(G, y, a); ev["runtime"] = t; ev["k_hat"] = khat
            accs.append(ev)
        r = dict(dataset=name, method="PHSCD")
        for kk in ["NMI", "ARI", "F1", "Purity", "Q", "runtime", "k_hat"]:
            r[kk] = float(np.mean([e[kk] for e in accs]))
            r[kk + "_std"] = float(np.std([e[kk] for e in accs]))
        rows.append(r); print("PHSCD", r, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "real_results.csv"), index=False)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "real_results.csv"), index=False)
print("DONE")
