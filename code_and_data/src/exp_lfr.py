# -*- coding: utf-8 -*-
"""Experiment 1: LFR benchmark sweep over mixing parameter mu."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import datasets, baselines as BL
from phscd import phscd

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

MUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
REPS = [0, 1, 2]
SEED_RATIO = 0.10
rows = []

for mu in MUS:
    for rep in REPS:
        G, y = datasets.lfr_benchmark(mu, n=1000, seed=rep)
        k_true = len(np.unique(y))
        rng = np.random.default_rng(rep)
        seeds, labs = [], []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            ch = rng.choice(idx, size=max(1, int(SEED_RATIO * len(idx))), replace=False)
            seeds += list(ch); labs += [c] * len(ch)

        def rec(method, assign, t, khat=""):
            rows.append(dict(mu=mu, rep=rep, method=method, k_true=k_true, k_hat=khat,
                             NMI=BL.normalized_mutual_info_score(y, assign),
                             ARI=BL.adjusted_rand_score(y, assign), runtime=t))
            print(f"mu={mu} rep={rep} {method:12s} NMI={rows[-1]['NMI']:.4f} t={t:.1f}s", flush=True)

        for mname, fn in [("Louvain", BL.run_louvain), ("Leiden", BL.run_leiden),
                          ("Infomap", BL.run_infomap), ("Greedy", BL.run_greedy),
                          ("LPA", BL.run_lpa)]:
            t0 = time.time(); a = fn(G, seed=rep); rec(mname, a, time.time() - t0)
        t0 = time.time(); a = BL.run_spectral(G, k_true, seed=rep); rec("Spectral", a, time.time() - t0)
        t0 = time.time(); a = BL.run_n2v_kmeans(G, k_true, seed=rep, num_walks=10, walk_length=40)
        rec("N2V+KM", a, time.time() - t0)
        t0 = time.time()
        a, khat, _ = phscd(G, seeds=seeds, labels=labs, k_true=k_true, p=1.0, q=2.0,
                           num_walks=10, walk_length=40, seed=rep)
        rec("PHSCD", a, time.time() - t0, khat=khat)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "lfr_results.csv"), index=False)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "lfr_results.csv"), index=False)
print("DONE")
