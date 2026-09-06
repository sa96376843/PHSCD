# -*- coding: utf-8 -*-
"""Experiment 5: scalability (runtime vs. network size on LFR benchmarks)."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import datasets, baselines as BL
from phscd import phscd

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rows = []
for n in [500, 1000, 2000, 4000]:
    G, y = datasets.lfr_benchmark(0.4, n=n, seed=0)
    k_true = len(np.unique(y))
    rng = np.random.default_rng(0)
    seeds, labs = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        ch = rng.choice(idx, size=max(1, int(0.1 * len(idx))), replace=False)
        seeds += list(ch); labs += [c] * len(ch)
    for mname, fn in [("Louvain", lambda: BL.run_louvain(G, seed=0)),
                      ("Leiden", lambda: BL.run_leiden(G, seed=0)),
                      ("Infomap", lambda: BL.run_infomap(G, seed=0)),
                      ("Spectral", lambda: BL.run_spectral(G, k_true, seed=0)),
                      ("N2V+KM", lambda: BL.run_n2v_kmeans(G, k_true, seed=0, num_walks=10, walk_length=40)),
                      ("PHSCD", lambda: phscd(G, seeds=seeds, labels=labs, k_true=k_true, p=1.0, q=2.0,
                                              num_walks=10, walk_length=40, seed=0)[0])]:
        t0 = time.time(); a = fn(); t = time.time() - t0
        nmi = BL.normalized_mutual_info_score(y, a)
        rows.append(dict(n=n, m=G.number_of_edges(), method=mname, runtime=t, NMI=nmi))
        print(n, mname, "t=", round(t, 2), "NMI=", round(nmi, 4), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "runtime_results.csv"), index=False)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "runtime_results.csv"), index=False)
print("DONE")
