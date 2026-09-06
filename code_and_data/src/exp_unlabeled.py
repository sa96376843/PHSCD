# -*- coding: utf-8 -*-
"""Run unsupervised PHSCD-U on unlabeled networks (Adjnoun, Lesmis) for Table 4."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import datasets, baselines as BL
from phscd import phscd

RES = os.path.join(os.path.dirname(__file__), "..", "results")
ds = datasets.real_datasets()
rows = []
for name in ["Adjnoun", "Lesmis"]:
    G, y = ds[name]
    qs, ts, ks = [], [], []
    for rep in range(3):
        t0 = time.time()
        a, khat, _ = phscd(G, use_seeds=False, p=1.0, q=2.0, num_walks=10, walk_length=40, seed=rep)
        dt = time.time() - t0
        qs.append(BL.modularity(G, a)); ts.append(dt); ks.append(khat)
        print(name, rep, "k_hat", khat, "Q", round(qs[-1], 4), f"{dt:.1f}s", flush=True)
    rows.append(dict(dataset=name, method="PHSCD-U", runtime=np.mean(ts),
                     NMI=np.nan, ARI=np.nan, F1=np.nan, Purity=np.nan, Q=np.mean(qs),
                     NMI_std="", ARI_std="", F1_std="", Purity_std="", Q_std=np.std(qs),
                     runtime_std=np.std(ts), k_hat=np.mean(ks), k_hat_std=np.std(ks)))
path = os.path.join(RES, "real_results.csv")
df = pd.read_csv(path)
df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
df.to_csv(path, index=False)
print("DONE")
