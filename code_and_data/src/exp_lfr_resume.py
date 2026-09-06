# -*- coding: utf-8 -*-
"""Resume LFR experiment: only missing cells (mu=0.7, reps 0..2). Appends to lfr_results.csv."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, csv
import datasets, baselines as BL
from phscd import phscd

RES = os.path.join(os.path.dirname(__file__), "..", "results")
path = os.path.join(RES, "lfr_results.csv")
done = set()
with open(path) as f:
    for r in csv.DictReader(f):
        done.add((r["mu"], r["rep"], r["method"]))

SEED_RATIO = 0.10
METHODS = ["Louvain", "Leiden", "Infomap", "Greedy", "LPA", "Spectral", "N2V+KM", "PHSCD"]

with open(path, "a", newline="") as f:
    w = csv.writer(f)
    for mu in [0.7]:
        for rep in [0, 1, 2]:
            G, y = datasets.lfr_benchmark(mu, n=1000, seed=rep)
            y = np.asarray(y)
            k_true = len(np.unique(y))
            rng = np.random.default_rng(rep)
            seeds, labs = [], []
            for c in np.unique(y):
                idx = np.where(y == c)[0]
                ch = rng.choice(idx, size=max(1, int(SEED_RATIO * len(idx))), replace=False)
                seeds += list(ch); labs += [c] * len(ch)

            def rec(method, assign, t, khat=""):
                w.writerow([mu, rep, method, k_true, khat,
                            BL.normalized_mutual_info_score(y, assign),
                            BL.adjusted_rand_score(y, assign), round(t, 2)])
                f.flush()
                print(f"mu={mu} rep={rep} {method:12s} NMI={BL.normalized_mutual_info_score(y, assign):.4f} t={t:.1f}s", flush=True)

            for mname, fn in [("Louvain", BL.run_louvain), ("Leiden", BL.run_leiden),
                              ("Infomap", BL.run_infomap), ("Greedy", BL.run_greedy),
                              ("LPA", BL.run_lpa)]:
                if (str(mu), str(rep), mname) in done: continue
                t0 = time.time(); a = fn(G, seed=rep); rec(mname, a, time.time() - t0)
            if (str(mu), str(rep), "Spectral") not in done:
                t0 = time.time(); a = BL.run_spectral(G, k_true, seed=rep); rec("Spectral", a, time.time() - t0)
            if (str(mu), str(rep), "N2V+KM") not in done:
                t0 = time.time(); a = BL.run_n2v_kmeans(G, k_true, seed=rep, num_walks=10, walk_length=40)
                rec("N2V+KM", a, time.time() - t0)
            if (str(mu), str(rep), "PHSCD") not in done:
                t0 = time.time()
                a, khat, _ = phscd(G, seeds=seeds, labels=labs, k_true=k_true, p=1.0, q=2.0,
                                   num_walks=10, walk_length=40, seed=rep)
                rec("PHSCD", a, time.time() - t0, khat=khat)
print("DONE", flush=True)
