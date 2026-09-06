# -*- coding: utf-8 -*-
"""Generate all paper-1 tables as Markdown snippets (tables_md/*.md)."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import datasets

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "tables_md")
os.makedirs(OUT, exist_ok=True)
ORDER = ["Louvain", "Leiden", "Infomap", "Greedy", "LPA", "Spectral", "N2V+KM", "PHSCD"]

def w(name, txt):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(txt + "\n")
    print("wrote", name)

# ---------- Table 1: dataset statistics ----------
ds = datasets.real_datasets()
META = {"Karate": ("Social", "Club split"), "Dolphins": ("Animal", "GN first split"),
        "Polbooks": ("Co-purchase", "Political leaning"), "Football": ("Sports", "Conference"),
        "Polblogs": ("Hyperlink", "Political leaning"), "Adjnoun": ("Language", "--"),
        "Lesmis": ("Co-appearance", "--")}
lines = ["| Dataset | Type | $n$ | $m$ | $\\langle k\\rangle$ | #Comm. | Ground truth |",
         "|---|---|---|---|---|---|---|"]
for name, (G, y) in ds.items():
    k = len(np.unique(y)) if y is not None else "--"
    avgk = 2 * G.number_of_edges() / G.number_of_nodes()
    lines.append(f"| {name} | {META[name][0]} | {G.number_of_nodes()} | {G.number_of_edges()} "
                 f"| {avgk:.2f} | {k} | {META[name][1]} |")
w("tab1_datasets.md", "\n".join(lines))

# ---------- Table 2: LFR ----------
df = pd.read_csv(os.path.join(RES, "lfr_results.csv"))
piv_nmi = df.pivot_table(index="method", columns="mu", values="NMI")
piv_ari = df.pivot_table(index="method", columns="mu", values="ARI")
mus = list(piv_nmi.columns)
lines = ["| Metric | Method | " + " | ".join(f"$\\mu$={c:.1f}" for c in mus) + " |",
         "|---|---|" + "---|" * len(mus)]
for metric, piv in [("NMI", piv_nmi), ("ARI", piv_ari)]:
    for i, m in enumerate(ORDER):
        lines.append(f"| {metric if i == 0 else ''} | {m} | "
                     + " | ".join(f"{piv.loc[m, c]:.3f}" for c in mus) + " |")
w("tab2_lfr.md", "\n".join(lines))

# ---------- Table 3: real networks ----------
df = pd.read_csv(os.path.join(RES, "real_results.csv"))
labeled = ["Karate", "Dolphins", "Polbooks", "Football", "Polblogs"]
lines = ["| Dataset | Method | NMI | ARI | F1 | Purity | $Q$ |", "|---|---|---|---|---|---|---|"]
for d in labeled:
    sub = df[df.dataset == d]
    for i, m in enumerate(ORDER):
        r = sub[sub.method == m]
        if len(r) == 0: continue
        r = r.iloc[0]
        lines.append(f"| {d if i == 0 else ''} | {m} | {r['NMI']:.3f} | {r['ARI']:.3f} "
                     f"| {r['F1']:.3f} | {r['Purity']:.3f} | {r['Q']:.3f} |")
w("tab3_real.md", "\n".join(lines))

# ---------- Table 4: modularity on unlabeled ----------
lines = ["| Method | Adjnoun $Q$ | Lesmis $Q$ |", "|---|---|---|"]
for m in ["Louvain", "Leiden", "Infomap", "Greedy", "LPA", "PHSCD-U (unsupervised)"]:
    vals = []
    for d in ["Adjnoun", "Lesmis"]:
        key = "PHSCD-U" if m.startswith("PHSCD-U") else m
        r = df[(df.dataset == d) & (df.method == key)]
        vals.append(f"{r.iloc[0]['Q']:.3f}" if len(r) else "--")
    lines.append(f"| {m} | {vals[0]} | {vals[1]} |")
w("tab4_unlabeled.md", "\n".join(lines))

# ---------- Table 5: ablation ----------
df = pd.read_csv(os.path.join(RES, "ablation_results.csv"))
variants = ["PHSCD (full)", "w/o persistence", "w/o seeds", "w/o refinement", "w/o bias (p=q=1)"]
lines = ["| Dataset | Variant | NMI | ARI | F1 |", "|---|---|---|---|---|"]
for d in ["Karate", "Dolphins", "Polbooks", "Football", "Polblogs", "LFR(mu=0.4)"]:
    sub = df[df.dataset == d].groupby("variant").mean(numeric_only=True)
    for i, v in enumerate(variants):
        r = sub.loc[v]
        lines.append(f"| {d if i == 0 else ''} | {v} | {r['NMI']:.3f} | {r['ARI']:.3f} | {r['F1']:.3f} |")
w("tab5_ablation.md", "\n".join(lines))

# ---------- Table 6: hyperparameters ----------
lines = ["| Symbol | Meaning | Value |", "|---|---|---|",
         "| $d$ | embedding dimension | 64 |",
         "| $r$ | walks per node | 10 |",
         "| $\\ell$ | walk length | 40 |",
         "| $w$ | skip-gram window | 10 |",
         "| $p, q$ | return / in-out bias | 1.0, 2.0 |",
         "| $n_{neg}$ | negative samples | 5 |",
         "| $k_{nn}$ | kNN sparsification | 10 |",
         "| $n_{cand}$ | persistence candidates | 6 |",
         "| $\\rho$ | seed ratio | 10% |",
         "| $\\tau_q$ | margin quantile | 0.30 |",
         "| $T$ | max refinement sweeps | 10 |"]
w("tab6_params.md", "\n".join(lines))

# ---------- Table 7: runtime ----------
rp = os.path.join(RES, "runtime_results.csv")
if os.path.exists(rp):
    df = pd.read_csv(rp)
    piv_t = df.pivot_table(index="method", columns="n", values="runtime")
    piv_n = df.pivot_table(index="method", columns="n", values="NMI")
    ns = list(piv_t.columns)
    lines = ["| Metric | Method | " + " | ".join(f"$n$={c}" for c in ns) + " |",
             "|---|---|" + "---|" * len(ns)]
    for metric, piv, fmt in [("Time (s)", piv_t, "{:.2f}"), ("NMI", piv_n, "{:.3f}")]:
        for i, m in enumerate(["Louvain", "Leiden", "Infomap", "Spectral", "N2V+KM", "PHSCD"]):
            if m not in piv.index: continue
            lines.append(f"| {metric if i == 0 else ''} | {m} | "
                         + " | ".join(fmt.format(piv.loc[m, c]) for c in ns) + " |")
    w("tab7_runtime.md", "\n".join(lines))
else:
    print("runtime_results.csv not found; skip tab7")
print("ALL MD TABLES DONE")
