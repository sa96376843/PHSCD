# -*- coding: utf-8 -*-
"""Generate all LaTeX tables from result CSVs."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import datasets

RES = os.path.join(os.path.dirname(__file__), "..", "results")
TAB = os.path.join(os.path.dirname(__file__), "..", "tables")
os.makedirs(TAB, exist_ok=True)


def w(name, txt):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(txt)
    print("wrote", name)


# ---------- Table 1: dataset statistics ----------
ds = datasets.real_datasets()
rows = []
META = {
    "Karate":   ("Social", "Zachary 1977", "club split"),
    "Dolphins": ("Animal", "Lusseau et al. 2003", "Newman-Girvan split"),
    "Polbooks": ("Co-purchase", "Krebs 2004", "political leaning"),
    "Football": ("Sports", "Girvan & Newman 2002", "conference"),
    "Polblogs": ("Hyperlink", "Adamic & Glance 2005", "political leaning"),
    "Adjnoun":  ("Language", "Newman 2006", "--"),
    "Lesmis":   ("Co-appearance", "Knuth 1993", "--"),
}
for name, (G, y) in ds.items():
    k = len(np.unique(y)) if y is not None else "--"
    avgk = 2 * G.number_of_edges() / G.number_of_nodes()
    rows.append((name, META[name][0], G.number_of_nodes(), G.number_of_edges(),
                 round(avgk, 2), k, META[name][2]))
t = "\\begin{tabular}{llrrrrl}\n\\toprule\nDataset & Type & $n$ & $m$ & $\\langle k\\rangle$ & \\#Comm. & Ground truth \\\\\n\\midrule\n"
for r in rows:
    t += f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n"
w("tab1_datasets.tex", t)

# ---------- Table 2: LFR results (mean NMI/ARI) ----------
df = pd.read_csv(os.path.join(RES, "lfr_results.csv"))
ORDER = ["Louvain", "Leiden", "Infomap", "Greedy", "LPA", "Spectral", "N2V+KM", "PHSCD"]
piv_nmi = df.pivot_table(index="method", columns="mu", values="NMI")
piv_ari = df.pivot_table(index="method", columns="mu", values="ARI")
t = "\\begin{tabular}{l" + "c" * len(piv_nmi.columns) + "}\n\\toprule\n"
t += "Method & " + " & ".join([f"$\\mu$={c:.1f}" for c in piv_nmi.columns]) + " \\\\\n\\midrule\n"
for m in ORDER:
    t += m + " & " + " & ".join([f"{piv_nmi.loc[m, c]:.3f}" for c in piv_nmi.columns]) + " \\\\\n"
t += "\\midrule\n"
for m in ORDER:
    t += m + " & " + " & ".join([f"{piv_ari.loc[m, c]:.3f}" for c in piv_ari.columns]) + " \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n"
w("tab2_lfr.tex", t)

# ---------- Table 3: real networks, full metrics ----------
df = pd.read_csv(os.path.join(RES, "real_results.csv"))
labeled = ["Karate", "Dolphins", "Polbooks", "Football", "Polblogs"]
methods = ORDER
t = "\\begin{tabular}{llccccc}\n\\toprule\nDataset & Method & NMI & ARI & F1 & Purity & $Q$ \\\\\n\\midrule\n"
for d in labeled:
    sub = df[df.dataset == d]
    t += "\\midrule\n" if d != labeled[0] else ""
    for m in methods:
        r = sub[sub.method == m]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        t += f"{d} & {m} & {r['NMI']:.3f} & {r['ARI']:.3f} & {r['F1']:.3f} & {r['Purity']:.3f} & {r['Q']:.3f} \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n"
w("tab3_real.tex", t)

# ---------- Table 4: modularity on unlabeled networks ----------
t = "\\begin{tabular}{lcc}\n\\toprule\nMethod & Adjnoun $Q$ & Lesmis $Q$ \\\\\n\\midrule\n"
for m in ["Louvain", "Leiden", "Infomap", "Greedy", "LPA"]:
    vals = []
    for d in ["Adjnoun", "Lesmis"]:
        r = df[(df.dataset == d) & (df.method == m)]
        vals.append(f"{r.iloc[0]['Q']:.3f}" if len(r) else "--")
    t += f"{m} & {vals[0]} & {vals[1]} \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n"
w("tab4_unlabeled.tex", t)

# ---------- Table 5: ablation ----------
df = pd.read_csv(os.path.join(RES, "ablation_results.csv"))
variants = ["PHSCD (full)", "w/o persistence", "w/o seeds", "w/o refinement", "w/o bias (p=q=1)"]
t = "\\begin{tabular}{llccc}\n\\toprule\nDataset & Variant & NMI & ARI & F1 \\\\\n\\midrule\n"
for d in ["Karate", "Dolphins", "Polbooks", "Football", "Polblogs", "LFR(mu=0.4)"]:
    sub = df[df.dataset == d]
    for i, v in enumerate(variants):
        r = sub[sub.variant == v].iloc[0]
        dn = d if i == 0 else ""
        t += f"{dn} & {v} & {r['NMI']:.3f} & {r['ARI']:.3f} & {r['F1']:.3f} \\\\\n"
    t += "\\midrule\n"
t = t.rsplit("\\midrule", 1)[0] + "\\bottomrule\n\\end{tabular}\n"
w("tab5_ablation.tex", t)

# ---------- Table 6: hyperparameters ----------
t = """\\begin{tabular}{lll}
\\toprule
Symbol & Meaning & Value \\\\\n\\midrule
$d$ & embedding dimension & 64 \\\\
$r$ & walks per node & 10 \\\\
$\\ell$ & walk length & 40 \\\\
$w$ & skip-gram window & 10 \\\\
$p, q$ & return / in-out bias & 1.0, 2.0 \\\\
$n_{neg}$ & negative samples & 5 \\\\
$k_{nn}$ & kNN sparsification & 10 \\\\
$n_{cand}$ & persistence candidates & 6 \\\\
$\\rho$ & seed ratio & 10\\% \\\\
$\\tau_q$ & margin quantile & 0.30 \\\\
$T$ & max refinement sweeps & 10 \\\\
\\bottomrule
\\end{tabular}
"""
w("tab6_params.tex", t)

# ---------- Table 7: runtime ----------
df = pd.read_csv(os.path.join(RES, "runtime_results.csv"))
piv = df.pivot_table(index="method", columns="n", values="runtime")
t = "\\begin{tabular}{l" + "c" * len(piv.columns) + "}\n\\toprule\n"
t += "Method & " + " & ".join([f"$n$={c}" for c in piv.columns]) + " \\\\\n\\midrule\n"
for m in ["Louvain", "Leiden", "Infomap", "Spectral", "N2V+KM", "PHSCD"]:
    t += m + " & " + " & ".join([f"{piv.loc[m, c]:.2f}" for c in piv.columns]) + " \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n"
w("tab7_runtime.tex", t)
print("ALL TABLES DONE")
