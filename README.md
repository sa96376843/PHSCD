# PHSCD

**Persistent-Homology-guided Semi-supervised Community Detection in Complex Networks**

PHSCD is a community-detection framework for networks in which only a small subset of nodes is labelled and the number of communities may be unknown. It combines biased random-walk embeddings, zero-dimensional persistent homology, seed-constrained clustering, and topology-guided boundary refinement.

This repository contains the implementation, network datasets, experiment scripts, plotting utilities, and result tables used in the accompanying manuscript.

## Method overview

PHSCD consists of four stages:

1. **Node embedding:** generate second-order biased random walks and learn node representations with Skip-gram.
2. **Community-number estimation:** construct a cosine-distance graph, compute its minimum spanning tree, and use zero-dimensional persistence gaps to generate candidate community numbers.
3. **Seed-guided clustering:** initialise cluster centres from labelled seeds and keep their assignments fixed during constrained clustering.
4. **Boundary refinement:** identify low-margin nodes and accept reassignments that improve local modularity.

The implementation also supports **PHSCD-U**, an unsupervised variant that estimates the community number from persistence candidates and selects among them using modularity.

## Repository structure

```text
PHSCD/
├── README.md
└── code_and_data/
    ├── data/          # Bundled GML networks and source archives
    ├── results/       # Experiment results in CSV format
    └── src/
        ├── phscd.py               # Core PHSCD implementation
        ├── datasets.py            # Real-network and LFR loaders
        ├── baselines.py           # Baselines and evaluation metrics
        ├── exp_*.py               # Experiment scripts
        ├── fig*.py                 # Figure-generation scripts
        └── make_tables*.py         # LaTeX and Markdown table generators
```

Generated figures and tables are written to `code_and_data/figures/`, `code_and_data/tables/`, and `code_and_data/tables_md/`.

## Installation

The experiments were developed for Python 3.12. A clean environment is strongly recommended.

```bash
conda create -n phscd python=3.12 -y
conda activate phscd
python -m pip install numpy==1.26.4 scipy==1.12.0 pandas==2.2.2 networkx==3.3 scikit-learn==1.5.1 gensim==4.3.3 matplotlib==3.9.1 python-igraph==0.11.6 leidenalg==0.10.2 python-louvain==0.16
```

`scipy==1.12.0` is recommended because later SciPy releases removed `scipy.linalg.triu`, which is imported by Gensim 4.3.x.

Clone the repository and enter its root directory:

```bash
git clone https://github.com/sa96376843/PHSCD.git
cd PHSCD
```

## Quick start

The following example runs PHSCD on Zachary's Karate Club network with one labelled seed from each reference community:

```python
import sys
from pathlib import Path

import numpy as np

src = Path("code_and_data/src").resolve()
sys.path.insert(0, str(src))

import datasets
from phscd import phscd

G, y = datasets.load_karate()

seed_nodes = []
seed_labels = []
for label in np.unique(y):
    node = int(np.flatnonzero(y == label)[0])
    seed_nodes.append(node)
    seed_labels.append(int(label))

assignment, k_hat, embedding = phscd(
    G,
    seeds=seed_nodes,
    labels=seed_labels,
    p=1.0,
    q=2.0,
    num_walks=10,
    walk_length=40,
    seed=0,
)

print("Estimated communities:", k_hat)
print("Assignments:", assignment)
print("Embedding shape:", embedding.shape)
```

For fully unsupervised operation:

```python
assignment, k_hat, embedding = phscd(
    G,
    use_seeds=False,
    p=1.0,
    q=2.0,
    num_walks=10,
    walk_length=40,
    seed=0,
)
```

## Reproducing the experiments

Run all commands from the repository root. The scripts write directly to `code_and_data/results/` and may overwrite the included CSV files. Preserve a copy of the published results before rerunning experiments with modified code or parameters.

### Real-world networks

```bash
python code_and_data/src/exp_real.py
python code_and_data/src/exp_unlabeled.py
```

`exp_real.py` evaluates the supervised and standard baseline methods. Run `exp_unlabeled.py` afterwards to append the PHSCD-U results for networks without reference labels.

### LFR benchmark

```bash
python code_and_data/src/exp_lfr.py
```

The script evaluates mixing parameters `0.1` to `0.7` with random seeds `0`, `1`, and `2`. The auxiliary script `exp_lfr_resume.py` is only intended to resume missing cells at `mu=0.7`.

### Ablation study

```bash
python code_and_data/src/exp_ablation.py
```

The evaluated variants remove persistence-based model selection, seed guidance, boundary refinement, or random-walk bias.

### Parameter sensitivity

```bash
python code_and_data/src/exp_sensitivity.py
```

This script evaluates the node2vec parameters `(p, q)` and labelled-seed ratios from 0% to 20%.

### Scalability

```bash
python code_and_data/src/exp_runtime.py
```

This experiment evaluates LFR networks with 500, 1000, 2000, and 4000 nodes. The embedding-based experiments are CPU-intensive and can take minutes to hours depending on the machine.

## Generating tables

Generate LaTeX table fragments:

```bash
python code_and_data/src/make_tables.py
```

Generate Markdown versions of the same tables:

```bash
python code_and_data/src/make_tables_md.py
```

## Generating figures

Run the framework script first to create the output directory, followed by the data-driven figure scripts:

```bash
python code_and_data/src/fig1_framework.py
python code_and_data/src/fig2_persistence.py
python code_and_data/src/fig3_fig4_lfr.py
python code_and_data/src/fig5_tsne.py
python code_and_data/src/fig6_ablation.py
python code_and_data/src/fig7_fig8_sensitivity.py
python code_and_data/src/fig9_karate.py
python code_and_data/src/fig10_runtime.py
```

The plotting scripts read the committed CSV files in `code_and_data/results/` and export both PNG and PDF files where implemented.

## Datasets

The repository contains seven real-world networks:

| Dataset | Source used by the loader | Reference labels |
|---|---|---|
| Karate | NetworkX built-in dataset | Club split |
| Dolphins | `dolphins.gml` | Girvan-Newman first bisection |
| Polbooks | `polbooks.gml` | Political category |
| Football | `football.gml` | Conference |
| Polblogs | `polblogs.gml` | Political leaning |
| Adjnoun | `adjnoun.gml` | Not provided |
| Lesmis | `lesmis.gml` | Not provided |

Synthetic LFR networks are generated at runtime with `networkx.LFR_benchmark_graph`. Users should consult the accompanying manuscript and the original dataset sources before redistributing third-party data.

## Outputs

| File | Contents |
|---|---|
| `results/lfr_results.csv` | LFR accuracy and runtime results |
| `results/real_results.csv` | Results on the seven real-world networks |
| `results/ablation_results.csv` | Component ablation results |
| `results/sensitivity_pq.csv` | Random-walk parameter sensitivity |
| `results/sensitivity_seedratio*.csv` | Seed-ratio sensitivity |
| `results/runtime_results.csv` | Scalability measurements |

The main evaluation metrics are normalized mutual information (NMI), adjusted Rand index (ARI), pairwise F1, purity, and modularity.

## Reproducibility notes

- Stochastic experiments use fixed random seeds, typically `0`, `1`, and `2`.
- Gensim is configured with `workers=1` to reduce nondeterminism.
- Runtime values depend on hardware and software versions and should not be expected to match exactly across machines.
- The committed CSV files are the archived outputs associated with the manuscript experiments.

## Citation

If this repository supports your research, please cite the accompanying manuscript. Publication metadata will be added after publication.

```bibtex
@unpublished{li2026phscd,
  title  = {PHSCD: Persistent-Homology-Guided Semi-Supervised Community Detection in Complex Networks},
  author = {Li, Xiaoming and Feng, Jianhang and Chen, Deng and Bai, Hongpeng},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

No explicit software license is currently included in this repository. Please contact the authors before reusing or redistributing the code. Dataset reuse remains subject to the terms of the original data providers.
