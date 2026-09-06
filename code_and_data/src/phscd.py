# -*- coding: utf-8 -*-
"""
PHSCD: Persistent-Homology-guided Semi-supervised Community Detection
via biased random-walk embeddings.

Pipeline:
  (1) node2vec biased random walks + Skip-gram with negative sampling (gensim)
  (2) 0-dimensional persistent homology on the embedding similarity filtration
      -> data-driven estimation of the number of communities k
  (3) seed-guided constrained clustering (labeled nodes keep their labels)
  (4) margin-based boundary refinement driven by local modularity gain
"""
import sys as _sys
_sys.path.append('/mnt/agents/pylibs')
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans


# --------------------------------------------------------------------------
# 1. Biased random-walk embedding (node2vec, Grover & Leskovec 2016)
# --------------------------------------------------------------------------
def _biased_walks(G, num_walks, walk_length, p, q, seed=0):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    nbrs = {v: list(G.neighbors(v)) for v in nodes}
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for v0 in nodes:
            walk = [v0]
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_nbrs = nbrs[cur]
                if not cur_nbrs:
                    break
                if len(walk) == 1:
                    walk.append(cur_nbrs[rng.integers(len(cur_nbrs))])
                else:
                    prev = walk[-2]
                    probs = np.empty(len(cur_nbrs))
                    for i, nb in enumerate(cur_nbrs):
                        if nb == prev:
                            probs[i] = 1.0 / p
                        elif G.has_edge(nb, prev):
                            probs[i] = 1.0
                        else:
                            probs[i] = 1.0 / q
                    probs /= probs.sum()
                    walk.append(cur_nbrs[rng.choice(len(cur_nbrs), p=probs)])
            walks.append([str(x) for x in walk])
    return walks


def node2vec_embed(G, dim=64, num_walks=10, walk_length=80, p=1.0, q=1.0,
                   window=10, epochs=5, seed=0):
    from gensim.models import Word2Vec
    walks = _biased_walks(G, num_walks, walk_length, p, q, seed)
    model = Word2Vec(walks, vector_size=dim, window=window, min_count=0,
                     sg=1, negative=5, hs=0, workers=1, epochs=epochs, seed=seed)
    nodes = list(G.nodes())
    Z = np.vstack([model.wv[str(v)] for v in nodes])
    return nodes, Z


# --------------------------------------------------------------------------
# 2. Zero-dimensional persistent homology -> estimate number of communities
# --------------------------------------------------------------------------
def _cosine_sim(Z):
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Zn @ Zn.T


def persistence_candidates(Z, k_max=20, n_neighbors=10, n_cand=6):
    """0-dim persistent homology of the embedding similarity filtration.
    The merge tree of a finite metric space equals its MST; component
    lifetimes are the MST edge weights. Candidate community counts are
    derived from the largest 'persistence gaps' of the descending weight
    sequence; the final k is selected by modularity maximisation."""
    n = Z.shape[0]
    S = _cosine_sim(Z)
    D = 1.0 - S
    k_nn = min(n_neighbors, n - 1)
    knn = np.full_like(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k_nn + 1]
    rows = np.repeat(np.arange(n), k_nn + 1)
    knn[rows, idx.ravel()] = D[rows, idx.ravel()]
    knn = np.minimum(knn, knn.T)
    mst = minimum_spanning_tree(csr_matrix(knn)).tocoo()
    w = np.sort(mst.data)
    w = w[w > 1e-12]
    if len(w) < 2:
        return [2], w
    desc = w[::-1]
    cand = min(k_max, len(desc) - 1)
    gaps = desc[:cand] - desc[1:cand + 1]
    order = np.argsort(gaps)[::-1][:n_cand]
    ks = sorted({int(i) + 2 for i in order} | {2})
    return ks, w


# --------------------------------------------------------------------------
# 3. Seed-guided constrained k-means
# --------------------------------------------------------------------------
def seeded_kmeans(Z, seeds, labels, k, n_init=10, seed=0, max_iter=100):
    """seeds: indices of labeled nodes; labels: their community ids (0..k-1).
    Labeled points are never reassigned; centers initialised at seed centroids."""
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    centers = np.zeros((k, Z.shape[1]))
    seed_mask = np.zeros(n, dtype=bool)
    seed_mask[seeds] = True
    for c in range(k):
        pts = Z[np.asarray(seeds)[np.asarray(labels) == c]]
        centers[c] = pts.mean(axis=0) if len(pts) else Z[rng.integers(n)]
    assign = np.full(n, -1)
    assign[seeds] = labels
    for _ in range(max_iter):
        d2 = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_assign = d2.argmin(1)
        new_assign[seeds] = labels                  # hard constraint
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(k):
            pts = Z[assign == c]
            if len(pts):
                centers[c] = pts.mean(0)
    return assign


# --------------------------------------------------------------------------
# 4. Margin-based boundary refinement via local modularity gain
# --------------------------------------------------------------------------
def _modularity_gain(G, v, comm, assign, m2, degrees):
    """Gain of assigning v to community comm: proportional to
    A(v,comm) - gamma * d_v * vol(comm) / 2m  (resolution gamma=1)."""
    nb_comm = {}
    for u in G.neighbors(v):
        nb_comm[assign[u]] = nb_comm.get(assign[u], 0) + 1
    vol = {}
    for u in G.nodes():
        vol[assign[u]] = vol.get(assign[u], 0) + degrees[u]
    return nb_comm.get(comm, 0) - degrees[v] * vol.get(comm, 0) / m2


def boundary_refine(G, Z, assign, seeds, labels, margin_pct=0.3, max_iter=10):
    n = Z.shape[0]
    k = assign.max() + 1
    centers = np.vstack([Z[assign == c].mean(0) for c in range(k)])
    S = _cosine_sim(np.vstack([Z]))
    Sc = _cosine_sim(np.vstack([centers]))
    d = 1 - Z @ (centers.T / (np.linalg.norm(centers, axis=1) + 1e-12))
    part = np.partition(d, 1, axis=1)
    margins = part[:, 1] - part[:, 0]
    thr = np.quantile(margins, margin_pct)
    boundary = np.where(margins <= thr)[0]
    boundary = [v for v in boundary if v not in set(seeds)]
    degrees = dict(G.degree())
    m2 = 2.0 * G.number_of_edges()
    nodes = list(G.nodes())
    idx_of = {v: i for i, v in enumerate(nodes)}
    changed, it = True, 0
    while changed and it < max_iter:
        changed = False
        it += 1
        for v in boundary:
            best_c, best_g = assign[v], 0.0
            cand = {assign[u] for u in G.neighbors(nodes[v])}
            for c in cand:
                g = _modularity_gain(G, nodes[v], c, assign, m2, degrees)
                cur = _modularity_gain(G, nodes[v], assign[v], assign, m2, degrees)
                if g - cur > best_g + 1e-9:
                    best_c, best_g = c, g - cur
            if best_c != assign[v]:
                assign[v] = best_c
                changed = True
    return assign, it


# --------------------------------------------------------------------------
# Full algorithm
# --------------------------------------------------------------------------
def _nx_modularity(G, assign):
    comms = {}
    for v, c in zip(G.nodes(), assign):
        comms.setdefault(int(c), set()).add(v)
    return nx.community.modularity(G, list(comms.values()))


def phscd(G, seeds=None, labels=None, k_true=None, dim=64, p=1.0, q=1.0,
          num_walks=10, walk_length=40, use_persistence=True,
          use_seeds=True, use_refine=True, seed=0, Z=None):
    """Returns (assignment array aligned with list(G.nodes()), estimated k).
    Pass precomputed embeddings via Z to skip the (expensive) walk/training
    stage when running multiple configurations on the same graph."""
    if Z is None:
        nodes, Z = node2vec_embed(G, dim=dim, num_walks=num_walks,
                                  walk_length=walk_length, p=p, q=q, seed=seed)
    else:
        nodes = list(G.nodes())
    n = len(nodes)
    has_seeds = bool(use_seeds and seeds is not None and len(seeds))
    if has_seeds:
        # Semi-supervised mode: the number of distinct seed labels is a strong
        # closed-world prior on k; persistence candidates serve the
        # unsupervised variant (ablated below).
        k = len(set(labels))
        assign = seeded_kmeans(Z, seeds, labels, k, seed=seed)
    else:
        # Unsupervised mode: persistence-derived candidates + modularity choice
        if use_persistence:
            cand, _ = persistence_candidates(Z, k_max=max(4, int(np.sqrt(n))))
        else:
            cand = [k_true if k_true is not None else 8]
        best = (None, -np.inf, None)
        for k in cand:
            a = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
            q_mod = _nx_modularity(G, a)
            if q_mod > best[1]:
                best = (a, q_mod, k)
        assign, _, k = best
    if use_refine:
        assign, _ = boundary_refine(G, Z, assign,
                                    seeds if (use_seeds and seeds is not None) else [],
                                    labels if labels is not None else [])
    return assign, k, Z
