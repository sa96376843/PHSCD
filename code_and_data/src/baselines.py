# -*- coding: utf-8 -*-
"""Baseline community detection methods and evaluation metrics."""
import sys as _sys
_sys.path.append('/mnt/agents/pylibs')
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
import community as community_louvain
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import (normalized_mutual_info_score, adjusted_rand_score,
                             f1_score, confusion_matrix)
from scipy.optimize import linear_sum_assignment


# ----------------------------- baselines ----------------------------------
def run_louvain(G, seed=0):
    p = community_louvain.best_partition(G, random_state=seed)
    return np.array([p[v] for v in G.nodes()])


def run_leiden(G, seed=0):
    g = ig.Graph.from_networkx(G)
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed)
    return np.array(part.membership)


def run_infomap(G, seed=0):
    g = ig.Graph.from_networkx(G)
    part = g.community_infomap()
    return np.array(part.membership)


def run_greedy(G, seed=0):
    comms = nx.community.greedy_modularity_communities(G)
    assign = np.empty(G.number_of_nodes(), dtype=int)
    for i, c in enumerate(comms):
        for v in c:
            assign[v] = i
    return assign


def run_lpa(G, seed=0):
    comms = nx.community.asyn_lpa_communities(G, seed=seed)
    assign = np.full(G.number_of_nodes(), -1)
    for i, c in enumerate(comms):
        for v in c:
            assign[v] = i
    return assign


def run_spectral(G, k, seed=0):
    A = nx.to_numpy_array(G)
    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                            assign_labels="kmeans", random_state=seed)
    A_aff = A / (A.max() + 1e-12)
    return sc.fit_predict(A_aff)


def run_n2v_kmeans(G, k, seed=0, **kw):
    """The pipeline of Davison et al. (NeurIPS 2024): node2vec + k-means."""
    from phscd import node2vec_embed
    nodes, Z = node2vec_embed(G, seed=seed, **kw)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


# ----------------------------- metrics ------------------------------------
def pairwise_f1(y_true, y_pred):
    """Pair-counting F1 computed from the contingency table (vectorised)."""
    C = confusion_matrix(y_true, y_pred)
    comb2 = lambda x: x * (x - 1) / 2
    tp = comb2(C).sum()
    same_pred = comb2(C.sum(axis=0)).sum()
    same_true = comb2(C.sum(axis=1)).sum()
    fp, fn = same_pred - tp, same_true - tp
    return 2 * tp / (2 * tp + fp + fn + 1e-12)


def purity(y_true, y_pred):
    C = confusion_matrix(y_true, y_pred)
    return C.max(axis=0).sum() / len(y_true)


def modularity(G, assign):
    comms = {}
    for v, c in zip(G.nodes(), assign):
        comms.setdefault(c, set()).add(v)
    return nx.community.modularity(G, list(comms.values()))


def evaluate_all(G, y_true, assign):
    out = {"NMI": normalized_mutual_info_score(y_true, assign),
           "ARI": adjusted_rand_score(y_true, assign),
           "F1":  pairwise_f1(y_true, assign),
           "Purity": purity(y_true, assign),
           "Q": modularity(G, assign)}
    return out
