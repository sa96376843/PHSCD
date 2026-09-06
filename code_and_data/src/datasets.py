# -*- coding: utf-8 -*-
"""Dataset loading utilities: real-world networks (Newman collection) and LFR benchmark."""
import sys as _sys
_sys.path.append('/mnt/agents/pylibs')
import os
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def load_gml(name, label_attr="value", directed_ok=False):
    """Load a GML network, return (undirected simple graph relabelled 0..n-1, labels or None)."""
    path = os.path.join(DATA, f"{name}.gml")
    if name == "polblogs":  # contains parallel edges -> parse via igraph
        import igraph as ig
        g = ig.Graph.Read_GML(path)
        g = g.as_undirected(combine_edges=None).simplify()
        labels_all = np.array(g.vs["value"])
        # keep giant connected component and align labels
        comps = g.components()
        giant_id = int(np.argmax(comps.sizes()))
        keep = [i for i, m in enumerate(comps.membership) if m == giant_id]
        g = g.induced_subgraph(keep)
        labels_raw = labels_all[keep]
        G = nx.Graph()
        G.add_nodes_from(range(g.vcount()))
        G.add_edges_from(g.get_edgelist())
        # drop any residual isolates and realign labels
        keep2 = [v for v in G.nodes() if G.degree(v) > 0]
        G = G.subgraph(keep2).copy()
        labels_raw = labels_raw[keep2]
        G = nx.convert_node_labels_to_integers(G)
        return G, LabelEncoder().fit_transform(labels_raw)
    G = nx.read_gml(path, label="id")
    if G.is_directed():
        G = G.to_undirected()
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    raw_labels = None
    if label_attr is not None:
        attrs = nx.get_node_attributes(G, label_attr)
        if len(attrs) == G.number_of_nodes():
            raw_labels = [attrs[v] for v in G.nodes()]
    G = nx.convert_node_labels_to_integers(G)
    labels = LabelEncoder().fit_transform(raw_labels) if raw_labels is not None else None
    return G, labels


def load_karate():
    G = nx.karate_club_graph()
    raw = [G.nodes[v]["club"] for v in G.nodes()]
    G = nx.convert_node_labels_to_integers(G)
    return G, LabelEncoder().fit_transform(raw)


def real_datasets():
    """Seven real-world networks; five with ground-truth labels."""
    out = {}
    G, y = load_karate();                       out["Karate"]    = (G, y)
    # Dolphins: ground truth not distributed in the GML metadata; following the
    # literature (e.g., Newman & Girvan 2004) we use the Girvan-Newman first
    # bisection as the reference division.
    Gd, _ = load_gml("dolphins", label_attr=None)
    split = next(nx.community.girvan_newman(Gd))
    ref = np.empty(Gd.number_of_nodes(), dtype=int)
    for i, c in enumerate(split):
        for v in c:
            ref[v] = i
    out["Dolphins"] = (Gd, ref)
    out["Polbooks"]  = load_gml("polbooks")     # 3 communities
    out["Football"]  = load_gml("football")     # 12 conferences
    out["Polblogs"]  = load_gml("polblogs")     # 2 political leanings
    out["Adjnoun"]   = load_gml("adjnoun", label_attr=None)   # no ground truth
    out["Lesmis"]    = load_gml("lesmis",  label_attr=None)   # no ground truth
    return out


def lfr_benchmark(mu, n=1000, seed=0):
    """LFR benchmark graph with ground-truth communities (Lancichinetti et al. 2008)."""
    G = nx.LFR_benchmark_graph(n, tau1=2.5, tau2=1.5, mu=mu, average_degree=15,
                               max_degree=50, min_community=20, max_community=100,
                               seed=seed)
    G = nx.Graph(G)
    comms = list(nx.get_node_attributes(G, "community").values())
    y = np.empty(len(comms), dtype=int)
    mapping = {}
    for i, c in enumerate(comms):
        key = frozenset(c)
        if key not in mapping:
            mapping[key] = len(mapping)
        y[i] = mapping[key]
    return G, y
