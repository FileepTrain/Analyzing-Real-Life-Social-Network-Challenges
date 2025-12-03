#!/usr/bin/env python3
import json

import networkx as nx

DATA_FILE = "facebook_combined.txt"
TOP_K = 20  # top-k for degree, PageRank, betweenness
STATS_JSON = "centrality_stats.json"


# ---------------------------------------------------------------------
# BASIC GRAPH STATS
# ---------------------------------------------------------------------
def load_graph(path: str) -> nx.Graph:
    print(f"[INFO] Loading graph from {path} ...")
    G = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())
    print("[INFO] Done.\n")
    return G


def basic_stats(G: nx.Graph):
    print("===== BASIC STATS (COMBINED GRAPH) =====")
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)

    print(f"Nodes: {n}")
    print(f"Edges: {m}")
    print(f"Density: {density:.6f}")

    avg_clust = nx.average_clustering(G)
    print(f"Average clustering coefficient: {avg_clust:.4f}\n")


# ---------------------------------------------------------------------
# CENTRALITIES
# ---------------------------------------------------------------------
def degree_analysis(G: nx.Graph, top_k: int = TOP_K):
    print("===== DEGREE / INFLUENCE =====")
    degrees = dict(G.degree())
    top_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"Top {top_k} nodes by degree:")
    for node, deg in top_degree:
        print(f"  {node}: {deg}")
    print()

    return top_degree, degrees


def pagerank_analysis(G: nx.Graph, top_k: int = TOP_K):
    print("===== PAGERANK =====")
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-06)
    top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"Top {top_k} nodes by PageRank:")
    for node, score in top_pr:
        print(f"  {node}: {score:.6f}")
    print()

    return top_pr, pr


def betweenness_analysis(G: nx.Graph, top_k: int = TOP_K):
    print("===== BETWEENNESS CENTRALITY =====")
    bc = nx.betweenness_centrality(G, normalized=True)
    top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"Top {top_k} nodes by betweenness:")
    for node, score in top_bc:
        print(f"  {node}: {score:.6f}")
    print()

    return top_bc, bc


# ---------------------------------------------------------------------
# EPIDEMIC NUMBER R0
# ---------------------------------------------------------------------
def extra_epidemic_numbers(G: nx.Graph):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_deg = 2 * m / n

    print("===== EPIDEMIC-STYLE R0 =====")
    print(f"Average degree (k̄): {avg_deg:.2f}")
    p = 0.05
    R0 = p * avg_deg
    print(f"Assuming p = {p}, R0 ≈ {R0:.2f}\n")

    # return so we can also store in JSON
    return avg_deg, p, R0


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    G = load_graph(DATA_FILE)
    basic_stats(G)

    deg_top, degrees = degree_analysis(G)
    pr_top,  pr      = pagerank_analysis(G)
    bc_top,  bc      = betweenness_analysis(G)

    avg_deg, p, R0 = extra_epidemic_numbers(G)

    payload = {
        "data_file": DATA_FILE,
        "top_k": TOP_K,
        "avg_degree": avg_deg,
        "p_share": p,
        "R0": R0,
        "deg_top": [[int(n), int(d)] for n, d in deg_top],
        "pr_top":  [[int(n), float(s)] for n, s in pr_top],
        "bc_top":  [[int(n), float(s)] for n, s in bc_top],

        # NEW: full centrality dicts for cascade_super_sim.py
        "pagerank":    {int(n): float(s) for n, s in pr.items()},
        "betweenness": {int(n): float(s) for n, s in bc.items()},
    }


    with open(STATS_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Saved centrality stats to {STATS_JSON}")
    print("[DONE basic_stats.py]")


if __name__ == "__main__":
    main()
