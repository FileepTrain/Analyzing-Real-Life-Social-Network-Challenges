#!/usr/bin/env python3
"""
Analyze each Facebook circle individually.

Assumptions:
- facebook_combined.txt is in the same directory as this script
- There is a subdirectory called "facebook" containing:
    0.circles, 107.circles, 348.circles, ...
  each in SNAP's format:
    circleX: node1 node2 node3 ...

For each circle we compute:
- size (nodes in the combined graph)
- number of internal edges
- density
- average clustering coefficient
- average degree
- an epidemic-style info R0 = p * avg_degree (with p = 0.05)
"""

import os
import csv
import networkx as nx

DATA_FILE = "facebook_combined.txt"
CIRCLES_DIR = "facebook"   # folder with *.circles files
P_SHARE = 0.05             # hypothetical probability of sharing fake news


def load_graph(path: str) -> nx.Graph:
    """Load the combined Facebook graph from an edge list file."""
    print(f"[INFO] Loading graph from {path} ...")
    G = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())
    print(f"[INFO] Done. Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}\n")
    return G


def iter_circles(circles_dir: str):
    """
    Generator over all circles in all *.circles files.

    Yields dicts:
      {
        "ego": ego_id (str),
        "circle_name": "circle0", "circle1", ...,
        "raw_nodes": [list of node IDs as ints]
      }
    """
    for fname in os.listdir(circles_dir):
        if not fname.endswith(".circles"):
            continue
        ego_id = fname.split(".")[0]  # e.g. "107" from "107.circles"
        path = os.path.join(circles_dir, fname)
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) <= 1:
                    continue
                circle_name = parts[0].rstrip(":")
                try:
                    nodes = [int(x) for x in parts[1:]]
                except ValueError:
                    nodes = [int(x) for x in parts[1:] if x.isdigit()]
                yield {
                    "ego": ego_id,
                    "circle_name": circle_name,
                    "raw_nodes": nodes,
                }


def analyze_circle(G: nx.Graph, ego: str, circle_name: str, nodes):
    """
    Given the full graph G and a list of node IDs, compute stats
    for the induced subgraph on those nodes.
    """
    # Keep only nodes that actually exist in the combined graph
    nodes_in_G = [n for n in nodes if G.has_node(n)]
    size = len(nodes_in_G)

    if size == 0:
        # empty circle (w.r.t. combined graph)
        return {
            "ego": ego,
            "circle_name": circle_name,
            "size": 0,
            "edges": 0,
            "density": 0.0,
            "avg_clustering": 0.0,
            "avg_degree": 0.0,
            "R0_info": 0.0,
        }

    if size == 1:
        # single node: no edges
        return {
            "ego": ego,
            "circle_name": circle_name,
            "size": 1,
            "edges": 0,
            "density": 0.0,
            "avg_clustering": 0.0,
            "avg_degree": 0.0,
            "R0_info": 0.0,
        }

    H = G.subgraph(nodes_in_G).copy()
    edges = H.number_of_edges()
    density = nx.density(H)
    avg_clustering = nx.average_clustering(H)

    # average degree in H = 2m / n
    avg_degree = 2 * edges / size
    R0_info = P_SHARE * avg_degree

    return {
        "ego": ego,
        "circle_name": circle_name,
        "size": size,
        "edges": edges,
        "density": density,
        "avg_clustering": avg_clustering,
        "avg_degree": avg_degree,
        "R0_info": R0_info,
    }


def main():
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] {DATA_FILE} not found in current directory.")
        return
    if not os.path.isdir(CIRCLES_DIR):
        print(f"[ERROR] Circles directory '{CIRCLES_DIR}' not found.")
        return

    G = load_graph(DATA_FILE)

    all_stats = []

    print("===== PER-CIRCLE ANALYSIS =====")

    for circle in iter_circles(CIRCLES_DIR):
        ego = circle["ego"]
        circle_name = circle["circle_name"]
        nodes = circle["raw_nodes"]

        stats = analyze_circle(G, ego, circle_name, nodes)
        all_stats.append(stats)

        # Print a compact line per circle
        print(
            f"Ego {stats['ego']} {stats['circle_name']}: "
            f"size={stats['size']}, edges={stats['edges']}, "
            f"density={stats['density']:.4f}, "
            f"clustering={stats['avg_clustering']:.4f}, "
            f"avg_deg={stats['avg_degree']:.2f}, "
            f"R0_info≈{stats['R0_info']:.2f}"
        )

    if not all_stats:
        print("\n[WARN] No circles found to analyze.")
        return

    # Summary
    num_circles = len(all_stats)
    total_nodes = sum(s["size"] for s in all_stats)
    avg_size = total_nodes / num_circles

    avg_clust = sum(s["avg_clustering"] for s in all_stats) / num_circles
    avg_density = sum(s["density"] for s in all_stats) / num_circles
    avg_R0 = sum(s["R0_info"] for s in all_stats) / num_circles

    print("\n===== SUMMARY OVER ALL CIRCLES =====")
    print(f"Total circles analyzed: {num_circles}")
    print(f"Average circle size: {avg_size:.2f}")
    print(f"Average circle clustering: {avg_clust:.4f}")
    print(f"Average circle density: {avg_density:.6f}")
    print(f"Average info R0 across circles (p={P_SHARE}): {avg_R0:.2f}")

    # Save to CSV for further use
    out_csv = "circle_stats.csv"
    fieldnames = [
        "ego",
        "circle_name",
        "size",
        "edges",
        "density",
        "avg_clustering",
        "avg_degree",
        "R0_info",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_stats:
            writer.writerow(s)

    print(f"\n[INFO] Saved circle statistics to {out_csv}")


if __name__ == "__main__":
    main()
