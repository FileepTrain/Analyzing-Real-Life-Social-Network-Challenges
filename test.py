import os
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
DATA_FILE = "facebook_combined.txt"   # combined edge list
CIRCLES_DIR = "facebook"              # folder with 0.circles, 107.circles, ...


# ---------------------------------------------------------------------
# COMBINED GRAPH ANALYSIS
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

    # Connected components
    comps = list(nx.connected_components(G))
    print(f"Number of connected components: {len(comps)}")
    largest_cc = max(comps, key=len)
    print(f"Largest component size: {len(largest_cc)}")

    # Subgraph of the largest connected component (for distances)
    Gcc = G.subgraph(largest_cc).copy()

    # Clustering
    avg_clust = nx.average_clustering(Gcc)
    print(f"Average clustering coefficient (largest CC): {avg_clust:.4f}")

    # Distances (only on largest CC)
    print("[INFO] Computing diameter and average shortest path on largest CC (may take a bit)...")
    diameter = nx.diameter(Gcc)
    aspl = nx.average_shortest_path_length(Gcc)
    print(f"Diameter (largest CC): {diameter}")
    print(f"Average shortest path length (largest CC): {aspl:.4f}")
    print()


def degree_analysis(G: nx.Graph, top_k: int = 10):
    print("===== DEGREE / INFLUENCE (COMBINED GRAPH) =====")
    degrees = dict(G.degree())
    # Top-degree nodes
    top_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(f"Top {top_k} nodes by degree (node, degree):")
    for node, deg in top_degree:
        print(f"  {node}: {deg}")
    print()

    # Degree distribution plot
    print("[INFO] Plotting degree distribution ...")
    degree_values = list(degrees.values())
    counts = Counter(degree_values)
    x = sorted(counts.keys())
    y = [counts[d] for d in x]

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title("Degree distribution (Facebook combined)")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("degree_distribution_loglog.png")
    plt.close()
    print("[INFO] Saved: degree_distribution_loglog.png\n")


def pagerank_analysis(G: nx.Graph, top_k: int = 10):
    print("===== PAGERANK (INFLUENCE, COMBINED GRAPH) =====")
    print("[INFO] Computing PageRank (pure Python, no SciPy)...")

    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-06)

    top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(f"Top {top_k} nodes by PageRank (node, score):")
    for node, score in top_pr:
        print(f"  {node}: {score:.6f}")
    print()


def betweenness_analysis(G: nx.Graph, top_k: int = 10):
    """
    Compute betweenness centrality and print top-k nodes.

    NOTE: Exact betweenness on the full graph is okay for 4k nodes,
    but if it ever feels slow you can switch to the 'k=' parameter
    for approximation.
    """
    print("===== BETWEENNESS CENTRALITY (COMBINED GRAPH) =====")
    print("[INFO] Computing betweenness centrality (this may take a bit)...")

    # Exact betweenness:
    # bc = nx.betweenness_centrality(G, normalized=True, endpoints=False)
    # If you want faster approximate betweenness, uncomment this instead:
    # bc = nx.betweenness_centrality(G, k=500, normalized=True, endpoints=False, seed=42)

    bc = nx.betweenness_centrality(G, normalized=True, endpoints=False)

    top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(f"Top {top_k} nodes by betweenness centrality (node, score):")
    for node, score in top_bc:
        print(f"  {node}: {score:.6f}")
    print()


def community_analysis(G: nx.Graph):
    """
    Use greedy modularity communities (built into NetworkX) to find clusters.
    Interpretable as potential 'echo chambers' in the network.
    """
    from networkx.algorithms.community import greedy_modularity_communities

    print("===== COMMUNITY / ECHO CHAMBERS (COMBINED GRAPH) =====")
    print("[INFO] Detecting communities using greedy modularity ...")
    communities = list(greedy_modularity_communities(G))
    print(f"Number of communities found: {len(communities)}")

    # Sizes of communities
    sizes = sorted([len(c) for c in communities], reverse=True)
    print("Sizes of the 10 largest communities:")
    for i, s in enumerate(sizes[:10], start=1):
        print(f"  Community {i}: {s} nodes")

    # Simple bar plot of community sizes (top 20)
    plt.figure()
    plt.bar(range(1, min(21, len(sizes)) + 1), sizes[:20])
    plt.xlabel("Community rank (by size)")
    plt.ylabel("Number of nodes")
    plt.title("Largest communities in Facebook combined graph")
    plt.tight_layout()
    plt.savefig("community_sizes_top20.png")
    plt.close()
    print("[INFO] Saved: community_sizes_top20.png\n")


def extra_epidemic_numbers(G: nx.Graph):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_deg = 2 * m / n
    print("===== EPIDEMIC-STYLE NUMBERS (COMBINED GRAPH) =====")
    print(f"Average degree (k̄): {avg_deg:.2f}")
    # Example hypothetical transmission probability
    p = 0.05
    R0_info = p * avg_deg
    print(f"Assuming p = {p}, info R0 ≈ {R0_info:.2f}")
    print()


# ---------------------------------------------------------------------
# CIRCLE-LEVEL CLUSTERING + DENSITY ANALYSIS
# ---------------------------------------------------------------------
def circle_clustering_and_density_analysis(G: nx.Graph, circles_dir: str = CIRCLES_DIR):
    """
    Compare global clustering/density with clustering and density inside user-defined circles.
    Also reports how many circles there are in total.
    """
    print("===== CIRCLE-LEVEL CLUSTERING & DENSITY (USER-DEFINED GROUPS) =====")
    if not os.path.isdir(circles_dir):
        print(f"[WARN] Circles directory '{circles_dir}' not found; skipping circle analysis.\n")
        return

    # Global metrics over full graph (for comparison)
    global_clust = nx.average_clustering(G)
    global_density = nx.density(G)
    print(f"Global average clustering (whole graph): {global_clust:.4f}")
    print(f"Global density (whole graph): {global_density:.6f}")

    circle_stats = []

    for fname in os.listdir(circles_dir):
        if not fname.endswith(".circles"):
            continue

        ego_id = fname.split(".")[0]  # e.g., "107" from "107.circles"
        path = os.path.join(circles_dir, fname)

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) <= 1:
                    continue
                circle_name = parts[0].rstrip(":")  # e.g. "circle0:"
                try:
                    nodes = [int(x) for x in parts[1:]]
                except ValueError:
                    nodes = [int(x) for x in parts[1:] if x.isdigit()]

                # Keep only nodes that are in the combined graph
                nodes_in_G = [n for n in nodes if G.has_node(n)]
                size = len(nodes_in_G)

                if size < 2:
                    clust = 0.0
                    density = 0.0
                else:
                    H = G.subgraph(nodes_in_G).copy()
                    clust = nx.average_clustering(H)
                    density = nx.density(H)

                circle_stats.append(
                    {
                        "ego": ego_id,
                        "circle": circle_name,
                        "size": size,
                        "clustering": clust,
                        "density": density,
                    }
                )

    if not circle_stats:
        print("[WARN] No circles found in directory; skipping.\n")
        return

    # Basic aggregates
    num_circles = len(circle_stats)
    sizes = [c["size"] for c in circle_stats]
    clusts = [c["clustering"] for c in circle_stats]
    densities = [c["density"] for c in circle_stats]

    avg_circle_size = sum(sizes) / num_circles
    avg_circle_clust = sum(clusts) / num_circles
    avg_circle_density = sum(densities) / num_circles

    print(f"\nTotal number of circles analyzed: {num_circles}")
    print(f"Average circle size: {avg_circle_size:.2f}")
    print(f"Average circle clustering: {avg_circle_clust:.4f}")
    print(f"Average circle density: {avg_circle_density:.6f}")

    # Top / bottom 5 circles by clustering
    circle_by_clust = sorted(circle_stats, key=lambda x: x["clustering"], reverse=True)
    print("\nTop 5 most clustered circles (strong echo chambers):")
    for c in circle_by_clust[:5]:
        print(
            f"  Ego {c['ego']} {c['circle']}: "
            f"size={c['size']}, clustering={c['clustering']:.4f}, density={c['density']:.4f}"
        )

    print("\nBottom 5 least clustered circles (more open groups):")
    for c in circle_by_clust[-5:]:
        print(
            f"  Ego {c['ego']} {c['circle']}: "
            f"size={c['size']}, clustering={c['clustering']:.4f}, density={c['density']:.4f}"
        )

    # Top 5 densest circles
    circle_by_density = sorted(circle_stats, key=lambda x: x["density"], reverse=True)
    print("\nTop 5 most dense circles (many internal edges):")
    for c in circle_by_density[:5]:
        print(
            f"  Ego {c['ego']} {c['circle']}: "
            f"size={c['size']}, clustering={c['clustering']:.4f}, density={c['density']:.4f}"
        )

    # Histogram of circle clustering values
    plt.figure()
    plt.hist(clusts, bins=20)
    plt.xlabel("Circle clustering coefficient")
    plt.ylabel("Number of circles")
    plt.title("Distribution of circle-level clustering")
    plt.tight_layout()
    plt.savefig("circle_clustering_hist.png")
    plt.close()
    print("\n[INFO] Saved: circle_clustering_hist.png")

    # Histogram of circle density values
    plt.figure()
    plt.hist(densities, bins=20)
    plt.xlabel("Circle density")
    plt.ylabel("Number of circles")
    plt.title("Distribution of circle-level density")
    plt.tight_layout()
    plt.savefig("circle_density_hist.png")
    plt.close()
    print("[INFO] Saved: circle_density_hist.png\n")


# ---------------------------------------------------------------------
# FULL GRAPH PLOT WITH ONE CIRCLE HIGHLIGHTED
# ---------------------------------------------------------------------
def load_ego_circles(ego_id: str, circles_dir: str = CIRCLES_DIR):
    """Return (circle_names, list_of_node_lists) for a given ego."""
    path = os.path.join(circles_dir, f"{ego_id}.circles")
    names = []
    circles = []
    if not os.path.exists(path):
        return names, circles

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
            names.append(circle_name)
            circles.append(nodes)

    return names, circles


def plot_graph_with_circle(
    G: nx.Graph,
    ego_id: str = "107",
    circle_index: int = 0,
    circles_dir: str = CIRCLES_DIR,
):
    """
    Plot the full combined graph, highlighting one circle:
    - Circle nodes in red
    - All other nodes in light gray
    """
    print("===== PLOTTING FULL GRAPH WITH ONE CIRCLE HIGHLIGHTED =====")

    if not os.path.isdir(circles_dir):
        print(f"[WARN] Circles directory '{circles_dir}' not found; skipping graph plot.\n")
        return

    circle_names, circle_lists = load_ego_circles(ego_id, circles_dir)
    if not circle_lists:
        print(f"[WARN] No circles found for ego {ego_id}; plotting plain graph.\n")
        circle_nodes = set()
        circle_label = "none"
    else:
        if circle_index < 0 or circle_index >= len(circle_lists):
            circle_index = 0
        circle_nodes = set(node for node in circle_lists[circle_index] if G.has_node(node))
        circle_label = circle_names[circle_index]
        print(
            f"[INFO] Highlighting ego {ego_id} {circle_label} "
            f"(size {len(circle_nodes)} nodes) in the full graph."
        )

    # Layout for full graph (4039 nodes) – spring_layout is okay but may take a bit
    print("[INFO] Computing layout for full graph (this may take a moment)...")
    pos = nx.spring_layout(G, seed=42)

    # Colors: red in circle, lightgray otherwise
    node_colors = [
        "red" if node in circle_nodes else "lightgray"
        for node in G.nodes()
    ]

    node_sizes = [
        30 if node in circle_nodes else 10
        for node in G.nodes()
    ]

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, linewidths=0)

    plt.axis("off")
    plt.title(f"Facebook combined graph with ego {ego_id} {circle_label} highlighted")
    plt.tight_layout()
    out_name = f"graph_with_circle_ego{ego_id}_{circle_label}.png"
    plt.savefig(out_name, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {out_name}\n")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    G = load_graph(DATA_FILE)
    basic_stats(G)
    degree_analysis(G)
    pagerank_analysis(G)
    betweenness_analysis(G)  # <-- new
    community_analysis(G)
    extra_epidemic_numbers(G)
    circle_clustering_and_density_analysis(G)
    # Plot full graph, highlighting one circle (ego 107, first circle by default)
    plot_graph_with_circle(G, ego_id="107", circle_index=0)
    print("[DONE] Analysis complete. Check the printed stats and PNG files for plots.")


if __name__ == "__main__":
    main()
