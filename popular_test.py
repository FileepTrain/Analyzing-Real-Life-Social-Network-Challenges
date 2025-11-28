import os
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_FILE = "facebook_combined.txt"  # combined graph edge list
CIRCLES_DIR = "facebook"             # folder containing *.circles files
TOP_K = 10                           # how many top PageRank nodes to analyze


# ---------------------------------------------------------------------
# LOADING GRAPH
# ---------------------------------------------------------------------
def load_graph(path: str) -> nx.Graph:
    print(f"[INFO] Loading graph from {path} ...")
    G = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())
    print(f"[INFO] Done. Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}\n")
    return G


# ---------------------------------------------------------------------
# COMMUNITY DETECTION
# ---------------------------------------------------------------------
def compute_communities(G: nx.Graph):
    """
    Detect communities using greedy modularity and build:
      - communities: list of sets of nodes
      - node_to_comm: dict node -> community index
      - comm_stats: dict comm_id -> {'size', 'edges', 'density', 'avg_clustering'}
    """
    print("===== COMMUNITY DETECTION (COMBINED GRAPH) =====")
    print("[INFO] Detecting communities using greedy modularity ...")
    communities = list(greedy_modularity_communities(G))
    print(f"[INFO] Number of communities found: {len(communities)}")

    node_to_comm = {}
    for idx, comm in enumerate(communities):
        for n in comm:
            node_to_comm[n] = idx

    # Precompute stats for each community
    comm_stats = {}
    for idx, comm_nodes in enumerate(communities):
        H = G.subgraph(comm_nodes).copy()
        size = H.number_of_nodes()
        edges = H.number_of_edges()
        density = nx.density(H) if size > 1 else 0.0
        avg_clust = nx.average_clustering(H) if size > 1 else 0.0
        comm_stats[idx] = {
            "size": size,
            "edges": edges,
            "density": density,
            "avg_clustering": avg_clust,
        }

    # Show sizes of largest communities (by size)
    sizes_sorted = sorted((s["size"] for s in comm_stats.values()), reverse=True)
    print("Sizes of the 10 largest communities (by size):")
    for i, s in enumerate(sizes_sorted[:10], start=1):
        print(f"  Community {i}: {s} nodes")
    print()

    return communities, node_to_comm, comm_stats


# ---------------------------------------------------------------------
# PAGERANK
# ---------------------------------------------------------------------
def compute_pagerank(G: nx.Graph, top_k: int):
    """
    Compute PageRank (pure Python implementation from NetworkX)
    and return:
      - pr: dict node -> PageRank score
      - top_nodes: list of (node, score) pairs for top_k
    """
    print("===== PAGERANK (POPULARITY) =====")
    print("[INFO] Computing PageRank (pure Python, no SciPy)...")
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-06)

    top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(f"Top {top_k} nodes by PageRank (node, score):")
    for node, score in top_nodes:
        print(f"  {node}: {score:.6f}")
    print()

    return pr, top_nodes


# ---------------------------------------------------------------------
# CIRCLES LOADING
# ---------------------------------------------------------------------
def load_ego_circles(ego_id: int, circles_dir: str = CIRCLES_DIR):
    """
    For a given ego (node id), load its circles from <ego>.circles if present.

    Returns:
      list of dicts: [
        {
          'circle_name': 'circle0',
          'nodes': [int, int, ...]
        },
        ...
      ]
    """
    fname = os.path.join(circles_dir, f"{ego_id}.circles")
    circles = []

    if not os.path.exists(fname):
        return circles

    with open(fname, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 1:
                continue
            circle_name = parts[0].rstrip(":")
            try:
                nodes = [int(x) for x in parts[1:]]
            except ValueError:
                nodes = [int(x) for x in parts[1:] if x.isdigit()]
            circles.append({"circle_name": circle_name, "nodes": nodes})

    return circles


def analyze_circle_subgraph(G: nx.Graph, nodes):
    """
    Given a list of node IDs and the full graph G,
    compute stats for the induced subgraph:
      - size
      - edges
      - density
      - average clustering
    Only nodes present in G are used.
    """
    nodes_in_G = [n for n in nodes if G.has_node(n)]
    size = len(nodes_in_G)
    if size < 2:
        return {
            "size": size,
            "edges": 0,
            "density": 0.0,
            "avg_clustering": 0.0,
        }
    H = G.subgraph(nodes_in_G).copy()
    edges = H.number_of_edges()
    density = nx.density(H)
    avg_clust = nx.average_clustering(H)
    return {
        "size": size,
        "edges": edges,
        "density": density,
        "avg_clustering": avg_clust,
    }


# ---------------------------------------------------------------------
# MAIN ANALYSIS: POPULAR NODES + THEIR CONTEXT
# ---------------------------------------------------------------------
def analyze_popular_nodes_context(G: nx.Graph, top_nodes, node_to_comm, comm_stats):
    """
    For each popular node:
      - print its degree, PageRank, local clustering
      - print stats of its community (size, density, clustering)
      - if it has circles, print stats for each circle
    """
    print("===== POPULAR NODES: COMMUNITY & CIRCLE CONTEXT =====\n")

    for node, score in top_nodes:
        print(f"--- Popular node {node} ---")
        deg = G.degree[node]
        local_clust = nx.clustering(G, node)
        print(f"PageRank: {score:.6f}")
        print(f"Degree: {deg}")
        print(f"Local clustering coefficient: {local_clust:.4f}")

        # Community info
        comm_id = node_to_comm.get(node, None)
        if comm_id is not None:
            cs = comm_stats[comm_id]
            print(f"Community ID: {comm_id}")
            print(
                f"Community size: {cs['size']} | "
                f"edges: {cs['edges']} | "
                f"density: {cs['density']:.6f} | "
                f"avg clustering: {cs['avg_clustering']:.4f}"
            )
        else:
            print("Community ID: None (node not assigned to any community?)")

        # Circle info (if node is an ego with a .circles file)
        circles = load_ego_circles(node, CIRCLES_DIR)
        if not circles:
            print("Circles: none found for this node (no <node>.circles file)\n")
            continue

        print(f"Circles for ego {node}: {len(circles)} total")
        for c in circles:
            stats = analyze_circle_subgraph(G, c["nodes"])
            print(
                f"  {c['circle_name']}: "
                f"size={stats['size']}, "
                f"edges={stats['edges']}, "
                f"density={stats['density']:.6f}, "
                f"avg_clustering={stats['avg_clustering']:.4f}"
            )
        print()  # blank line between nodes


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] {DATA_FILE} not found in current directory.")
        return
    if not os.path.isdir(CIRCLES_DIR):
        print(f"[WARN] Circles directory '{CIRCLES_DIR}' not found. Circle analysis will be skipped.")

    G = load_graph(DATA_FILE)
    communities, node_to_comm, comm_stats = compute_communities(G)
    pr, top_nodes = compute_pagerank(G, TOP_K)
    analyze_popular_nodes_context(G, top_nodes, node_to_comm, comm_stats)
    print("[DONE] Popular node context analysis complete.")


if __name__ == "__main__":
    main()
