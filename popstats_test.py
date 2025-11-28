import collections
import random

import networkx as nx
import matplotlib.pyplot as plt

DATA_FILE = "facebook_combined.txt"
TOP_K = 20  # top-k for degree, PageRank, betweenness
NUM_RANDOM_RUNS = 200  # number of random cascades for stats


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
    print(f"Assuming p = {p}, R0 ≈ {p * avg_deg:.2f}\n")


# ---------------------------------------------------------------------
# CASCADE SIMULATIONS
# ---------------------------------------------------------------------
def cascade_return_adopted(G, seed, q=0.2, max_steps=50):
    """
    Run a threshold cascade and return the set of adopted nodes.
    """
    adopted = {seed}

    for _ in range(max_steps):
        new = set()

        for node in G.nodes():
            if node in adopted:
                continue
            neigh = list(G.neighbors(node))
            if not neigh:
                continue
            active = sum(1 for v in neigh if v in adopted)
            if active / len(neigh) >= q:
                new.add(node)

        if not new:
            break

        adopted |= new

    return adopted


def cascade_return_adopted_with_blocked(G, seed, blocked, q=0.2, max_steps=50):
    """
    Threshold cascade where 'blocked' nodes never adopt.
    Return the set of adopted nodes.
    """
    blocked = set(blocked)
    if seed in blocked:
        return set()

    adopted = {seed}

    for _ in range(max_steps):
        new = set()

        for node in G.nodes():
            if node in adopted or node in blocked:
                continue
            neigh = list(G.neighbors(node))
            if not neigh:
                continue
            active = sum(1 for v in neigh if v in adopted)
            if active / len(neigh) >= q:
                new.add(node)

        if not new:
            break

        adopted |= new

    return adopted


def simulate_threshold_cascade(G, seed, q=0.2, max_steps=50):
    return len(cascade_return_adopted(G, seed, q=q, max_steps=max_steps))


def simulate_threshold_cascade_multi(G, seeds, q=0.2, max_steps=50):
    """
    Seeds is an iterable of starting active nodes.
    """
    adopted = set(seeds)

    if not adopted:
        return 0

    for _ in range(max_steps):
        new = set()

        for node in G.nodes():
            if node in adopted:
                continue
            neigh = list(G.neighbors(node))
            if not neigh:
                continue
            active = sum(1 for v in neigh if v in adopted)
            if active / len(neigh) >= q:
                new.add(node)

        if not new:
            break

        adopted |= new

    return len(adopted)


# ---------------------------------------------------------------------
# CASCADE GROUP TESTING
# ---------------------------------------------------------------------
def cascade_tests_split(G, deg_top, pr_top, bc_top, q=0.2):
    print("===== CASCADE TESTS (TOP-20 CENTRAL NODES) =====")

    deg_nodes = {n for n, _ in deg_top}
    pr_nodes = {n for n, _ in pr_top}
    bc_nodes = {n for n, _ in bc_top}

    all_seeds = deg_nodes | pr_nodes | bc_nodes

    # Precompute cascade sizes for these central nodes
    cascade_sizes = {node: simulate_threshold_cascade(G, node, q=q) for node in all_seeds}

    # Grouping
    all_three = deg_nodes & pr_nodes & bc_nodes
    deg_pr = (deg_nodes & pr_nodes) - all_three
    deg_bc = (deg_nodes & bc_nodes) - all_three
    pr_bc = (pr_nodes & bc_nodes) - all_three

    only_deg = deg_nodes - (pr_nodes | bc_nodes)
    only_pr = pr_nodes - (deg_nodes | bc_nodes)
    only_bc = bc_nodes - (deg_nodes | pr_nodes)

    # ---------- PRINT: CENTRAL NODE CASCADE POWER ----------
    print(">>> NODES IN ALL THREE LISTS (Degree, PageRank, Betweenness)")
    for n in sorted(all_three, key=lambda x: cascade_sizes[x], reverse=True):
        print(f"  {n}: cascade size = {cascade_sizes[n]}")
    print()

    print(">>> NODES IN EXACTLY TWO LISTS")

    def show(name, S):
        print(f"  {name}:")
        for n in sorted(S, key=lambda x: cascade_sizes[x], reverse=True):
            print(f"    {n}: cascade size = {cascade_sizes[n]}")

    show("Degree + PageRank", deg_pr)
    show("Degree + Betweenness", deg_bc)
    show("PageRank + Betweenness", pr_bc)
    print()

    print(">>> NODES IN ONLY ONE LIST")
    show("Degree only", only_deg)
    show("PageRank only", only_pr)
    show("Betweenness only", only_bc)
    print()

    # ---------- JOINT SEEDING ----------
    print(">>> JOINT SEEDING OF ALL-THREE NODES")
    joint_sz = simulate_threshold_cascade_multi(G, all_three, q=q)
    print(f"  Seeds {sorted(all_three)} → cascade size = {joint_sz}\n")

    # ---------- RANDOM CASCADE STATS (NORMAL vs STUBBORN) ----------
    print("===== RANDOM CASCADE STATS: NORMAL VS STUBBORN ALL-THREE =====")
    if not all_three:
        print("No nodes are in all three lists; cannot define stubborn core.\n")
        return

    blocked = set(all_three)
    all_nodes = list(G.nodes())
    candidate_seeds = [v for v in all_nodes if v not in blocked]

    random.seed(42)

    total_size_normal = 0
    total_size_blocked = 0

    infection_count_normal = {v: 0 for v in all_nodes}
    infection_count_blocked = {v: 0 for v in all_nodes}

    for _ in range(NUM_RANDOM_RUNS):
        seed = random.choice(candidate_seeds)

        adopted_normal = cascade_return_adopted(G, seed, q=q)
        adopted_blocked = cascade_return_adopted_with_blocked(G, seed, blocked, q=q)

        total_size_normal += len(adopted_normal)
        total_size_blocked += len(adopted_blocked)

        for v in adopted_normal:
            infection_count_normal[v] += 1
        for v in adopted_blocked:
            infection_count_blocked[v] += 1

    n = len(all_nodes)
    avg_size_normal = total_size_normal / NUM_RANDOM_RUNS
    avg_size_blocked = total_size_blocked / NUM_RANDOM_RUNS

    print(f"Using {NUM_RANDOM_RUNS} random seeds (excluding all-three as seeds)")
    print(f"Average cascade size (NORMAL):  {avg_size_normal:.2f}")
    print(f"Average cascade size (STUBBORN all-three): {avg_size_blocked:.2f}")
    print(f"Difference (normal - stubborn): {avg_size_normal - avg_size_blocked:.2f}\n")

    # ---------- INFECTION PROBABILITY ANALYSIS (NORMAL CASE) ----------
    print("===== INFECTION PROBABILITIES (NORMAL CASE) =====")

    # infection probability per node
    prob_normal = {v: infection_count_normal[v] / NUM_RANDOM_RUNS for v in all_nodes}

    # average over all nodes
    avg_prob_all = sum(prob_normal.values()) / n

    # average for all-three nodes
    avg_prob_all_three = sum(prob_normal[v] for v in all_three) / len(all_three)

    # average for all other nodes
    others = [v for v in all_nodes if v not in all_three]
    avg_prob_others = sum(prob_normal[v] for v in others) / len(others)

    print(f"Average infection probability (all nodes):      {avg_prob_all:.4f}")
    print(f"Average infection probability (all-three set):  {avg_prob_all_three:.4f}")
    print(f"Average infection probability (other nodes):    {avg_prob_others:.4f}")
    print()
    print("Note: In the STUBBORN scenario, infection probability of the all-three nodes is 0 by definition.\n")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    G = load_graph(DATA_FILE)
    basic_stats(G)

    deg_top, _ = degree_analysis(G)
    pr_top, _ = pagerank_analysis(G)
    bc_top, _ = betweenness_analysis(G)

    extra_epidemic_numbers(G)

    cascade_tests_split(G, deg_top, pr_top, bc_top, q=0.2)

    print("[DONE]")


if __name__ == "__main__":
    main()
