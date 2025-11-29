#!/usr/bin/env python3
import argparse
import json
import os
import random

import networkx as nx

DATA_FILE = "facebook_combined.txt"
CIRCLES_DIR = "facebook"
STATS_JSON = "centrality_stats.json"
NUM_RANDOM_RUNS = 200  # number of random cascades for stats


# ---------------------------------------------------------------------
# GRAPH LOADING
# ---------------------------------------------------------------------
def load_graph(path: str) -> nx.Graph:
    print(f"[INFO] Loading graph from {path} ...")
    G = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())
    print("[INFO] Done.\n")
    return G


# ---------------------------------------------------------------------
# CASCADE SIMULATIONS (YOUR ORIGINAL LOGIC)
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


def simulate_threshold_cascade_multi_blocked(G, seeds, blocked, q=0.2, max_steps=50):
    """
    Multi-seed version with 'blocked' nodes that never adopt.
    """
    blocked = set(blocked)
    adopted = set(seeds) - blocked

    if not adopted:
        return 0

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

    return len(adopted)


# ---------------------------------------------------------------------
# CIRCLE LOADING
# ---------------------------------------------------------------------
def load_all_circles(circles_dir: str = CIRCLES_DIR):
    """
    Load all circles from all *.circles files.

    Returns a list of lists of node IDs (ints), one list per circle.
    """
    circles = []
    if not os.path.isdir(circles_dir):
        print(f"[WARN] Circles directory '{circles_dir}' not found; skipping circle-based cascades.")
        return circles

    for fname in os.listdir(circles_dir):
        if not fname.endswith(".circles"):
            continue
        path = os.path.join(circles_dir, fname)
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) <= 1:
                    continue
                try:
                    nodes = [int(x) for x in parts[1:]]
                except ValueError:
                    nodes = [int(x) for x in parts[1:] if x.isdigit()]
                if nodes:
                    circles.append(nodes)

    return circles


# ---------------------------------------------------------------------
# CASCADE GROUP TESTING (RESTORED + EXTENDED)
# ---------------------------------------------------------------------
def cascade_tests_split(G, deg_top, pr_top, bc_top, q=0.2):
    # ---------------- ORIGINAL TOP-20 CENTRALITY CASCADE TESTS ----------------
    print("===== CASCADE TESTS (TOP-20 CENTRAL NODES) =====")
    print(f"[INFO] Using threshold q = {q}\n")

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

    # ---------------- RANDOM SINGLE-SEED: NORMAL vs STUBBORN ----------------
    print("===== RANDOM CASCADE STATS: NORMAL VS STUBBORN ALL-THREE (SINGLE-SEED) =====")
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
    print("===== INFECTION PROBABILITIES (NORMAL CASE, SINGLE-SEED) =====")

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

    # ---------------- NEW: RANDOM CIRCLE-SEED CASCADE STATS ----------------
    print("===== RANDOM CIRCLE-SEED CASCADE STATS: NORMAL VS STUBBORN ALL-THREE =====")

    circles = load_all_circles(CIRCLES_DIR)
    if not circles:
        print("No circles available; skipping circle-based cascade experiment.\n")
        return

    # Filter circles to those that overlap G and have at least one non-blocked node
    valid_circles = []
    for nodes in circles:
        nodes_in_G = [v for v in nodes if G.has_node(v)]
        if not nodes_in_G:
            continue
        if all(v in blocked for v in nodes_in_G):
            continue
        valid_circles.append(nodes_in_G)

    if not valid_circles:
        print("No valid circles with non-blocked nodes; skipping circle-based experiment.\n")
        return

    total_circle_normal = 0
    total_circle_blocked = 0

    for _ in range(NUM_RANDOM_RUNS):
        seeds = random.choice(valid_circles)

        size_normal = simulate_threshold_cascade_multi(G, seeds, q=q)
        size_blocked = simulate_threshold_cascade_multi_blocked(G, seeds, blocked, q=q)

        total_circle_normal += size_normal
        total_circle_blocked += size_blocked

    avg_circle_normal = total_circle_normal / NUM_RANDOM_RUNS
    avg_circle_blocked = total_circle_blocked / NUM_RANDOM_RUNS

    print(f"Using {NUM_RANDOM_RUNS} random circle seeds")
    print(f"Average cascade size (NORMAL, circle-seed):   {avg_circle_normal:.2f}")
    print(f"Average cascade size (STUBBORN, circle-seed): {avg_circle_blocked:.2f}")
    print(f"Difference (normal - stubborn):              {avg_circle_normal - avg_circle_blocked:.2f}\n")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Threshold cascade simulations on Facebook graph.")
    parser.add_argument(
        "--q",
        type=float,
        default=0.2,
        help="Adoption threshold q (default=0.2)",
    )
    args = parser.parse_args()
    q = args.q

    # Load precomputed stats from basic_stats.py
    try:
        with open(STATS_JSON, "r") as f:
            stats = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] {STATS_JSON} not found. Run basic_stats.py first.")
        return

    deg_top = [(int(n), int(d)) for n, d in stats["deg_top"]]
    pr_top = [(int(n), float(s)) for n, s in stats["pr_top"]]
    bc_top = [(int(n), float(s)) for n, s in stats["bc_top"]]

    print("[INFO] Loaded centrality stats from basic_stats.py\n")

    G = load_graph(DATA_FILE)

    cascade_tests_split(G, deg_top, pr_top, bc_top, q=q)

    print("[DONE cascade_sim.py]")


if __name__ == "__main__":
    main()
