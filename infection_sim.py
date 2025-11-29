#!/usr/bin/env python3
import argparse
import json
import os
import random

import networkx as nx

# ---------------------------------------------------------------------
# DEFAULT CONFIG (can be overridden via CLI)
# ---------------------------------------------------------------------
DATA_FILE = "facebook_combined.txt"
CIRCLES_DIR = "facebook"
STATS_JSON = "centrality_stats.json"

DEFAULT_BETA = 0.2      # default infection probability
DEFAULT_GAMMA = 0.1     # default recovery probability
DEFAULT_RUNS = 200      # default number of random experiments
MAX_STEPS = 50          # max time steps for SIR


# ---------------------------------------------------------------------
# GRAPH LOADING
# ---------------------------------------------------------------------
def load_graph(path: str) -> nx.Graph:
    print(f"[INFO] Loading graph from {path} ...")
    G = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())
    print(f"[INFO] Done. Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}\n")
    return G


# ---------------------------------------------------------------------
# SIR INFECTION DYNAMICS
# ---------------------------------------------------------------------
def sir_simulation(G: nx.Graph, seeds, immune=None,
                   beta: float = DEFAULT_BETA, gamma: float = DEFAULT_GAMMA,
                   max_steps: int = MAX_STEPS):
    """
    Run a simple SIR infection process on graph G.

    States:
      S = susceptible
      I = infected (currently spreading)
      R = recovered/immune (cannot be infected anymore)

    Args:
      G: undirected graph
      seeds: initial infected nodes (iterable)
      immune: nodes that start as R and can never be infected (iterable or None)
      beta: probability that an infected node infects each susceptible neighbor per step
      gamma: probability that an infected node recovers per step
      max_steps: safety cap on the number of time steps

    Returns:
      infected_ever: set of all nodes that were ever infected (I at some point)
      final_R: set of nodes in state R at the end
      final_I: set of nodes still infected at the end (usually empty)
    """
    seeds = set(seeds)
    immune = set(immune) if immune is not None else set()

    # Initialize all nodes as susceptible
    S = set(G.nodes())
    I = set()
    R = set()

    # Immune nodes start recovered
    R |= immune
    S -= immune

    # Infect initial seeds (if not immune)
    seeds = seeds - R
    I |= seeds
    S -= seeds

    infected_ever = set(I)

    for _ in range(max_steps):
        if not I:
            break

        new_I = set()
        new_R = set()

        # Infection step
        for u in I:
            for v in G.neighbors(u):
                if v in S and random.random() < beta:
                    new_I.add(v)

        # Recovery step
        for u in I:
            if random.random() < gamma:
                new_R.add(u)

        # Update states
        S -= new_I
        I |= new_I
        I -= new_R
        R |= new_R

        infected_ever |= new_I

    return infected_ever, R, I


def simulate_sir_size(G: nx.Graph, seed, immune=None,
                      beta: float = DEFAULT_BETA, gamma: float = DEFAULT_GAMMA,
                      max_steps: int = MAX_STEPS) -> int:
    infected_ever, _, _ = sir_simulation(G, [seed], immune, beta, gamma, max_steps)
    return len(infected_ever)


def simulate_sir_size_multi(G: nx.Graph, seeds, immune=None,
                            beta: float = DEFAULT_BETA, gamma: float = DEFAULT_GAMMA,
                            max_steps: int = MAX_STEPS) -> int:
    infected_ever, _, _ = sir_simulation(G, seeds, immune, beta, gamma, max_steps)
    return len(infected_ever)


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
        print(f"[WARN] Circles directory '{circles_dir}' not found; skipping circle-based infection experiments.")
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
# INFECTION EXPERIMENTS (ANALOGOUS TO CASCADE TESTS)
# ---------------------------------------------------------------------
def infection_tests_split(
    G: nx.Graph,
    deg_top,
    pr_top,
    bc_top,
    beta: float,
    gamma: float,
    num_random_runs: int,
):
    random.seed(42)

    print("===== INFECTION TESTS (SIR MODEL) WITH TOP-20 CENTRAL NODES =====")
    print(f"Parameters: beta={beta}, gamma={gamma}, max_steps={MAX_STEPS}, runs={num_random_runs}\n")

    deg_nodes = {n for n, _ in deg_top}
    pr_nodes = {n for n, _ in pr_top}
    bc_nodes = {n for n, _ in bc_top}

    all_seeds = deg_nodes | pr_nodes | bc_nodes

    # ---------- BASELINE: RANDOM NODE INFECTION SIZE ----------
    print(">>> BASELINE: RANDOM SINGLE-NODE SEEDS (no immune core)")
    all_nodes = list(G.nodes())
    total_baseline = 0
    for _ in range(num_random_runs):
        seed = random.choice(all_nodes)
        size = simulate_sir_size(G, seed, immune=None, beta=beta, gamma=gamma)
        total_baseline += size
    avg_baseline = total_baseline / num_random_runs
    print(f"Average infection size (random seed baseline): {avg_baseline:.2f}\n")

    # ---------- TOP-20 CENTRAL NODES INFECTION POWER ----------
    print(">>> CENTRAL NODES: INFECTION SIZES (no immune core)")

    infection_sizes = {
        node: simulate_sir_size(G, node, immune=None, beta=beta, gamma=gamma)
        for node in all_seeds
    }

    avg_central = sum(infection_sizes.values()) / len(infection_sizes)
    print(f"Average infection size (all top-20 central nodes as seeds): {avg_central:.2f}")
    print(f"Central / baseline ratio: {avg_central / avg_baseline:.2f}\n")

    # Grouping like in cascade_sim.py
    all_three = deg_nodes & pr_nodes & bc_nodes
    deg_pr = (deg_nodes & pr_nodes) - all_three
    deg_bc = (deg_nodes & bc_nodes) - all_three
    pr_bc = (pr_nodes & bc_nodes) - all_three

    only_deg = deg_nodes - (pr_nodes | bc_nodes)
    only_pr = pr_nodes - (deg_nodes | bc_nodes)
    only_bc = bc_nodes - (deg_nodes | pr_nodes)

    print(">>> NODES IN ALL THREE LISTS (Degree, PageRank, Betweenness)")
    for n in sorted(all_three, key=lambda x: infection_sizes[x], reverse=True):
        print(f"  {n}: infection size = {infection_sizes[n]}")
    print()

    print(">>> NODES IN EXACTLY TWO LISTS")

    def show(name, S):
        print(f"  {name}:")
        for n in sorted(S, key=lambda x: infection_sizes[x], reverse=True):
            print(f"    {n}: infection size = {infection_sizes[n]}")

    show("Degree + PageRank", deg_pr)
    show("Degree + Betweenness", deg_bc)
    show("PageRank + Betweenness", pr_bc)
    print()

    print(">>> NODES IN ONLY ONE LIST")
    show("Degree only", only_deg)
    show("PageRank only", only_pr)
    show("Betweenness only", only_bc)
    print()

    # ---------- JOINT SEEDING OF ALL-THREE ----------
    print(">>> JOINT SEEDING OF ALL-THREE NODES (SIR, no immune core)")
    if all_three:
        joint_sz = simulate_sir_size_multi(G, all_three, immune=None, beta=beta, gamma=gamma)
        print(f"  Seeds {sorted(all_three)} → infection size = {joint_sz}\n")
    else:
        print("  No nodes in the intersection of all three lists.\n")

    # ---------- RANDOM SINGLE-SEED: NORMAL vs STUBBORN (IMMUNE CORE) ----------
    print("===== RANDOM SINGLE-SEED INFECTION: NORMAL VS STUBBORN ALL-THREE =====")
    if not all_three:
        print("No nodes are in all three lists; cannot define stubborn core.\n")
        return

    immune_core = set(all_three)
    candidate_seeds = [v for v in all_nodes if v not in immune_core]

    total_normal = 0
    total_stubborn = 0

    for _ in range(num_random_runs):
        seed = random.choice(candidate_seeds)

        normal_size = simulate_sir_size(G, seed, immune=None, beta=beta, gamma=gamma)
        stubborn_size = simulate_sir_size(G, seed, immune=immune_core, beta=beta, gamma=gamma)

        total_normal += normal_size
        total_stubborn += stubborn_size

    avg_normal = total_normal / num_random_runs
    avg_stubborn = total_stubborn / num_random_runs

    print(f"Using {num_random_runs} random single seeds (excluding immune core)")
    print(f"Average infection size (NORMAL):   {avg_normal:.2f}")
    print(f"Average infection size (STUBBORN): {avg_stubborn:.2f}")
    print(f"Difference (normal - stubborn):    {avg_normal - avg_stubborn:.2f}\n")

    # ---------- RANDOM CIRCLE-SEED: NORMAL vs STUBBORN ----------
    print("===== RANDOM CIRCLE-SEED INFECTION: NORMAL VS STUBBORN ALL-THREE =====")

    circles = load_all_circles(CIRCLES_DIR)
    if not circles:
        print("No circles available; skipping circle-based infection experiment.\n")
        return

    # Filter circles that overlap with G and contain at least one non-immune node
    valid_circles = []
    for nodes in circles:
        nodes_in_G = [v for v in nodes if G.has_node(v)]
        if not nodes_in_G:
            continue
        if all(v in immune_core for v in nodes_in_G):
            continue
        valid_circles.append(nodes_in_G)

    if not valid_circles:
        print("No valid circles with non-immune nodes; skipping circle-based infection.\n")
        return

    total_circle_normal = 0
    total_circle_stubborn = 0

    for _ in range(num_random_runs):
        seeds = random.choice(valid_circles)

        normal_size = simulate_sir_size_multi(G, seeds, immune=None, beta=beta, gamma=gamma)
        stubborn_size = simulate_sir_size_multi(G, seeds, immune=immune_core, beta=beta, gamma=gamma)

        total_circle_normal += normal_size
        total_circle_stubborn += stubborn_size

    avg_circle_normal = total_circle_normal / num_random_runs
    avg_circle_stubborn = total_circle_stubborn / num_random_runs

    print(f"Using {num_random_runs} random circle seeds")
    print(f"Average infection size (NORMAL, circle-seed):   {avg_circle_normal:.2f}")
    print(f"Average infection size (STUBBORN, circle-seed): {avg_circle_stubborn:.2f}")
    print(f"Difference (normal - stubborn):                {avg_circle_normal - avg_circle_stubborn:.2f}\n")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SIR infection simulations on Facebook graph.")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA,
                        help=f"Infection probability per edge per step (default={DEFAULT_BETA})")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help=f"Recovery probability per infected per step (default={DEFAULT_GAMMA})")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"Number of random experiments for averages (default={DEFAULT_RUNS})")
    args = parser.parse_args()

    beta = args.beta
    gamma = args.gamma
    num_runs = args.runs

    # Load precomputed centrality stats from basic_stats.py
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

    infection_tests_split(G, deg_top, pr_top, bc_top,
                          beta=beta, gamma=gamma, num_random_runs=num_runs)

    print("[DONE infection_sim.py]")


if __name__ == "__main__":
    main()
