#!/usr/bin/env python3
import argparse
import json
import os
import random
from statistics import mean, stdev

import networkx as nx

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_FILE = "facebook_combined.txt"
CIRCLES_DIR = "facebook"
STATS_JSON = "centrality_stats.json"

DEFAULT_Q = 0.2
DEFAULT_Q_SUPER = 0.5  # higher adoption threshold for supernodes
DEFAULT_RANDOM_RUNS = 200
DEFAULT_PATIENCE = 50   # early-stop after this many consecutive non-super candidates; -1 = no early stop
DEFAULT_K_SIGMA = 2.0   # supernode threshold: mu + k * sigma
DEFAULT_SHADOW_FRAC = 0.3  # not used directly now; we sweep 0..1 but keep for CLI completeness


# ---------------------------------------------------------------------
# GRAPH LOADING
# ---------------------------------------------------------------------
def load_graph(path: str) -> nx.Graph:
    print(f"[INFO] Loading graph from {path} ...")
    G = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())
    print(f"[INFO] Done. Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}\n")
    return G


# ---------------------------------------------------------------------
# CASCADE SIMULATIONS
# ---------------------------------------------------------------------
def cascade_return_adopted(G, seed, q=0.2, max_steps=50):
    """
    Run a threshold cascade and return the set of adopted nodes.
    Adoption rule:
      node adopts if (active_neighbors / degree) >= q
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
# SUPERNODE FINDING WITH k-SIGMA THRESHOLD + EARLY STOPPING
# ---------------------------------------------------------------------
def find_cascade_supernodes(
    G,
    pagerank_dict,
    betweenness_dict,
    q=DEFAULT_Q,
    num_random_runs=DEFAULT_RANDOM_RUNS,
    patience=DEFAULT_PATIENCE,
    k_sigma=DEFAULT_K_SIGMA,
):
    """
    - Compute random baseline cascade sizes for num_random_runs seeds.
    - Let mu = mean, sigma = std dev of those sizes.
    - Threshold = mu + k_sigma * sigma.
    - Iterate down the full PageRank and betweenness rankings in parallel.
    - For each new candidate node, compute its cascade size.
    - Supernodes = those with cascade size >= threshold.
    - Stop once we've seen 'patience' consecutive candidates that are NOT supernodes.
      If patience == -1, disable early stopping and scan all ranked nodes.
    """
    random.seed(42)
    all_nodes = list(G.nodes())

    # ---- 1) Random baseline ----
    print("===== BASELINE: RANDOM SINGLE-SEED CASCADES =====")
    random_sizes = []
    for _ in range(num_random_runs):
        seed = random.choice(all_nodes)
        sz = simulate_threshold_cascade(G, seed, q=q)
        random_sizes.append(sz)

    if num_random_runs > 0:
        mu = mean(random_sizes)
    else:
        mu = 0.0

    if num_random_runs > 1:
        sigma = stdev(random_sizes)
    else:
        sigma = 0.0

    threshold = mu + k_sigma * sigma

    print(f"Using {num_random_runs} random seeds")
    print(f"Mean cascade size (mu):            {mu:.4f}")
    print(f"Std dev of cascade sizes (sigma):  {sigma:.4f}")
    print(f"k for k-sigma rule:                {k_sigma:.2f}")
    print(f"Supernode threshold = mu + k*sigma = {threshold:.4f}\n")

    # ---- 2) Build full ranked lists ----
    sorted_pr_nodes = [int(n) for n, _ in sorted(
        pagerank_dict.items(), key=lambda x: x[1], reverse=True
    )]
    sorted_bc_nodes = [int(n) for n, _ in sorted(
        betweenness_dict.items(), key=lambda x: x[1], reverse=True
    )]

    print("===== RANKED CANDIDATE NODES (FULL LISTS, WITH EARLY STOPPING) =====")
    print(f"Total nodes in graph:           {len(all_nodes)}")
    print(f"Nodes with PageRank entries:    {len(sorted_pr_nodes)}")
    print(f"Nodes with Betweenness entries: {len(sorted_bc_nodes)}")
    print(f"Early-stop patience:            {patience} "
          f"(use -1 for no early stop / full scan)\n")

    # ---- 3) Walk down rankings with early stopping (or full scan if patience == -1) ----
    supernodes = set()
    seen_candidates = set()
    consecutive_non_super = 0
    evaluated_count = 0

    i = 0
    max_len = max(len(sorted_pr_nodes), len(sorted_bc_nodes))

    if patience == -1:
        # No early stopping — scan the entire ranking
        stop_condition = lambda: i < max_len
    else:
        # Normal mode — stop after N consecutive non-super candidates
        stop_condition = lambda: i < max_len and consecutive_non_super < patience

    print("===== CANDIDATE CASCADE SIZES (IN RANK ORDER) =====")

    while stop_condition():
        for ranking_name, ranking_list in (("PR", sorted_pr_nodes), ("BC", sorted_bc_nodes)):
            if i >= len(ranking_list):
                continue
            v = ranking_list[i]
            if v in seen_candidates:
                continue

            seen_candidates.add(v)
            evaluated_count += 1

            sz = simulate_threshold_cascade(G, v, q=q)
            in_pr = v in pagerank_dict
            in_bc = v in betweenness_dict
            membership = []
            if in_pr:
                membership.append("PR")
            if in_bc:
                membership.append("BC")
            membership_str = "+".join(membership) if membership else "?"

            is_super = sz >= threshold
            tag = "<-- SUPER" if is_super else ""

            print(f"  RankStep {i:4d} | Node {v:4d} [{membership_str}] cascade size = {sz} {tag}")

            if is_super:
                supernodes.add(v)
                consecutive_non_super = 0
            else:
                consecutive_non_super += 1

            if patience != -1 and consecutive_non_super >= patience:
                break

        i += 1

    print("\n[INFO] Stopped scanning candidates.")
    print(f"  Total candidates evaluated:           {evaluated_count}")
    print(f"  Total unique candidates seen:         {len(seen_candidates)}")
    print(f"  Consecutive non-super streak at stop: {consecutive_non_super}\n")

    print("===== SUPER NODES (cascade size >= mu + k*sigma) =====")
    if not supernodes:
        print("No supernodes found under this definition.\n")
        return [], (mu, sigma, threshold)

    supernodes_sorted = sorted(supernodes)
    print(f"Supernodes (node IDs): {supernodes_sorted}")
    print(f"Total supernodes: {len(supernodes_sorted)} "
          f"(out of {len(seen_candidates)} evaluated candidates)")
    print(f"Threshold: cascade size >= {threshold:.4f}\n")

    return supernodes_sorted, (mu, sigma, threshold)


# ---------------------------------------------------------------------
# THRESHOLD-PROTECTION: HIGHER q FOR SUPERNODES
# ---------------------------------------------------------------------
def simulate_threshold_cascade_multi_super_protect_q(
    G,
    seeds,
    supernodes,
    q_normal=0.2,
    q_super=0.5,
    max_steps=50,
):
    """
    Multi-seed threshold cascade where:
      - Non-supernodes adopt if   active_neighbors / degree >= q_normal
      - Supernodes adopt if       active_neighbors / degree >= q_super

    This makes it harder for the cascade to 'reach' supernodes.
    """
    supernodes = set(supernodes)
    adopted = set(seeds)

    if not adopted:
        return 0

    neighbors = {v: list(G.neighbors(v)) for v in G.nodes()}

    for _ in range(max_steps):
        new = set()

        for node in G.nodes():
            if node in adopted:
                continue

            neigh = neighbors[node]
            if not neigh:
                continue

            active = sum(1 for v in neigh if v in adopted)
            threshold = q_super if node in supernodes else q_normal

            if active / len(neigh) >= threshold:
                new.add(node)

        if not new:
            break

        adopted |= new

    return len(adopted)


# ---------------------------------------------------------------------
# BRIDGE-SHADOW: ASYMMETRIC INCOMING BLOCKING TO SUPERNODES
# ---------------------------------------------------------------------
def simulate_threshold_cascade_multi_bridge_shadow_incoming(
    G,
    seeds,
    supernodes,
    shadow_block_order,
    drop_fraction,
    q=0.2,
    max_steps=50,
):
    """
    Multi-seed threshold cascade with ASYMMETRIC bridge-based shadow blocking
    of INCOMING influence to supernodes:

      - For each supernode s, we take the prefix of its neighbor-order list
        from shadow_block_order[s] of length floor(deg(s) * drop_fraction).
      - These neighbors' influence into s is ignored when computing s's
        adoption (s cannot "hear" them), BUT they still count in the
        denominator (degree) so adoption cannot become easier.
      - Edges are NOT removed from the graph; neighbors can still see s
        if s adopts (outgoing influence is intact).
    """
    supernodes = set(supernodes)
    adopted = set(seeds)
    if not adopted:
        return 0

    neighbors = {v: list(G.neighbors(v)) for v in G.nodes()}

    # Precompute which neighbors are blocked for each supernode at this fraction
    incoming_blocked = {s: set() for s in supernodes}
    for s in supernodes:
        neigh = neighbors.get(s, [])
        deg = len(neigh)
        if deg == 0:
            continue
        k_drop = int(deg * drop_fraction)
        if k_drop <= 0:
            continue

        # Order in shadow_block_order[s] may include nodes not in neigh if
        # the graph has changed; restrict to actual neighbors first.
        ordered_neighs = [u for u in shadow_block_order.get(s, []) if u in neigh]
        incoming_blocked[s] = set(ordered_neighs[:k_drop])

    for _ in range(max_steps):
        new = set()

        for node in G.nodes():
            if node in adopted:
                continue

            neigh = neighbors[node]
            if not neigh:
                continue

            degree = len(neigh)

            if node in supernodes:
                blocked_set = incoming_blocked.get(node, set())
                active = sum(1 for v in neigh if v in adopted and v not in blocked_set)
            else:
                active = sum(1 for v in neigh if v in adopted)

            if active / degree >= q:
                new.add(node)

        if not new:
            break

        adopted |= new

    return len(adopted)


# ---------------------------------------------------------------------
# SUPERNODE-BASED CASCADE EXPERIMENTS
# ---------------------------------------------------------------------
def supernode_cascade_experiments(
    G,
    supernodes,
    q=DEFAULT_Q,
    q_super=DEFAULT_Q_SUPER,
    num_random_runs=DEFAULT_RANDOM_RUNS,
    shadow_frac=DEFAULT_SHADOW_FRAC,  # kept for CLI symmetry; sweep ignores this value
):
    """
    After we have supernodes, run:
      - Joint seeding of all supernodes vs random nodes (same count)
      - Random single-seed NORMAL vs STUBBORN (supernodes blocked)
      - Infection probabilities (NORMAL case)
      - Random circle-seed NORMAL vs STUBBORN (supernodes)
      - CONTROL: random stubborn set with same size as supernodes (circle-seed)
      - THRESHOLD PROTECTION: higher q for supernodes
      - BRIDGE-SHADOW: asymmetric incoming blocking to supernodes, fraction sweep
    """
    if not supernodes:
        print("[INFO] No supernodes provided; skipping supernode cascade experiments.\n")
        return

    print("===== SUPERNODE-BASED CASCADE EXPERIMENTS =====")
    print(f"[INFO] Using q = {q}, q_super = {q_super}, runs = {num_random_runs}, shadow_frac (base) = {shadow_frac}\n")

    blocked = set(supernodes)
    all_nodes = list(G.nodes())
    candidate_seeds = [v for v in all_nodes if v not in blocked]

    # ---------- JOINT SEEDING OF SUPERNODES ----------
    print(">>> JOINT SEEDING OF SUPERNODES")
    joint_size = simulate_threshold_cascade_multi(G, supernodes, q=q)
    print(f"  Seeds (supernodes) {sorted(supernodes)}")
    print(f"  → joint cascade size (supernodes): {joint_size}\n")

    # ---------- JOINT SEEDING OF RANDOM NODES (SAME COUNT) ----------
    print(">>> JOINT SEEDING OF RANDOM NODES (SAME COUNT AS SUPERNODES)")
    k = len(supernodes)
    total_joint_random = 0
    for _ in range(num_random_runs):
        seeds_random = random.sample(all_nodes, k)
        total_joint_random += simulate_threshold_cascade_multi(G, seeds_random, q=q)
    avg_joint_random = total_joint_random / num_random_runs

    print(f"  Number of supernodes: {k}")
    print(f"  Single joint cascade size (supernodes):        {joint_size:.2f}")
    print(f"  Average joint cascade size (random {k}): {avg_joint_random:.2f}")
    print(f"  Difference (supernodes - random):              {joint_size - avg_joint_random:.2f}\n")

    # ---------- RANDOM SINGLE-SEED: NORMAL vs STUBBORN ----------
    print("===== RANDOM CASCADE STATS: NORMAL VS STUBBORN SUPERNODES (SINGLE-SEED) =====")
    random.seed(42)

    total_size_normal = 0
    total_size_blocked = 0

    infection_count_normal = {v: 0 for v in all_nodes}
    infection_count_blocked = {v: 0 for v in all_nodes}

    for _ in range(num_random_runs):
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
    avg_size_normal = total_size_normal / num_random_runs
    avg_size_blocked = total_size_blocked / num_random_runs

    print(f"Using {num_random_runs} random seeds (excluding supernodes as seeds)")
    print(f"Average cascade size (NORMAL):         {avg_size_normal:.2f}")
    print(f"Average cascade size (STUBBORN super): {avg_size_blocked:.2f}")
    print(f"Difference (normal - stubborn):        {avg_size_normal - avg_size_blocked:.2f}\n")

    # ---------- INFECTION PROBABILITIES (NORMAL CASE) ----------
    print("===== INFECTION PROBABILITIES (NORMAL CASE, SINGLE-SEED) =====")

    prob_normal = {v: infection_count_normal[v] / num_random_runs for v in all_nodes}

    avg_prob_all = sum(prob_normal.values()) / n
    avg_prob_super = sum(prob_normal[v] for v in supernodes) / len(supernodes)
    others = [v for v in all_nodes if v not in supernodes]
    avg_prob_others = sum(prob_normal[v] for v in others) / len(others)

    print(f"Average infection probability (all nodes):      {avg_prob_all:.4f}")
    print(f"Average infection probability (supernodes):     {avg_prob_super:.4f}")
    print(f"Average infection probability (other nodes):    {avg_prob_others:.4f}")
    print()
    print("Note: In the STUBBORN scenario, infection probability of the supernodes is 0 by definition.\n")

    # ---------- RANDOM CIRCLE-SEED CASCADE STATS ----------
    print("===== RANDOM CIRCLE-SEED CASCADE STATS: NORMAL VS STUBBORN SUPERNODES =====")

    circles = load_all_circles(CIRCLES_DIR)
    if not circles:
        print("No circles available; skipping circle-based cascade experiment.\n")
        return

    valid_circles = []
    for nodes in circles:
        nodes_in_G = [v for v in nodes if G.has_node(v)]
        if not nodes_in_G:
            continue
        # exclude supernodes from seeds; if all are supernodes, skip this circle
        non_blocked = [v for v in nodes_in_G if v not in blocked]
        if not non_blocked:
            continue
        valid_circles.append(non_blocked)

    if not valid_circles:
        print("No valid circles with non-blocked nodes; skipping circle-based experiment.\n")
        return

    total_circle_normal = 0
    total_circle_blocked = 0
    seeds_used = []

    for _ in range(num_random_runs):
        seeds = random.choice(valid_circles)
        seeds_used.append(seeds)

        size_normal = simulate_threshold_cascade_multi(G, seeds, q=q)
        size_blocked = simulate_threshold_cascade_multi_blocked(G, seeds, blocked, q=q)

        total_circle_normal += size_normal
        total_circle_blocked += size_blocked

    avg_circle_normal = total_circle_normal / num_random_runs
    avg_circle_blocked = total_circle_blocked / num_random_runs

    print(f"Using {num_random_runs} random circle seeds")
    print(f"Average cascade size (NORMAL, circle-seed):           {avg_circle_normal:.2f}")
    print(f"Average cascade size (STUBBORN super, circle-seed):   {avg_circle_blocked:.2f}")
    print(f"Difference (normal - super-stubborn):                {avg_circle_normal - avg_circle_blocked:.2f}\n")

    # ---------- CONTROL: RANDOM STUBBORN SET OF SAME SIZE ----------
    print("===== CONTROL: RANDOM STUBBORN SET (SAME SIZE AS SUPERNODES) =====")

    k = len(supernodes)
    if k == 0:
        print("Supernode set is empty; cannot form random stubborn sets of same size.\n")
        return

    total_circle_blocked_random = 0

    for seeds in seeds_used:
        random_blocked = random.sample(all_nodes, k)
        size_blocked_random = simulate_threshold_cascade_multi_blocked(G, seeds, random_blocked, q=q)
        total_circle_blocked_random += size_blocked_random

    avg_circle_blocked_random = total_circle_blocked_random / num_random_runs

    print(f"Average cascade size (RANDOM stubborn, circle-seed): {avg_circle_blocked_random:.2f}")
    print(f"Reduction with supernodes stubborn:  {avg_circle_normal - avg_circle_blocked:.2f}")
    print(f"Reduction with random stubborn set:  {avg_circle_normal - avg_circle_blocked_random:.2f}")
    print(f"Extra reduction from targeting supernodes: "
          f"{(avg_circle_blocked_random - avg_circle_blocked):.2f}\n")

    # ---------- THRESHOLD PROTECTION: HIGHER q FOR SUPERNODES ----------
    print("===== THRESHOLD PROTECTION: HIGHER q FOR SUPERNODES =====")
    print(f"[INFO] Non-supernodes use q = {q}, supernodes use q_super = {q_super}\n")

    total_circle_protect_q = 0

    for seeds in seeds_used:
        size_protect_q = simulate_threshold_cascade_multi_super_protect_q(
            G,
            seeds,
            supernodes,
            q_normal=q,
            q_super=q_super,
        )
        total_circle_protect_q += size_protect_q

    avg_circle_protect_q = total_circle_protect_q / num_random_runs

    print(f"Average cascade size (q-super-protect, circle-seed): {avg_circle_protect_q:.2f}")
    print(f"Reduction vs NORMAL:                           {avg_circle_normal - avg_circle_protect_q:.2f}")
    print(f"Reduction vs STUBBORN supernodes:              {avg_circle_blocked - avg_circle_protect_q:.2f}")
    print(f"Reduction vs RANDOM stubborn set:              {avg_circle_blocked_random - avg_circle_protect_q:.2f}\n")

    # ---------- BRIDGE-SHADOW: ASYMMETRIC INCOMING BLOCKING (FRACTION SWEEP) ----------
    print("===== BRIDGE-SHADOW: ASYMMETRIC INCOMING BLOCKING TO SUPERNODES (FRACTION SWEEP) =====")
    print("  [INFO] Precomputing approximate edge betweenness centrality for bridge-shadow sweep...")
    eb = nx.edge_betweenness_centrality(G, k=100, normalized=True, seed=42)

    # Build neighbor order per supernode (highest betweenness edges first)
    shadow_block_order = {}
    for s in supernodes:
        neigh = [u for u in G.neighbors(s)]
        edge_scores = []
        for u in neigh:
            e = (s, u) if (s, u) in eb else (u, s)
            score = eb.get(e, 0.0)
            edge_scores.append((u, score))
        edge_scores.sort(key=lambda x: x[1], reverse=True)
        shadow_block_order[s] = [u for (u, _) in edge_scores]

    print("  [INFO] Bridge-shadow fractions will be tested from 0.0 to 1.0 in steps of 0.1\n")

    best_frac = None
    best_avg = None

    for frac_int in range(0, 11):  # 0, 0.1, ..., 1.0
        frac = frac_int / 10.0
        total_shadow = 0

        for seeds in seeds_used:
            size_shadow = simulate_threshold_cascade_multi_bridge_shadow_incoming(
                G,
                seeds,
                supernodes,
                shadow_block_order,
                drop_fraction=frac,
                q=q,
            )
            total_shadow += size_shadow

        avg_shadow = total_shadow / num_random_runs
        reduction = avg_circle_normal - avg_shadow
        print(f"  fraction={frac:.1f} → avg cascade size = {avg_shadow:.2f} "
              f"(reduction vs NORMAL = {reduction:.2f})")

        if best_avg is None or avg_shadow < best_avg:
            best_avg = avg_shadow
            best_frac = frac

    print("\n[RESULT] Optimal bridge-shadow incoming-block fraction (over tested values):")
    print(f"  fraction = {best_frac:.1f}")
    print(f"  avg cascade size at this fraction = {best_avg:.2f}")
    print(f"  reduction vs NORMAL = {avg_circle_normal - best_avg:.2f}\n")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Identify cascade supernodes (k-sigma rule) and run supernode-based cascade experiments, including q-protection and bridge-shadow tests."
    )
    parser.add_argument(
        "--q",
        type=float,
        default=DEFAULT_Q,
        help=f"Adoption threshold q (default={DEFAULT_Q})",
    )
    parser.add_argument(
        "--q_super",
        type=float,
        default=DEFAULT_Q_SUPER,
        help=f"Adoption threshold for supernodes q_super (default={DEFAULT_Q_SUPER})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RANDOM_RUNS,
        help=f"Number of random seeds for baseline and experiments (default={DEFAULT_RANDOM_RUNS})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=(
            "Early-stop patience: consecutive non-super candidates before stopping "
            f"(default={DEFAULT_PATIENCE}; use -1 for full scan / no early stop)"
        ),
    )
    parser.add_argument(
        "--k",
        type=float,
        default=DEFAULT_K_SIGMA,
        help=f"k in the k-sigma supernode rule (default={DEFAULT_K_SIGMA})",
    )
    parser.add_argument(
        "--shadow_frac",
        type=float,
        default=DEFAULT_SHADOW_FRAC,
        help=(
            "Base fraction of edges around each supernode to hide in shadow-blocking tests "
            f"(default={DEFAULT_SHADOW_FRAC}); current implementation sweeps 0..1 regardless."
        ),
    )
    args = parser.parse_args()

    q = args.q
    q_super = args.q_super
    num_runs = args.runs
    patience = args.patience
    k_sigma = args.k
    shadow_frac = args.shadow_frac

    # Load precomputed centrality stats from basic_stats.py
    try:
        with open(STATS_JSON, "r") as f:
            stats = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] {STATS_JSON} not found. Run basic_stats.py first.")
        return

    pagerank_dict = {int(n): float(s) for n, s in stats["pagerank"].items()}
    betweenness_dict = {int(n): float(s) for n, s in stats["betweenness"].items()}

    print("[INFO] Loaded centrality stats from basic_stats.py")
    print(
        f"[INFO] Using q = {q}, q_super = {q_super}, runs = {num_runs}, "
        f"patience = {patience}, k = {k_sigma}, shadow_frac (base) = {shadow_frac}\n"
    )

    G = load_graph(DATA_FILE)

    supernodes, (mu, sigma, threshold) = find_cascade_supernodes(
        G,
        pagerank_dict,
        betweenness_dict,
        q=q,
        num_random_runs=num_runs,
        patience=patience,
        k_sigma=k_sigma,
    )

    supernode_cascade_experiments(
        G,
        supernodes,
        q=q,
        q_super=q_super,
        num_random_runs=num_runs,
        shadow_frac=shadow_frac,
    )

    print("[DONE cascade_super_sim.py]")


if __name__ == "__main__":
    main()
