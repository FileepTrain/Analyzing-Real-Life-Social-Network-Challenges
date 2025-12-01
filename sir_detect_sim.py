"""
Adaptive SIR misinformation spread simulation.

States:
    S = Susceptible
    I = Infected (spreading misinformation)
    R = Recovered (corrected / flagged, no longer spreads)

Mechanism:
    - Start with base infection probability beta and recovery probability gamma.
    - Once the number of infected nodes reaches a threshold fraction of the
      total population, the platform "intervenes":
        * infection probability drops to beta_after
        * recovery probability increases to gamma_after

This models early detection/flagging and corrective labels on misinformed posts.

Example usage:

    py sir_detect_sim.py --input facebook_combined.txt --plot

    py sir_detect_sim.py --input facebook_combined.txt \
        --beta 0.05 --gamma 0.05 \
        --beta-after 0.01 --gamma-after 0.25 \
        --threshold-frac 0.10 --initial-infected 5 --max-steps 50 --plot

    py sir_detect_sim.py --input facebook_combined.txt \
        --seed-node 107 --threshold-frac 0.05 --plot
"""

import argparse
import random
from typing import List, Set, Hashable, Tuple, Optional

import networkx as nx
import matplotlib.pyplot as plt


# ----------------- CLI ARGUMENTS ----------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive SIR misinformation simulation with intervention threshold."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to SNAP-style undirected edge list file (e.g., facebook_combined.txt)."
    )
    parser.add_argument(
        "--beta", type=float, default=0.05,
        help="Initial infection probability per contact per step (default: 0.05)."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.05,
        help="Initial recovery probability per infected node per step (default: 0.05)."
    )
    parser.add_argument(
        "--beta-after", type=float, default=0.01,
        help="Infection probability AFTER intervention (default: 0.01)."
    )
    parser.add_argument(
        "--gamma-after", type=float, default=0.45,
        help="Recovery probability AFTER intervention (default: 0.45)."
    )
    parser.add_argument(
        "--threshold-frac", type=float, default=0.10,
        help="Fraction of total nodes infected at which intervention triggers "
             "(e.g., 0.10 = 10%% of nodes)."
    )
    parser.add_argument(
        "--initial-infected", type=int, default=1,
        help="Number of initial infected nodes (random if seed-node not set)."
    )
    parser.add_argument(
        "--seed-node", type=str, default=None,
        help="Optional specific node ID to infect initially (overrides --initial-infected)."
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Maximum number of simulation steps (default: 50)."
    )
    parser.add_argument(
        "--random-seed", type=int, default=None,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="If set, plot infected/recovered curves over time."
    )
    return parser.parse_args()


# ----------------- GRAPH LOADING ----------------- #

def load_undirected_snap_graph(path: str) -> nx.Graph:
    """
    Load a SNAP-style undirected edge list.

    Assumptions:
        - Lines starting with '#' are comments.
        - Each non-comment line has at least: u v
    """
    G = nx.Graph()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            G.add_edge(u, v)

    return G


# ----------------- CHOOSING INITIAL INFECTED ----------------- #

def choose_initial_infected(
    G: nx.Graph,
    initial_infected: int,
    seed_node: Optional[str]
) -> Set[Hashable]:
    """
    Choose initial infected nodes.

    - If seed_node is provided and exists in the graph, start from that node only.
    - Otherwise choose `initial_infected` distinct nodes uniformly at random.
    """
    nodes = list(G.nodes())

    if not nodes:
        return set()

    if seed_node is not None:
        if seed_node in G:
            return {seed_node}
        else:
            print(f"Warning: seed-node {seed_node!r} not found in graph. "
                  f"Falling back to random selection.")
            seed_node = None

    k = max(1, min(initial_infected, len(nodes)))
    return set(random.sample(nodes, k))


# ----------------- ADAPTIVE SIR SIMULATION ----------------- #

def sir_adaptive_simulation(
    G: nx.Graph,
    beta: float,
    gamma: float,
    beta_after: float,
    gamma_after: float,
    threshold_count: int,
    initial_infected: Set[Hashable],
    max_steps: int
) -> Tuple[List[int], List[int], int, Set[Hashable], Set[Hashable]]:
    """
    Run an SIR simulation where infection/recovery parameters change after
    a threshold number of infected nodes is reached.

    :param G: undirected graph
    :param beta: initial infection probability
    :param gamma: initial recovery probability
    :param beta_after: infection probability after intervention
    :param gamma_after: recovery probability after intervention
    :param threshold_count: infected count that triggers intervention
    :param initial_infected: set of initially infected nodes
    :param max_steps: maximum time steps
    :return: (infected_counts, recovered_counts, intervention_step,
              final_infected, final_recovered)
             intervention_step = -1 if threshold never reached.
    """
    infected: Set[Hashable] = set(initial_infected)
    recovered: Set[Hashable] = set()

    infected_counts: List[int] = [len(infected)]
    recovered_counts: List[int] = [len(recovered)]

    if len(infected) == 0:
        return infected_counts, recovered_counts, -1, infected, recovered

    beta_current = beta
    gamma_current = gamma
    intervention_step = -1
    intervention_active = False

    for step in range(1, max_steps + 1):
        # Check threshold *before* this step's transitions
        if (not intervention_active) and (len(infected) >= threshold_count):
            intervention_active = True
            intervention_step = step
            beta_current = beta_after
            gamma_current = gamma_after
            print(
                f"\n*** INTERVENTION TRIGGERED at step {step}: "
                f"infected={len(infected)} (>= {threshold_count}). "
                f"beta -> {beta_current}, gamma -> {gamma_current} ***\n"
            )

        new_infected: Set[Hashable] = set()
        new_recovered: Set[Hashable] = set()

        # Infection attempts with current beta
        for u in infected:
            for v in G.neighbors(u):
                if v in infected or v in recovered:
                    continue
                if random.random() < beta_current:
                    new_infected.add(v)

        # Recoveries with current gamma
        for u in infected:
            if random.random() < gamma_current:
                new_recovered.add(u)

        infected |= new_infected
        infected -= new_recovered
        recovered |= new_recovered

        infected_counts.append(len(infected))
        recovered_counts.append(len(recovered))

        print(
            f"Step {step}: newly infected = {len(new_infected)}, "
            f"newly recovered = {len(new_recovered)}, "
            f"total infected = {len(infected)}, "
            f"total recovered = {len(recovered)}"
        )

        # Stop if epidemic has died out
        if len(infected) == 0:
            print(f"No infected nodes remain at step {step}. Stopping early.")
            break

    return infected_counts, recovered_counts, intervention_step, infected, recovered


# ----------------- PLOTTING ----------------- #

def plot_adaptive_sir(
    infected_counts: List[int],
    recovered_counts: List[int],
    total_nodes: int,
    intervention_step: int
):
    steps = list(range(len(infected_counts)))
    infected_frac = [i / total_nodes for i in infected_counts]
    recovered_frac = [r / total_nodes for r in recovered_counts]

    # Counts
    plt.figure()
    plt.plot(steps, infected_counts, marker="o", label="Infected")
    plt.plot(steps, recovered_counts, marker="s", label="Recovered")
    if intervention_step >= 0 and intervention_step < len(steps):
        plt.axvline(intervention_step, linestyle="--", label="Intervention")
    plt.xlabel("Time step")
    plt.ylabel("Number of nodes")
    plt.title("Adaptive SIR: Counts Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Fractions
    plt.figure()
    plt.plot(steps, infected_frac, marker="o", label="Infected")
    plt.plot(steps, recovered_frac, marker="s", label="Recovered")
    if intervention_step >= 0 and intervention_step < len(steps):
        plt.axvline(intervention_step, linestyle="--", label="Intervention")
    plt.xlabel("Time step")
    plt.ylabel("Fraction of nodes")
    plt.title("Adaptive SIR: Fractions Over Time")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


# ----------------- MAIN ----------------- #

def main():
    args = parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)

    print(f"Loading graph from: {args.input}")
    G = load_undirected_snap_graph(args.input)
    print(f"Graph loaded. Nodes = {G.number_of_nodes():,}, "
          f"Edges = {G.number_of_edges():,}\n")

    if G.number_of_nodes() == 0:
        print("Graph is empty. Exiting.")
        return

    total_nodes = G.number_of_nodes()
    threshold_count = max(1, int(args.threshold_frac * total_nodes))
    print(f"Intervention threshold: {args.threshold_frac:.2%} of nodes "
          f"= {threshold_count} infected nodes.\n")

    initial_infected = choose_initial_infected(
        G,
        initial_infected=args.initial_infected,
        seed_node=args.seed_node,
    )

    print(f"Initial infected nodes ({len(initial_infected)}): "
          f"{sorted(initial_infected)}")
    print(
        f"Running adaptive SIR with beta={args.beta}, gamma={args.gamma}, "
        f"beta_after={args.beta_after}, gamma_after={args.gamma_after}, "
        f"max_steps={args.max_steps}...\n"
    )

    infected_counts, recovered_counts, intervention_step, final_I_set, final_R_set = (
        sir_adaptive_simulation(
            G,
            beta=args.beta,
            gamma=args.gamma,
            beta_after=args.beta_after,
            gamma_after=args.gamma_after,
            threshold_count=threshold_count,
            initial_infected=initial_infected,
            max_steps=args.max_steps,
        )
    )

    final_I = infected_counts[-1]
    final_R = recovered_counts[-1]

    print("\n=== ADAPTIVE SIR SUMMARY ===")
    print(f"Total nodes:                      {total_nodes:,}")
    print(f"Final infected (I):               {final_I:,} "
          f"({final_I / total_nodes:.4f})")
    print(f"Final recovered (R):              {final_R:,} "
          f"({final_R / total_nodes:.4f})")
    if intervention_step >= 0:
        print(f"Intervention triggered at step:   {intervention_step}")
    else:
        print("Intervention never triggered (threshold not reached).")
    print(f"Steps simulated (including t=0):  {len(infected_counts) - 1}")

    if args.plot:
        print("\nPlotting adaptive SIR curves...")
        plot_adaptive_sir(
            infected_counts,
            recovered_counts,
            total_nodes,
            intervention_step,
        )


if __name__ == "__main__":
    main()

