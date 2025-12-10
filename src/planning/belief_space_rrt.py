"""
Belief-Space RRT (RRBT) Planning Algorithm - Anytime Version

This implements RRBT with anytime RRT*-style behavior:
1. Builds a tree that balances path efficiency and information gain
2. Tracks the BEST valid solution (lowest cost with uncertainty < threshold)
3. Continues exploring for ALL iterations to find better solutions
4. Returns the best path found after max_iterations

The algorithm is generic and works with any belief type. The problem class
defines:
- Belief representation and initialization
- Belief propagation
- Cost computation  
- Goal checking (termination condition)
- Goal sampling from belief
"""

import numpy as np
from src.planning.rrt_tools import RRBT_tools
from src.planning.rrbt_tree import RRBTProblem

from typing import Callable, Optional


def rrbt_planning(
    problem: RRBTProblem,
    max_iterations: int = 2000,
    bias_prob_sample_q_goal: float = 0.05,
    bias_prob_sample_q_bin_light: float = 0.4,
    q_light_hint: np.ndarray = np.array(
        [0.663, 0.746, 0.514, -1.406, 0.996, -1.306, -1.028]
    ),
    visualize_callback: Optional[Callable] = None,
    visualize_interval: int = 1,
    verbose: bool = True,
    rng: np.random.Generator = None,
) -> tuple[tuple[list[tuple], np.ndarray] | None, int]:
    """
    RRBT Planning with anytime RRT*-style behavior.
    
    Unlike standard RRT which stops at the first solution, this implementation
    continues exploring for all max_iterations to find the BEST solution
    (lowest cost among paths that achieve the uncertainty threshold).
    
    Args:
        problem: Problem instance implementing the RRBTProblem protocol
        max_iterations: Planning iterations (runs ALL iterations)
        bias_prob_sample_q_goal: Probability of sampling toward goal
        bias_prob_sample_q_bin_light: Probability of sampling toward light region
        q_light_hint: Configuration hint for light region sampling
        visualize_callback: Optional callback for visualization
        visualize_interval: Iterations between visualization updates
        verbose: Print detailed progress information
        rng: NumPy random generator for reproducibility (optional)
        
    Returns:
        ((path_to_info, predicted_goal_config), iterations) or (None, iterations)
    """
    # Use provided rng or create default
    if rng is None:
        rng = np.random.default_rng()
    
    # Initialize tools (generic - works with any belief type)
    tools = RRBT_tools(problem)

    # Track the best valid solution found
    best_node = None
    best_cost = float('inf')
    first_solution_iter = None

    for k in range(max_iterations):
        print(f"   > RRBT Iter {k + 1}/{max_iterations} ", end="\r")
        
        # Sample configuration with biases
        eps = rng.random()

        if eps < bias_prob_sample_q_goal:
            # Sample the prior mean (best guess of goal)
            q_rand = problem.goal
        elif eps < (bias_prob_sample_q_goal + bias_prob_sample_q_bin_light):
            # Sample the light region (to gain information)
            q_rand = tuple(q_light_hint + rng.uniform(-0.1, 0.1, size=7))
        else:
            q_rand = tools.sample_node()

        # Extend tree toward sample
        last_node = tools.extend_towards(q_rand)

        # Check if this node is a valid solution AND better than current best
        if tools.node_reaches_goal(last_node, tol=None):
            if last_node.cost < best_cost:
                # Found a better solution!
                best_node = last_node
                best_cost = last_node.cost
                
                if first_solution_iter is None:
                    first_solution_iter = k + 1
                    print(
                        f"\n      First solution at iter {k + 1}: "
                        f"PathLen={last_node.path_length:.2f}, "
                        f"Cost={best_cost:.4f}"
                    )
                else:
                    print(
                        f"\n      Better solution at iter {k + 1}: "
                        f"PathLen={last_node.path_length:.2f}, "
                        f"Cost={best_cost:.4f}"
                    )
            elif first_solution_iter is not None:
                if verbose:
                    print(
                        f"\n            Found another solution at iter {k + 1}: "
                        f"PathLen={last_node.path_length:.2f}, "
                        f"Cost={last_node.cost:.4f}"
                    )

        # Visualize progress
        if visualize_callback and (k + 1) % visualize_interval == 0:
            visualize_callback(tools.tree, k + 1)

        # Progress report
        if k % 100 == 0:
            best_info = f", Best={best_cost:.3f}" if best_node else ""
            print(
                f"   > RRBT Iter {k + 1}/{max_iterations} "
                f"(PathLen: {last_node.path_length:.2f}{best_info})",
                end="\r",
            )

    # Final pass: Scan ALL nodes to find the best valid solution
    # This catches any improvements from rewiring that we might have missed
    print(f"\n   Scanning {len(tools.tree.nodes)} nodes for best solution...")
    for node in tools.tree.nodes:
        if tools.node_reaches_goal(node) and node.cost < best_cost:
            best_node = node
            best_cost = node.cost

    # Return the best solution found
    if best_node is not None:
        path_to_info = tools.backup_path_from_node(best_node)
        pred_q_goal = tools.sample_goal_from_belief(best_node)
        
        print(f"RRBT Complete after {max_iterations} iterations.")
        print(
            f"   Best solution: "
            f"PathLength={best_node.path_length:.2f}, "
            f"Cost={best_cost:.4f}"
        )
        if first_solution_iter:
            print(f"   First solution found at iteration {first_solution_iter}")

        if visualize_callback:
            visualize_callback(tools.tree, max_iterations)

        return (path_to_info, pred_q_goal), max_iterations
    
    # No valid solution found
    print(f"RRBT: No solution found after {max_iterations} iterations.")
    return None, max_iterations
