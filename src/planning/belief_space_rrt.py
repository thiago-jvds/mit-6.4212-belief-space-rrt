"""
Belief-Space RRT (RRBT) Planning Algorithm - Anytime Version

This implements RRBT with a discrete 3-bin Bayes filter and combined cost function:
    cost = path_length + λ × misclassification_risk

Where misclassification_risk = 1 - max(belief)

The planner uses RRT*-style anytime behavior:
1. Builds a tree that balances path efficiency and information gain
2. Tracks the BEST valid solution (lowest cost with misclassification_risk < threshold)
3. Continues exploring for ALL iterations to find better solutions
4. Returns the best path found after max_iterations
"""

import numpy as np
from src.planning.rrt_tools import RRBT_BinBelief_tools
from src.estimation.bayes_filter import calculate_misclassification_risk
import random

from typing import Callable, Optional


def rrbt_planning(
    problem,
    max_iterations: int = 2000,
    bias_prob_sample_q_goal: float = 0.05,  # Reduced bias since we don't know goal location
    bias_prob_sample_q_bin_light: float = 0.4,  # Increased bias to force info gathering
    q_light_hint: np.ndarray = np.array(
        [0.663, 0.746, 0.514, -1.406, 0.996, -1.306, -1.028]
    ),
    visualize_callback: Optional[Callable] = None,
    visualize_interval: int = 1,
    verbose: bool = True,
) -> tuple[tuple[list[tuple], np.ndarray] | None, int]:
    """
    RRBT Planning with anytime RRT*-style behavior.
    
    Unlike standard RRT which stops at the first solution, this implementation
    continues exploring for all max_iterations to find the BEST solution
    (lowest cost among paths that achieve the uncertainty threshold).
    
    Args:
        problem: IiwaProblemBelief instance
        max_iterations: Planning iterations (runs ALL iterations)
        prob_sample_q_goal: Probability of sampling toward goal
        prob_sample_q_light: Probability of sampling toward light region
        max_uncertainty: Threshold for misclassification_risk to consider a solution valid
        lambda_weight: Trade-off between path length and uncertainty
            - Small (0.1-1): Prioritize shorter paths
            - Medium (1-10): Balanced
            - Large (100+): Prioritize uncertainty reduction
        q_light_hint: Configuration hint for light region sampling
        visualize_callback: Optional callback for visualization
        visualize_interval: Iterations between visualization updates
        
    Returns:
        ((path_to_info, predicted_goal_config), iterations) or (None, iterations)
    """
    # 1. Initialize Tools with lambda_weight
    tools = RRBT_BinBelief_tools(
        problem,
    )

    # Track the best valid solution found
    best_node = None
    best_cost = float('inf')
    first_solution_iter = None

    for k in range(max_iterations):
        print(f"   > RRBT Iter {k + 1}/{max_iterations} ", end="\r")
        # 2. Sample
        eps = random.random()

        if eps < bias_prob_sample_q_goal:
            # Sample the Prior Mean (Best guess of goal)
            q_rand = problem.goal
        elif eps < (bias_prob_sample_q_goal + bias_prob_sample_q_bin_light):
            # Sample the Light Region (To gain info)
            q_rand = tuple(q_light_hint + np.random.uniform(-0.1, 0.1, size=7))
        else:
            q_rand = tools.sample_node()

        # 3. Extend
        last_node = tools.extend_towards(q_rand)

        # 4. Check if this node is a valid solution AND better than current best
        if problem.node_reaches_goal(last_node, tol=None):
            if last_node.cost < best_cost:
                # Found a better solution!
                best_node = last_node
                best_cost = last_node.cost
                misclass_risk = calculate_misclassification_risk(last_node.belief)
                
                if first_solution_iter is None:
                    first_solution_iter = k + 1
                    print(
                        f"\n      First solution at iter {k + 1}: "
                        f"MisclassRisk={misclass_risk:.4f}, "
                        f"PathLen={last_node.path_length:.2f}, "
                        f"Cost={best_cost:.4f}"
                    )
                else:
                    print(
                        f"\n      Better solution at iter {k + 1}: "
                        f"MisclassRisk={misclass_risk:.4f}, "
                        f"PathLen={last_node.path_length:.2f}, "
                        f"Cost={best_cost:.4f}"
                    )
            elif first_solution_iter is not None:
                if verbose:
                    misclass_risk = calculate_misclassification_risk(last_node.belief)
                    print(
                        f"\n            Found another solution at iter {k + 1}: "
                        f"MisclassRisk={misclass_risk:.4f}, "
                        f"PathLen={last_node.path_length:.2f}, "
                        f"Cost={last_node.cost:.4f}"
                    )

        # Visualize progress
        if visualize_callback and (k + 1) % visualize_interval == 0:
            visualize_callback(tools.rrbt_bin_belief_tree, k + 1)

        # Progress report
        if k % 100 == 0:
            misclass_risk = calculate_misclassification_risk(last_node.belief)
            best_info = f", Best={best_cost:.3f}" if best_node else ""
            print(
                f"   > RRBT Iter {k + 1}/{max_iterations} "
                f"(MisclassRisk: {misclass_risk:.3f}, PathLen: {last_node.path_length:.2f}{best_info})",
                end="\r",
            )

    # 5. Final pass: Scan ALL nodes to find the best valid solution
    # This catches any improvements from rewiring that we might have missed
    print(f"\n   Scanning {len(tools.rrbt_bin_belief_tree.nodes)} nodes for best solution...")
    for node in tools.rrbt_bin_belief_tree.nodes:
        misclass_risk = calculate_misclassification_risk(node.belief)
        if misclass_risk <= problem.max_bin_uncertainty and node.cost < best_cost:
            best_node = node
            best_cost = node.cost

    # 6. Return the best solution found
    if best_node is not None:
        path_to_info = tools.backup_path_from_node(best_node)
        pred_q_goal = tools.sample_object_position_from_belief(best_node)
        
        misclass_risk = calculate_misclassification_risk(best_node.belief)
        print(
            f"RRBT Complete after {max_iterations} iterations."
        )
        print(
            f"   Best solution: MisclassRisk={misclass_risk:.4f}, "
            f"PathLength={best_node.path_length:.2f}, "
            f"Cost={best_cost:.4f}"
        )
        print(f"   Final belief: {best_node.belief}")
        if first_solution_iter:
            print(
                f"   First solution found at iteration {first_solution_iter}"
            )

        if visualize_callback:
            visualize_callback(tools.rrbt_bin_belief_tree, max_iterations)

        return (path_to_info, pred_q_goal), max_iterations
    
    # No valid solution found
    print(f"RRBT: No solution found after {max_iterations} iterations.")
    print(f"   No path achieved misclassification_risk < {problem.max_bin_uncertainty}")
    return None, max_iterations
