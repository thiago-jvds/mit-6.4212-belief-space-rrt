import numpy as np
from src.planning.rrt_tools import RRBT_tools
import random


from typing import Callable, Optional


def rrbt_planning(
    problem,
    max_iterations: int = 2000,
    prob_sample_q_goal: float = 0.05,  # Reduced bias since we don't know goal location
    prob_sample_q_light: float = 0.4,  # Increased bias to force info gathering
    max_uncertainty: float = 0.01,
    q_light_hint: np.ndarray = np.array(
        [0.663, 0.746, 0.514, -1.406, 0.996, -1.306, -1.028]
    ),
    visualize_callback: Optional[Callable] = None,
    visualize_interval: int = 1,
) -> tuple[tuple[list[tuple], np.ndarray] | None, int]:
    """
    Returns:
        ((path_to_info, predicted_goal_config), iterations)
    """
    # 1. Initialize Tools (Initial Belief is High)
    # Ensure config sets initial_target_uncertainty high (e.g. 1.0)
    tools = RRBT_tools(
        problem, max_uncertainty=max_uncertainty, initial_uncertainty=1.0
    )

    for k in range(max_iterations):
        # 2. Sample
        eps = random.random()

        if eps < prob_sample_q_goal:
            # Sample the Prior Mean (Best guess of goal)
            q_rand = problem.goal
        elif eps < (prob_sample_q_goal + prob_sample_q_light):
            # Sample the Light Region (To gain info)
            q_rand = tuple(q_light_hint + np.random.uniform(-0.1, 0.1, size=7))
        else:
            q_rand = tools.sample_node()

        # 3. Extend
        last_node = tools.extend_towards(q_rand)

        # 4. Check Termination (Is Uncertainty Low?)
        # Note: We pass tol=None because geometric tolerance is ignored
        if tools.node_reaches_goal(last_node, tol=None):
            # A. Extract the path to this high-information state
            path_to_info = tools.backup_path_from_node(last_node)

            # B. Sample the specific target location from the belief
            pred_q_goal = tools.sample_final_goal(last_node)

            print(
                f"\n✅ RRBT: Uncertainty reduced to {last_node.cost:.4f}. Committing to target."
            )

            if visualize_callback:
                visualize_callback(tools.rrbt_tree, k + 1)

            return (path_to_info, pred_q_goal), k

        # Visualize progress
        if visualize_callback and (k + 1) % visualize_interval == 0:
            visualize_callback(tools.rrbt_tree, k + 1)

        if k % 10 == 0:
            print(
                f"   > RRBT Iteration {k + 1}/{max_iterations} (Cost: {last_node.cost:.3f})",
                end="\r",
            )

    print("\n❌ RRBT: Failed to reduce uncertainty below threshold.")
    tools.print_stats()
    return None, max_iterations
