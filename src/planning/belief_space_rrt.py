import numpy as np
from src.planning.rrt_tools import RRBT_tools
import random


def rrbt_planning(
    problem,
    max_iterations: int = 2000,
    prob_sample_q_goal: float = 0.15,
    prob_sample_q_light: float = 0.2,
    max_uncertainty: float = 0.1,
    q_light_hint: np.ndarray = np.array([0.663, 0.746, 0.514, -1.406, 0.996, -1.306, -1.028]),
) -> tuple[list[tuple] | None, int]:
    """
    Implements the Rapidly-exploring Random Belief Tree (RRBT) planner.

    Args:
        problem (IiwaProblemBelief): An instance of the IiwaProblemBelief class that
            defines the planning problem with belief space considerations.
        max_iterations (int, optional): Maximum number of iterations to run.
            Defaults to 2000.

    Returns:
        tuple[list[tuple] | None, int]:
            - path (list[tuple]): A list of configurations [q_start, ..., q_goal]
              representing the path that satisfies both collision and belief constraints.
              Returns None if no solution is found.
            - iterations (int): The number of iterations performed.
    """
    tools = RRBT_tools(problem, max_uncertainty)

    for k in range(max_iterations):
        print(f"   > RRBT Iteration {k + 1}/{max_iterations}", end="\r")

        # 1. Sample
        eps = random.random()
        if eps < prob_sample_q_goal:
            q_rand = problem.goal
        elif eps < (prob_sample_q_goal + prob_sample_q_light):
            q_rand = tuple(q_light_hint + np.random.uniform(-0.1, 0.1, size=7))
        else:
            q_rand = tools.sample_node()

        # 2. Extend (Handles Nearest + Interpolate + Rewire internally)
        last_node = tools.extend_towards(q_rand)

        # 3. Goal Check
        if tools.node_reaches_goal(last_node, tol=0.15):
            return tools.backup_path(last_node), k

    tools.print_stats()
    return None, max_iterations
