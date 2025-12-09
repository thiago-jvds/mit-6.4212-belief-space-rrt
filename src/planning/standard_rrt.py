import numpy as np
from src.planning.rrt_tools import RRT_tools
from src.simulation.simulation_tools import IiwaProblem


def rrt_planning(
    problem: IiwaProblem,
    max_iterations: int = 1000,
    prob_sample_q_goal: float = 0.05,
    rng: np.random.Generator = None,
) -> tuple[list[tuple] | None, int]:
    """
    Input:
        problem (IiwaProblem): instance of a utility class
        max_iterations: the maximum number of samples to be collected
        prob_sample_q_goal: the probability of sampling q_goal
        rng: NumPy random generator for reproducibility (optional)

    Output:
    (path, iterations) (tuple):
        path (list): [q_start, ..., q_goal]. Each element q is a configuration (not an RRT node).
        iterations (int): The number of iterations executed to obtain the solution.
                          If no solution is found, return (None, max_iterations).

    """
    # Use provided rng or create default
    if rng is None:
        rng = np.random.default_rng()
    
    rrt_tools = RRT_tools(problem)
    q_goal = problem.goal

    for k in range(max_iterations):
        print(f"  > RRT Iteration {k+1}/{max_iterations}", end="\r")
        q_sample = rrt_tools.sample_node_in_configuration_space()

        eps = rng.random()
        if eps < prob_sample_q_goal:
            q_sample = q_goal

        n_near = rrt_tools.find_nearest_node_in_RRT_graph(q_sample)

        qs = rrt_tools.calc_intermediate_qs_wo_collision(n_near.value, q_sample)

        last_node = n_near
        for i in range(len(qs)):
            last_node = rrt_tools.grow_rrt_tree(last_node, qs[i])

        if rrt_tools.node_reaches_goal(last_node, tol=0.15):
            path = rrt_tools.backup_path_from_node(last_node)
            return path, k

    return None, max_iterations

