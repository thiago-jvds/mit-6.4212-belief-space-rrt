import numpy as np
from src.planning.rrt_tools import RRT_tools
from src.simulation.simulation_tools import IiwaProblem


def rrt_planning(
    problem: IiwaProblem,
    max_iterations: int = 1000,
    prob_sample_q_goal: float = 0.05,
    rng: np.random.Generator = None,
    verbose: bool = False,
) -> tuple[list[tuple] | None, int]:
    """
    Standard RRT (Rapidly-exploring Random Tree) path planning algorithm.
    
    This function implements the classic RRT algorithm for collision-free
    motion planning in configuration space. It grows a tree from the start
    configuration towards randomly sampled configurations until the goal
    is reached.
    
    Input:
        problem (IiwaProblem): instance of a utility class containing:
            - start: starting configuration q_start
            - goal: goal configuration q_goal
            - collision checking utilities
        max_iterations: the maximum number of samples to be collected
        prob_sample_q_goal: the probability of sampling q_goal (goal bias)
        rng: NumPy random generator for reproducibility (optional)
        verbose: if True, print detailed progress information

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
    
    if verbose:
        print(f"    [RRT] Starting RRT planning...")
        print(f"    [RRT] Max iterations: {max_iterations}, Goal bias: {prob_sample_q_goal}")
    
    total_nodes_added = 0
    goal_samples = 0

    for k in range(max_iterations):
        # Progress indicator (overwriting line for cleaner output)
        if not verbose:
            print(f"    [RRT] Iteration {k+1}/{max_iterations}, nodes: {total_nodes_added}", end="\r")
        
        q_sample = rrt_tools.sample_node_in_configuration_space()

        eps = rng.random()
        if eps < prob_sample_q_goal:
            q_sample = q_goal
            goal_samples += 1
            if verbose:
                print(f"    [RRT] Iter {k+1}: Sampling GOAL configuration (goal bias triggered)")

        n_near = rrt_tools.find_nearest_node_in_RRT_graph(q_sample)

        qs = rrt_tools.calc_intermediate_qs_wo_collision(n_near.value, q_sample)

        last_node = n_near
        nodes_added_this_iter = 0
        for i in range(len(qs)):
            last_node = rrt_tools.grow_rrt_tree(last_node, qs[i])
            nodes_added_this_iter += 1
            total_nodes_added += 1
        
        if verbose and nodes_added_this_iter > 0:
            print(f"    [RRT] Iter {k+1}: Added {nodes_added_this_iter} nodes (total: {total_nodes_added})")

        if rrt_tools.node_reaches_goal(last_node, tol=0.15):
            path = rrt_tools.backup_path_from_node(last_node)
            # Clear the progress line and print final status
            print(f"    [RRT] SUCCESS! Found path in {k+1} iterations                    ")
            print(f"    [RRT] Total nodes in tree: {total_nodes_added}")
            print(f"    [RRT] Goal samples: {goal_samples}")
            print(f"    [RRT] Path length: {len(path)} waypoints")
            return path, k

    # Clear the progress line on failure
    print(f"    [RRT] FAILED after {max_iterations} iterations                    ")
    print(f"    [RRT] Total nodes explored: {total_nodes_added}")
    return None, max_iterations

