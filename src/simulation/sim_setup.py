"""
Simulation and Visualization Utilities for Belief-Space RRT.

This module provides utilities for:
- Visualizing planned paths with belief uncertainty indicators
- Visualizing the RRBT tree structure
- Debug utilities for belief propagation

Note: Functions marked with (LEGACY) use Kalman filter concepts and may not
work with the discrete Bayes filter. Use the main execution diagram in main.py
for proper Bayes filter execution.
"""

import numpy as np
from src.simulation.simulation_tools import IiwaProblemBinBelief
from src.estimation.bayes_filter import calculate_misclassification_risk, expected_posterior_all_bins
from typing import List
from pydrake.all import (
    Sphere,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Meshcat,
)


def visualize_belief_path(problem, path, meshcat, belief_nodes=None):
    """
    Visualizes the planned path with uncertainty indicators using discrete Bayes filter.

    Args:
        problem: IiwaProblemBelief instance
        path: List of configurations [q_start, ..., q_goal]
        meshcat: Meshcat instance
        belief_nodes: Optional list of BeliefNodes (if available)
    """
    plant = problem.collision_checker.plant
    context = problem.collision_checker.context_plant
    iiwa = plant.GetModelInstanceByName("iiwa")
    wsg_body = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))

    # Initialize with uniform prior
    belief = np.ones(problem.n_bins) / problem.n_bins

    for i, q in enumerate(path):
        # Get gripper position via forward kinematics
        plant.SetPositions(context, iiwa, np.array(q))
        X_Gripper = plant.EvalBodyPoseInWorld(context, wsg_body)
        gripper_pos = X_Gripper.translation()

        # Propagate belief using expected posterior
        tpr, fpr = problem.get_sensor_model(q)
        in_light = problem.is_in_light(q)
        
        if in_light:
            belief = expected_posterior_all_bins(
                belief, tpr, fpr,
                assumed_bin=problem.true_bin
            )
        
        misclass_risk = calculate_misclassification_risk(belief)

        # Color: Green if in light, Red if in dark
        color = Rgba(0, 1, 0, 0.5) if in_light else Rgba(1, 0, 0, 0.5)

        # Sphere radius proportional to misclassification risk (scaled for visibility)
        radius = max(0.01, min(0.1, misclass_risk * 0.15))  # Clamp between 1cm and 10cm

        # Draw sphere at gripper position
        sphere_name = f"path_uncertainty_{i}"
        meshcat.SetObject(sphere_name, Sphere(radius), color)
        meshcat.SetTransform(sphere_name, RigidTransform(RotationMatrix(), gripper_pos))

        # Draw path line segment
        if i > 0:
            prev_q = path[i - 1]
            plant.SetPositions(context, iiwa, np.array(prev_q))
            prev_pos = plant.EvalBodyPoseInWorld(context, wsg_body).translation()

            # Draw line segment
            line_name = f"path_line_{i}"
            meshcat.SetLine(
                line_name,
                np.column_stack([prev_pos, gripper_pos]),
                rgba=Rgba(1, 1, 0, 1),
            )  # Yellow path

    # Print path statistics
    print(f"\nPath Statistics:")
    print(f"   Total waypoints: {len(path)}")
    print(f"   Final misclassification risk: {misclass_risk:.6f}")
    print(f"   Final belief: {belief}")


def visualize_belief_tree(rrbt_tree, problem, meshcat, iteration):
    """
    Visualizes the RRBT tree in Meshcat for debugging.
    Shows only key structural nodes: root, leaves, and junctions.

    Args:
        rrbt_tree: RRBT_Tree instance containing the belief tree
        problem: IiwaProblemBelief instance (for FK and light region check)
        meshcat: Meshcat instance
        iteration: Current iteration number (for display)
    """
    # Clear previous tree visualization
    meshcat.Delete("rrbt_tree")

    plant = problem.collision_checker.plant
    context = problem.collision_checker.context_plant
    iiwa = plant.GetModelInstanceByName("iiwa")
    wsg_body = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))

    def get_gripper_pos(q):
        """Helper to compute gripper position via forward kinematics."""
        plant.SetPositions(context, iiwa, np.array(q))
        X_Gripper = plant.EvalBodyPoseInWorld(context, wsg_body)
        return X_Gripper.translation()

    # DEBUG: Sanity check parent pointers
    if iteration == 1:
        print("\n--- DEBUG TREE STRUCTURE ---")
        root = rrbt_tree.nodes[0]
        print(f"Root ID: {id(root)}")

        # check how many nodes point to root
        children_of_root = [n for n in rrbt_tree.nodes if n.parent == root]
        print(f"Nodes pointing to root: {len(children_of_root)}")

        # Check a random child of root
        if children_of_root:
            child = children_of_root[0]
            print(f"Child ID: {id(child)}, Child.parent ID: {id(child.parent)}")
            print(f"Parent match: {child.parent is root}")

            # Check grandchildren
            grandchild = [n for n in rrbt_tree.nodes if n.parent == child]
            print(f"Grandchildren of first child: {len(grandchild)}")
            if len(grandchild) > 0:
                print(
                    f"First child IS Intermediate (or Junction). is_key_node should be False."
                )
            else:
                print(f"First child IS Leaf. is_key_node should be True.")

        # Check for orphaned parents
        orphans = 0
        for n in rrbt_tree.nodes:
            if n.parent and n.parent not in rrbt_tree.nodes:
                orphans += 1
        print(f"Nodes with parents NOT in tree list: {orphans}")
        print("----------------------------\n")

    def is_key_node(node, all_nodes):
        """Check if node is structurally important: root, leaf, or junction."""
        is_root = node.parent is None

        # Always count children by traversing all nodes to ensure accuracy
        # (The .children attribute might be out of sync or uninitialized)
        num_children = sum(1 for n in all_nodes if n.parent == node)

        is_leaf = num_children == 0
        is_junction = num_children >= 2

        return is_root or is_leaf or is_junction

    # Collect key nodes and their positions
    key_nodes = [n for n in rrbt_tree.nodes if is_key_node(n, rrbt_tree.nodes)]

    # DEBUG: Count types
    n_roots = sum(1 for n in key_nodes if n.parent is None)
    n_leaves = sum(
        1 for n in key_nodes if sum(1 for x in rrbt_tree.nodes if x.parent == n) == 0
    )
    n_junctions = len(key_nodes) - n_roots - n_leaves

    print(
        f"\r   > RRBT Iter {iteration}: {len(rrbt_tree.nodes)} nodes -> {len(key_nodes)} keys "
        f"(R:{n_roots}, L:{n_leaves}, J:{n_junctions})",
        end="",
        flush=True,
    )

    # Draw key nodes as spheres
    for i, node in enumerate(key_nodes):
        pos = get_gripper_pos(node.value)
        in_light = problem.is_in_light(node.value)
        
        # Use misclassification risk for visualization
        misclass_risk = calculate_misclassification_risk(node.belief)

        # Color based on light/dark region (green for light, red for dark)
        # Alpha varies with misclassification risk (more uncertain = more transparent)
        alpha = max(0.3, min(0.9, 1.0 - misclass_risk))
        if in_light:
            color = Rgba(0, 0.8, 0.2, alpha)  # Green for light
        else:
            color = Rgba(0.9, 0.2, 0.1, alpha)  # Red for dark

        # Sphere radius proportional to misclassification risk (clamped for visibility)
        radius = max(0.005, min(0.03, misclass_risk * 0.05))

        # Ensure root is slightly more visible
        if node.parent is None:
            radius = max(radius, 0.02)  # Root should be at least 2cm

        meshcat.SetObject(f"rrbt_tree/node_{i}", Sphere(radius), color)
        meshcat.SetTransform(
            f"rrbt_tree/node_{i}", RigidTransform(RotationMatrix(), pos)
        )

    # Draw edges: trace from each leaf/junction back to previous key node
    edge_idx = 0
    for node in key_nodes:
        if node.parent is None:
            continue

        # Walk up to find the previous key node
        end_pos = get_gripper_pos(node.value)
        curr = node.parent
        while curr is not None and not is_key_node(curr, rrbt_tree.nodes):
            curr = curr.parent

        if curr is not None:
            start_pos = get_gripper_pos(curr.value)
            in_light = problem.is_in_light(node.value)
            edge_color = (
                Rgba(0.5, 1.0, 0.5, 0.7) if in_light else Rgba(1.0, 0.5, 0.5, 0.7)
            )
            meshcat.SetLine(
                f"rrbt_tree/edge_{edge_idx}",
                np.column_stack([start_pos, end_pos]),
                rgba=edge_color,
            )
            edge_idx += 1


# LEGACY FUNCTIONS - These use Kalman filter concepts and may not work with discrete Bayes filter
# Left here for reference but not recommended for use with the new system

def visualize_noisy_execution(problem, path, meshcat):
    """
    LEGACY: This function uses Kalman filter concepts (Q, R matrices) that are
    not available in the discrete Bayes filter implementation.
    
    Use the main execution diagram in main.py for proper Bayes filter execution.
    """
    print("WARNING: visualize_noisy_execution is a legacy function that uses Kalman filter.")
    print("         Use the main execution diagram for discrete Bayes filter execution.")
    raise NotImplementedError("Legacy function not compatible with discrete Bayes filter")
