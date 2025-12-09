"""
RRT and RRBT Tools

This module provides tools for RRT and RRBT (Belief-Space RRT) planning.
- RRT_tools: Standard RRT operations
- RRBT_tools: Belief-space RRT operations (generic - works with any belief type)
"""

from manipulation.exercises.trajectories.rrt_planner.rrt_planning import (
    RRT,
    TreeNode,
)
from src.simulation.simulation_tools import IiwaProblem
from src.planning.rrbt_tree import RRBT_Tree, RRBTProblem
import numpy as np


class RRT_tools:
    """Tools for standard RRT planning."""
    
    def __init__(self, problem: IiwaProblem) -> None:
        self.rrt_tree = RRT(TreeNode(problem.start), problem.cspace)
        problem.rrts = [self.rrt_tree]
        self.problem = problem

    def find_nearest_node_in_RRT_graph(self, q_sample: tuple) -> TreeNode:
        nearest_node = self.rrt_tree.nearest(q_sample)
        return nearest_node

    def sample_node_in_configuration_space(self) -> tuple:
        q_sample = self.problem.cspace.sample()
        return q_sample

    def calc_intermediate_qs_wo_collision(
        self, q_start: tuple, q_end: tuple
    ) -> list[tuple]:
        """
        Create interpolated samples from q_start to q_end.
        Returns all samples that are not in collision.

        Example interpolated path:
        q_start, qa, qb, (Obstacle), qc, q_end
        returns >>> q_start, qa, qb
        """
        return self.problem.safe_path(q_start, q_end)

    def grow_rrt_tree(self, parent_node: TreeNode, q_sample: tuple) -> TreeNode:
        """
        Add q_sample to the RRT tree as a child of the parent node.
        Returns the tree node generated from q_sample.
        """
        child_node = self.rrt_tree.add_configuration(parent_node, q_sample)
        return child_node

    def node_reaches_goal(self, node: TreeNode, tol: float = 1e-2) -> bool:
        """Returns true if the node is within tol of goal."""
        return self.problem.cspace.distance(node.value, self.problem.goal) <= tol

    def backup_path_from_node(self, node: TreeNode) -> list[tuple]:
        """Extract path from root to given node."""
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path


class RRBT_tools(RRT_tools):
    """
    Tools for Belief-Space RRT (RRBT) planning.
    
    This is a generic implementation that works with any belief type.
    The problem class defines:
    - Belief representation and initialization
    - Belief propagation
    - Cost computation
    - Goal check (termination condition)
    - Goal sampling from belief
    
    Args:
        problem: A problem instance implementing the RRBTProblem protocol
    """
    
    def __init__(self, problem: RRBTProblem) -> None:
        self.problem = problem
        self.tree = RRBT_Tree(problem)

    def sample_node(self) -> tuple:
        """Sample a random configuration from the configuration space."""
        return self.problem.cspace.sample()

    def extend_towards(self, q_rand: tuple):
        """
        Extend the tree toward q_rand using RRT* style insertion.
        
        Returns the last successfully inserted node (or the nearest node
        if no extension was possible).
        """
        # Find nearest node in tree
        dists = [
            self.problem.cspace.distance(n.value, q_rand) for n in self.tree.nodes
        ]
        nearest_idx = np.argmin(dists)
        node_near = self.tree.nodes[nearest_idx]

        # Get collision-free path from nearest to q_rand
        qs = self.calc_intermediate_qs_wo_collision(node_near.value, q_rand)
        if not qs:
            return node_near

        # Insert each waypoint along the path
        curr_parent = node_near
        for q_next in qs:
            neighbors = self.tree.get_nearest_neighbors(q_next, k=10)
            new_node = self.tree.InsertNode(q_next, neighbors, curr_parent)
            if new_node is None:
                break
            curr_parent = new_node

        return curr_parent

    def node_reaches_goal(self, node, tol=None) -> bool:
        """
        Check if node satisfies the goal condition.
        Delegates to problem.node_reaches_goal().
        
        For belief-space planning, this typically checks if uncertainty
        is below a threshold, NOT geometric distance to goal.
        """
        return self.problem.node_reaches_goal(node, tol)

    def sample_goal_from_belief(self, node):
        """
        Sample a goal configuration from the belief state.
        Delegates to problem.sample_goal_from_belief().
        
        This is the "commitment" step after uncertainty is reduced.
        """
        return self.problem.sample_goal_from_belief(node)

    def backup_path_from_node(self, node) -> list[tuple]:
        """Extract path from root to given node."""
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path
