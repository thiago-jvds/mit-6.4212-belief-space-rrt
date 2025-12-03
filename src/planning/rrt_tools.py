from manipulation.exercises.trajectories.rrt_planner.rrt_planning import (
    RRT,
    TreeNode,
)
from src.simulation.simulation_tools import IiwaProblem
import numpy as np
from src.planning.rrbt_tree import RRBT_Tree


class RRT_tools:
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
        """create more samples by linear interpolation from q_start
        to q_end. Return all samples that are not in collision

        Example interpolated path:
        q_start, qa, qb, (Obstacle), qc , q_end
        returns >>> q_start, qa, qb
        """
        return self.problem.safe_path(q_start, q_end)

    def grow_rrt_tree(self, parent_node: TreeNode, q_sample: tuple) -> TreeNode:
        """
        add q_sample to the rrt tree as a child of the parent node
        returns the rrt tree node generated from q_sample
        """
        child_node = self.rrt_tree.add_configuration(parent_node, q_sample)
        return child_node

    def node_reaches_goal(self, node: TreeNode, tol: float = 1e-2) -> bool:
        "returns true if the node is within tol of goal, false otherwise"
        return self.problem.cspace.distance(node.value, self.problem.goal) <= tol

    def backup_path_from_node(self, node: TreeNode) -> list[tuple]:
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path

class RRBT_tools(RRT_tools):
    def __init__(self, problem, max_uncertainty: float = 0.1) -> None:
        self.problem = problem
        self.MAX_UNCERTAINTY = max_uncertainty

        self.rrbt_tree = RRBT_Tree(problem, problem.start, max_uncertainty)

        self.rejected_collision = 0
        self.accepted_nodes = 0
        self.rejected_uncertainty = 0

    def sample_node(self):
        return self.problem.cspace.sample()

    def extend_towards(self, q_rand):
        """
        Steers towards q_rand and inserts nodes using RRBT logic.
        """
        # 1. Nearest (Geometric)
        # We use the tree's internal list of nodes
        # Find closest node in tree to q_rand
        dists = [
            self.problem.cspace.distance(n.value, q_rand) for n in self.rrbt_tree.nodes
        ]
        nearest_idx = np.argmin(dists)
        node_near = self.rrbt_tree.nodes[nearest_idx]

        # 2. Interpolate (Geometric path)
        qs = self.calc_intermediate_qs_wo_collision(node_near.value, q_rand)
        if not qs:
            return node_near

        curr_parent = node_near

        # 3. Insert Step-by-Step
        for q_next in qs:
            # Find neighbors for RRT* rewiring (radius ~ 2.0 or dynamic)
            neighbors = self.rrbt_tree.get_nearest_neighbors(q_next, k=10)

            # The Tree handles ChooseParent + Rewire
            new_node = self.rrbt_tree.InsertNode(q_next, neighbors, curr_parent)

            if new_node is None:
                break  # Propagation failed

            curr_parent = new_node

        return curr_parent

    def node_reaches_goal(self, node, tol=0.1):
        return self.problem.cspace.distance(node.value, self.problem.goal) < tol

    def backup_path(self, node):
        path = []
        while node:
            path.append(node.value)
            node = node.parent
        return path[::-1]

    def print_stats(self):
        total = (
            self.accepted_nodes + self.rejected_collision + self.rejected_uncertainty
        )
        if total == 0:
            total = 1
        print(
            f"Stats: Accepted: {self.accepted_nodes} | "
            f"Rej(Collision): {self.rejected_collision} | "
            f"Rej(Uncertainty): {self.rejected_uncertainty} ({self.rejected_uncertainty / total:.1%})"
        )
