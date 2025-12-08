from manipulation.exercises.trajectories.rrt_planner.rrt_planning import (
    RRT,
    TreeNode,
)
from src.simulation.simulation_tools import IiwaProblem
import numpy as np
from src.planning.rrbt_tree import RRBT_BinBelief_Tree


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

class RRBT_BinBelief_tools(RRT_tools):
    def __init__(
        self, 
        problem
    ) -> None:
        self.problem = problem
        self.rrbt_bin_belief_tree = RRBT_BinBelief_Tree(
            problem
        )


    def sample_node(self):
        return self.problem.cspace.sample()

    def extend_towards(self, q_rand):
        dists = [
            self.problem.cspace.distance(n.value, q_rand) for n in self.rrbt_bin_belief_tree.nodes
        ]
        nearest_idx = np.argmin(dists)
        node_near = self.rrbt_bin_belief_tree.nodes[nearest_idx]

        qs = self.calc_intermediate_qs_wo_collision(node_near.value, q_rand)
        if not qs:
            return node_near

        curr_parent = node_near
        for q_next in qs:
            neighbors = self.rrbt_bin_belief_tree.get_nearest_neighbors(q_next, k=10)
            new_node = self.rrbt_bin_belief_tree.InsertNode(q_next, neighbors, curr_parent)
            if new_node is None:
                break
            curr_parent = new_node

        return curr_parent

    def node_reaches_goal(self, node, tol=None):
        """
        Active Perception Termination Condition:
        We stop ONLY when we are confident about the target's location.
        We DO NOT check if the robot is geometrically at the goal (we don't know where it is!).
        
        IMPORTANT: We check misclassification_risk, NOT the combined cost!
        The combined cost includes path_length which is irrelevant for termination.
        Termination is purely about achieving low uncertainty.
        
        misclassification_risk = 1 - max(belief)
        """
        return self.problem.node_reaches_goal(node, tol=None)

    def sample_object_position_from_belief(self, node):
        """
        Simulate the 'Commitment' step.
        Now that uncertainty is low, we use the MAP (Maximum A Posteriori)
        estimate to select which bin the object is in.
        
        For now, we still return the true goal configuration since
        the bins are abstract and will be grounded later.
        """
        # Get the MAP estimate (bin with highest probability)
        map_bin = np.argmax(node.belief)
        
        # For now, return the true goal configuration
        # Later this will be grounded to the specific bin's configuration
        true_goal = np.array(self.problem.goal)
        pred_q_goal = true_goal

        return pred_q_goal
