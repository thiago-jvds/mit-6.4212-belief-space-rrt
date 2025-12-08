"""
RRBT Tree - Rapidly-exploring Random Belief Tree

This implements belief-space RRT with a discrete 3-bin Bayes filter.
The belief state is a probability vector [P(A), P(B), P(C)] representing
the probability that the object is in each bin.

Cost Function: cost = path_length + λ × misclassification_risk
Termination:   misclassification_risk < max_uncertainty

Where misclassification_risk = 1 - max(belief)

This ensures:
- Small λ → prefer shorter paths during tree growth
- Planner won't stop until uncertainty threshold is met
- Result: shortest path that achieves goal uncertainty
"""

from collections import deque
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import TreeNode
import numpy as np
from src.simulation.simulation_tools import IiwaProblemBinBelief
from src.estimation.bayes_filter import (
    calculate_misclassification_risk,
    expected_posterior_all_bins,
)


class BeliefNode(TreeNode):
    """
    A node in the belief-space RRT tree.
    
    Attributes:
        value: Joint configuration (7D tuple)
        parent: Parent BeliefNode
        belief: Probability vector [P(A), P(B), P(C)] representing uncertainty
        path_length: Cumulative joint-space distance from root
        cost: Combined cost = path_length + λ × misclassification_risk
    """
    
    def __init__(self, value, parent=None, belief=None, cost=0.0, path_length=0.0):
        super().__init__(value, parent)

        # RRBT Specifics - using discrete Bayes filter
        self.belief = belief  # Probability vector [P(A), P(B), P(C)]
        self.path_length = path_length  # Cumulative distance from root
        self.cost = cost  # Combined: path_length + λ × misclassification_risk


class RRBT_BinBelief_Tree:
    """
    Rapidly-exploring Random Belief Tree with discrete Bayes filter.
    
    Uses a combined cost function to balance:
    - Path efficiency (minimize distance traveled)
    - Information gain (minimize misclassification risk)
    
    Args:
        problem: IiwaProblemBelief instance
        root_value: Starting configuration
        max_uncertainty: Termination threshold for misclassification_risk
        initial_uncertainty: Not used (kept for API compatibility)
        lambda_weight: Trade-off parameter (higher = prioritize uncertainty)
    """
    
    def __init__(
        self, 
        problem
    ):
        self.problem = problem
        self.cspace = problem.cspace

        # Goal threshold for termination check
        self.GOAL_THRESHOLD = problem.max_bin_uncertainty

        # Initialize Root with UNIFORM PRIOR (maximum uncertainty)
        n_bins = problem.n_bins
        init_belief = np.ones(n_bins) / n_bins  # e.g. [1/2, 1/2]
        init_cost = self.problem.cost_function(0.0, init_belief)
        
        self.root = BeliefNode(
            value=problem.start, 
            parent=None, 
            belief=init_belief, 
            cost=init_cost,
            path_length=0.0,
        )
        self.nodes = [self.root]

    def get_nearest_neighbors(self, config, k=10):
        """Finds k-nearest neighbors (Simple Euclidean for now)"""
        dists = [self.cspace.distance(n.value, config) for n in self.nodes]
        # Get indices of k smallest distances
        indices = np.argsort(dists)[:k]
        return [self.nodes[i] for i in indices]

    def _is_ancestor(self, possible_ancestor, node):
        """
        Returns True if 'possible_ancestor' is found by walking up
        the parent pointers from 'node'.
        """
        curr = node
        while curr is not None:
            if curr == possible_ancestor:
                return True
            curr = curr.parent
        return False

    def Propagate(self, parent_node, q_target):
        """
        Propagate belief from parent_node to q_target using discrete Bayes filter.
        
        In light region: Uses expected_posterior_all_bins() to compute
                        expected belief after measuring all 3 bins,
                        assuming the object is in true_bin (for planning)
        In dark region: Belief unchanged (uninformative measurements)
        
        Returns dict with:
            - belief: Updated probability vector [P(A), P(B), P(C)]
            - path_length: Cumulative distance from root
            - cost: Combined cost = path_length + λ × misclassification_risk
        """
        belief_parent = parent_node.belief

        # Get sensor model (TPR, FPR) based on robot location
        tpr, fpr = self.problem.get_sensor_model(q_target)

        # Update belief based on light/dark region
        if self.problem.is_in_light(q_target):
            # In light: measure all bins (expected posterior for planning)
            # We assume the object is in true_bin for computing expected posterior
            belief_new = expected_posterior_all_bins(
                belief_parent, tpr, fpr, 
                assumed_bin=self.problem.true_bin
            )
        else:
            # In dark: uninformative measurements, belief unchanged
            belief_new = belief_parent.copy()
        
        # Compute path length increment
        dist_increment = self.cspace.distance(parent_node.value, q_target)
        path_length_new = parent_node.path_length + dist_increment
        
        # Compute combined cost using misclassification risk
        cost_new = self.problem.cost_function(path_length_new, belief_new)

        return {
            "belief": belief_new, 
            "cost": cost_new,
            "path_length": path_length_new,
        }

    def InsertNode(self, q_new, neighbors, nearest_node):
        """[Paper Algo 1]: ChooseParent + Insert + Rewire"""

        # 1. CHOOSE PARENT
        # Use nearest_node as the parent to maintain tree structure.
        # (Previously tried ALL neighbors, but this caused everything to connect
        # to ROOT since ROOT has lowest uncertainty, creating a "fan" not a tree)
        belief_result = self.Propagate(nearest_node, q_new)
        if belief_result is None:
            return None  # Can't connect to intended parent

        best_parent = nearest_node
        best_belief = belief_result

        # 2. CREATE NODE
        new_node = BeliefNode(
            q_new, 
            best_parent, 
            best_belief["belief"], 
            best_belief["cost"],
            best_belief["path_length"],
        )
        best_parent.children.append(new_node)
        self.nodes.append(new_node)

        # 3. REWIRE (The Queue / Section V.B)
        # Does new_node offer a better path to existing neighbors?
        for node in neighbors:
            if node == best_parent:
                continue

            if node == new_node:
                continue

            # We are considering making 'new_node' the parent of 'node'.
            # We must ensure 'node' is not currently an ancestor of 'new_node'.
            # If 'node' is an ancestor of 'new_node', making 'new_node' the parent
            # of 'node' creates a loop.
            if self._is_ancestor(node, new_node):
                continue

            # Try connecting new_node -> neighbor
            belief_rewire = self.Propagate(new_node, node.value)

            if belief_rewire and belief_rewire["cost"] < node.cost:
                # REWIRE DETECTED!
                if node in node.parent.children:
                    node.parent.children.remove(node)

                # Attach to new parent
                node.parent = new_node
                node.belief = belief_rewire["belief"]
                node.cost = belief_rewire["cost"]
                node.path_length = belief_rewire["path_length"]
                new_node.children.append(node)

                # Propagate improvements downstream
                self.UpdateChildren(node)

        return new_node

    def UpdateChildren(self, start_node):
        """[Paper V.B]: The Queue to update descendants."""
        queue = deque([start_node])

        iterations = 0
        MAX_OPERATIONS = len(self.nodes) * 2  # Generous upper bound

        while queue:
            u = queue.popleft()
            iterations += 1

            if iterations > MAX_OPERATIONS:
                print("CRITICAL ERROR: Infinite Loop detected in RRBT Rewire!")
                print("   A cycle likely exists in the tree.")
                break

            for v in list(u.children):
                belief_result = self.Propagate(u, v.value)

                if belief_result:
                    v.belief = belief_result["belief"]
                    v.cost = belief_result["cost"]
                    v.path_length = belief_result["path_length"]
                    queue.append(v)
                else:
                    # Branch is now invalid/dead due to parent change
                    u.children.remove(v)
                    # We strictly don't delete v from self.nodes in this simple ver
                    # but it is detached from the tree.
