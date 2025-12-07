"""
RRBT Tree - Rapidly-exploring Random Belief Tree

This implements the belief-space RRT with a combined cost function that
balances path length (distance traveled) and uncertainty (trace of covariance).

Cost Function: cost = path_length + λ × trace(Σ)
Termination:   trace(Σ) < max_uncertainty

This separation ensures:
- Small λ → prefer shorter paths during tree growth
- But planner won't stop until uncertainty threshold is met
- Result: shortest path that achieves goal uncertainty
"""

from collections import deque
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import TreeNode
import numpy as np


class BeliefNode(TreeNode):
    """
    A node in the belief-space RRT tree.
    
    Attributes:
        value: Joint configuration (7D tuple)
        parent: Parent BeliefNode
        sigma: Covariance matrix (7x7) representing uncertainty
        path_length: Cumulative joint-space distance from root
        cost: Combined cost = path_length + λ × trace(Σ)
    """
    
    def __init__(self, value, parent=None, sigma=None, cost=0.0, path_length=0.0):
        super().__init__(value, parent)

        # RRBT Specifics
        self.sigma = sigma  # Covariance Matrix (7x7)
        self.path_length = path_length  # Cumulative distance from root
        self.cost = cost  # Combined: path_length + λ × trace(Σ)


class RRBT_Tree:
    """
    Rapidly-exploring Random Belief Tree.
    
    Uses a combined cost function to balance:
    - Path efficiency (minimize distance traveled)
    - Information gain (minimize uncertainty)
    
    Args:
        problem: IiwaProblemBelief instance
        root_value: Starting configuration
        max_uncertainty: Termination threshold for trace(Σ)
        initial_uncertainty: Initial diagonal value for Σ₀
        lambda_weight: Trade-off parameter (higher = prioritize uncertainty)
    """
    
    def __init__(
        self, 
        problem, 
        root_value, 
        max_uncertainty, 
        initial_uncertainty=1.0,
        lambda_weight=1.0,
    ):
        self.problem = problem
        self.cspace = problem.cspace
        self.lambda_weight = lambda_weight

        # We store the goal threshold for termination check
        self.GOAL_THRESHOLD = max_uncertainty

        # Initialize Root with HIGH Uncertainty, ZERO path length
        init_sigma = np.eye(7) * initial_uncertainty
        init_trace = np.trace(init_sigma)
        init_cost = 0.0 + lambda_weight * init_trace  # path_length=0 at root
        
        self.root = BeliefNode(
            root_value, 
            parent=None, 
            sigma=init_sigma, 
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
        Propagate belief from parent_node to q_target.
        
        Returns dict with:
            - sigma: Updated covariance matrix
            - path_length: Cumulative distance from root
            - cost: Combined cost = path_length + λ × trace(Σ)
        """
        sigma_parent = parent_node.sigma

        # Get Dynamics (Q is None for static target)
        A, _, C, R = self.problem.get_dynamics_and_observation(q_target)

        # 1. Prediction (Static Target: Sigma stays constant)
        # sigma_pred = A @ sigma_parent @ A.T + Q (where Q=0, A=I)
        sigma_pred = sigma_parent

        # 2. Update (Measurement reduces Sigma based on light/dark region)
        S = C @ sigma_pred @ C.T + R
        try:
            K_T = np.linalg.solve(S, (sigma_pred @ C.T).T)
            K = K_T.T
        except np.linalg.LinAlgError:
            return None

        sigma_new = (np.eye(7) - K @ C) @ sigma_pred
        
        # 3. Compute path length increment
        dist_increment = self.cspace.distance(parent_node.value, q_target)
        path_length_new = parent_node.path_length + dist_increment
        
        # 4. Compute combined cost
        trace_sigma = np.trace(sigma_new)
        cost_new = path_length_new + self.lambda_weight * trace_sigma

        return {
            "sigma": sigma_new, 
            "cost": cost_new,
            "path_length": path_length_new,
        }

    def InsertNode(self, q_new, neighbors, nearest_node):
        """[Paper Algo 1]: ChooseParent + Insert + Rewire"""

        # 1. CHOOSE PARENT
        # Use nearest_node as the parent to maintain tree structure.
        # (Previously tried ALL neighbors, but this caused everything to connect
        # to ROOT since ROOT has lowest uncertainty, creating a "fan" not a tree)
        belief = self.Propagate(nearest_node, q_new)
        if belief is None:
            return None  # Can't connect to intended parent

        best_parent = nearest_node
        best_belief = belief

        # 2. CREATE NODE
        new_node = BeliefNode(
            q_new, 
            best_parent, 
            best_belief["sigma"], 
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

            # # Check if path from new_node to node is collision-free
            # if not self.problem.safe_path(new_node.value, node.value):
            #     continue  # Skip this neighbor node b/c path has collision

            # Try connecting new_node -> neighbor
            belief_rewire = self.Propagate(new_node, node.value)

            if belief_rewire and belief_rewire["cost"] < node.cost:
                # REWIRE DETECTED!
                if node in node.parent.children:
                    node.parent.children.remove(node)

                # Attach to new parent
                node.parent = new_node
                node.sigma = belief_rewire["sigma"]
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
                print("🚨 CRITICAL ERROR: Infinite Loop detected in RRBT Rewire!")
                print("   A cycle likely exists in the tree.")
                break

            for v in list(u.children):
                belief = self.Propagate(u, v.value)

                if belief:
                    v.sigma = belief["sigma"]
                    v.cost = belief["cost"]
                    v.path_length = belief["path_length"]
                    queue.append(v)
                else:
                    # Branch is now invalid/dead due to parent change
                    u.children.remove(v)
                    # We strictly don't delete v from self.nodes in this simple ver
                    # but it is detached from the tree.
