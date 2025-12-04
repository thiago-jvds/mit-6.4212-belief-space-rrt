from collections import deque
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import TreeNode
import numpy as np


class BeliefNode(TreeNode):
    def __init__(self, value, parent=None, sigma=None, cost=0.0):
        super().__init__(value, parent)

        # RRBT Specifics
        self.sigma = sigma  # Covariance Matrix (7x7)
        self.cost = cost  # Trace(Sigma) or cumulative objective


class RRBT_Tree:
    def __init__(self, problem, root_value, max_uncertainty, initial_uncertainty=1.0):
        self.problem = problem
        self.cspace = problem.cspace

        # We store the goal threshold, but don't prune intermediate nodes
        self.GOAL_THRESHOLD = max_uncertainty

        # Initialize Root with HIGH Uncertainty
        init_sigma = np.eye(7) * initial_uncertainty
        self.root = BeliefNode(root_value, None, init_sigma, np.trace(init_sigma))
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
        sigma_parent = parent_node.sigma

        # Get Dynamics (Q is None)
        A, _, C, R = self.problem.get_dynamics_and_observation(q_target)

        # 1. Prediction (Static Target: Sigma stays constant)
        # sigma_pred = A @ sigma_parent @ A.T + Q (where Q=0, A=I)
        sigma_pred = sigma_parent

        # 2. Update (Measurement reduces Sigma)
        S = C @ sigma_pred @ C.T + R
        try:
            K_T = np.linalg.solve(S, (sigma_pred @ C.T).T)
            K = K_T.T
        except np.linalg.LinAlgError:
            return None

        sigma_new = (np.eye(7) - K @ C) @ sigma_pred
        cost_new = np.trace(sigma_new)

        # CHANGE: No pruning here. Return belief even if cost is high.
        return {"sigma": sigma_new, "cost": cost_new}

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
            q_new, best_parent, best_belief["sigma"], best_belief["cost"]
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
                # Remove from old parent
                if node in node.parent.children:
                    node.parent.children.remove(node)

                # Attach to new parent
                node.parent = new_node
                node.sigma = belief_rewire["sigma"]
                node.cost = belief_rewire["cost"]
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
                    queue.append(v)
                else:
                    # Branch is now invalid/dead due to parent change
                    u.children.remove(v)
                    # We strictly don't delete v from self.nodes in this simple ver
                    # but it is detached from the tree.
