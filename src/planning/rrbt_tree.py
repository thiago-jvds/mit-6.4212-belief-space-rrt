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
    """
    The 'Graph' described in the Paper.
    Handles Propagate, Insert, Rewire, and the Priority Queue.
    """

    def __init__(self, problem, root_value, max_uncertainty):
        self.problem = problem
        self.cspace = problem.cspace
        self.MAX_UNCERTAINTY = max_uncertainty

        # Initialize Root
        self.root = BeliefNode(
            root_value, 
            parent=None, 
            sigma=np.eye(7) * 1e-6, 
            cost=np.trace(np.eye(7) * 1e-6)
        )
        self.nodes = [self.root]  # Keep track of all nodes for 'Nearest'

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
        """[Paper V.A]: Belief Propagation (Physics)"""
        sigma_parent = parent_node.sigma
        A, Q, C, R = self.problem.get_dynamics_and_observation(q_target)

        # KF Prediction & Update
        sigma_pred = A @ sigma_parent @ A.T + Q
        S = C @ sigma_pred @ C.T + R
        try:
            K_T = np.linalg.solve(S, (sigma_pred @ C.T).T)
            K = K_T.T
        except np.linalg.LinAlgError:
            return None

        sigma_new = (np.eye(7) - K @ C) @ sigma_pred
        cost_new = np.trace(sigma_new)

        if cost_new > self.MAX_UNCERTAINTY:
            return None

        return {"sigma": sigma_new, "cost": cost_new}

    def InsertNode(self, q_new, neighbors, nearest_node):
        """[Paper Algo 1]: ChooseParent + Insert + Rewire"""
        
        # 1. CHOOSE BEST PARENT (from ALL neighbors, not just nearest)
        best_parent = None
        best_belief = None

        # Try nearest_node first
        belief = self.Propagate(nearest_node, q_new)
        if belief is not None:
            best_belief = belief
            best_parent = nearest_node

        # Try ALL neighbors to find a valid parent with lowest cost
        for node in neighbors:
            if node == nearest_node:
                continue
            
            # Geometric check (Simplified: assumes localized connection is safe)
            # In full implementation, check collision(node -> q_new) here
            
            # # Check if path from node to q_new is collision-free
            # if not self.problem.safe_path(node.value, q_new):
            #     continue  # Skip this neighbor node b/c path has collision
            
            belief = self.Propagate(node, q_new)
            if belief is not None:
                if best_belief is None or belief["cost"] < best_belief["cost"]:
                    best_belief = belief
                    best_parent = node

        # Only fail if NO neighbor can serve as parent
        if best_belief is None:
            return None

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
