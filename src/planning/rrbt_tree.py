"""
RRBT Tree - Rapidly-exploring Random Belief Tree

This implements a generic belief-space RRT tree that works with any belief
representation. The problem class defines how beliefs are propagated and
costs are computed.

The tree structure and algorithms (InsertNode, Rewire, UpdateChildren) are
generic. Problem-specific logic (belief propagation, cost function) is
delegated to the problem class.
"""

from collections import deque
from typing import Any, Protocol
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import TreeNode
import numpy as np


class BeliefNode(TreeNode):
    """
    A node in the belief-space RRT tree.
    
    Attributes:
        value: Joint configuration (7D tuple)
        parent: Parent BeliefNode
        belief: Belief state (type depends on problem - vector, matrix, etc.)
        path_length: Cumulative joint-space distance from root
        cost: Combined cost (computed by problem's cost function)
    """
    
    def __init__(self, value, parent=None, belief=None, cost=0.0, path_length=0.0):
        super().__init__(value, parent)
        self.belief = belief  # Generic - problem defines the type
        self.path_length = path_length
        self.cost = cost


class RRBTProblem(Protocol):
    """
    Protocol defining what a problem class must provide for RRBT.
    
    Problem classes should implement these methods to define:
    - How belief is represented and initialized
    - How belief propagates between configurations
    - How cost is computed from path length and belief
    """
    
    @property
    def cspace(self): ...
    
    @property
    def start(self) -> tuple: ...
    
    def get_initial_belief(self) -> Any:
        """Return the initial belief state (e.g., uniform prior)."""
        ...
    
    def propagate_belief(self, parent_node: BeliefNode, q_target: tuple) -> dict | None:
        """
        Propagate belief from parent_node to q_target.
        
        Returns dict with keys:
            - 'belief': Updated belief state
            - 'cost': Combined cost
            - 'path_length': Cumulative distance from root
        Or None if propagation fails.
        """
        ...
    
    def compute_cost(self, path_length: float, belief: Any) -> float:
        """Compute cost from path length and belief state."""
        ...


class RRBT_Tree:
    """
    Rapidly-exploring Random Belief Tree - Generic Implementation.
    
    This tree structure works with any belief representation. The problem
    class defines:
    - Initial belief (get_initial_belief)
    - Belief propagation (propagate_belief)
    - Cost computation (compute_cost)
    
    Args:
        problem: A problem instance implementing the RRBTProblem protocol
    """
    
    def __init__(self, problem: RRBTProblem):
        self.problem = problem
        self.cspace = problem.cspace

        # Initialize root with problem's initial belief
        init_belief = problem.get_initial_belief()
        init_cost = problem.compute_cost(0.0, init_belief)
        
        self.root = BeliefNode(
            value=problem.start, 
            parent=None, 
            belief=init_belief, 
            cost=init_cost,
            path_length=0.0,
        )
        self.nodes = [self.root]

    def get_nearest_neighbors(self, config, k=10):
        """Find k-nearest neighbors using configuration space distance."""
        dists = [self.cspace.distance(n.value, config) for n in self.nodes]
        indices = np.argsort(dists)[:k]
        return [self.nodes[i] for i in indices]

    def _is_ancestor(self, possible_ancestor, node):
        """
        Check if 'possible_ancestor' is found by walking up parent pointers from 'node'.
        Used to prevent cycles during rewiring.
        """
        curr = node
        while curr is not None:
            if curr == possible_ancestor:
                return True
            curr = curr.parent
        return False

    def Propagate(self, parent_node: BeliefNode, q_target: tuple) -> dict | None:
        """
        Propagate belief from parent_node to q_target.
        Delegates to problem.propagate_belief().
        
        Returns dict with 'belief', 'cost', 'path_length' or None if fails.
        """
        return self.problem.propagate_belief(parent_node, q_target)

    def InsertNode(self, q_new, neighbors, nearest_node) -> BeliefNode | None:
        """
        [RRT* Algorithm]: ChooseParent + Insert + Rewire
        
        Args:
            q_new: New configuration to insert
            neighbors: List of nearby nodes for potential rewiring
            nearest_node: The nearest node (used as parent)
            
        Returns:
            The newly created BeliefNode, or None if insertion failed
        """
        # 1. CHOOSE PARENT - Use nearest_node to maintain tree structure
        belief_result = self.Propagate(nearest_node, q_new)
        if belief_result is None:
            return None

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

        # 3. REWIRE - Check if new_node offers better paths to neighbors
        for node in neighbors:
            if node == best_parent or node == new_node:
                continue

            # Prevent cycles: don't rewire if node is an ancestor of new_node
            if self._is_ancestor(node, new_node):
                continue

            # Try connecting new_node -> neighbor
            belief_rewire = self.Propagate(new_node, node.value)

            if belief_rewire and belief_rewire["cost"] < node.cost:
                # REWIRE: new_node provides a better path to 'node'
                if node in node.parent.children:
                    node.parent.children.remove(node)

                # Attach to new parent
                node.parent = new_node
                node.belief = belief_rewire["belief"]
                node.cost = belief_rewire["cost"]
                node.path_length = belief_rewire["path_length"]
                new_node.children.append(node)

                # Propagate improvements to descendants
                self.UpdateChildren(node)

        return new_node

    def UpdateChildren(self, start_node: BeliefNode):
        """
        Propagate belief/cost updates to all descendants after rewiring.
        Uses BFS to update the subtree rooted at start_node.
        """
        queue = deque([start_node])
        iterations = 0
        MAX_OPERATIONS = len(self.nodes) * 2  # Safety limit

        while queue:
            u = queue.popleft()
            iterations += 1

            if iterations > MAX_OPERATIONS:
                print("CRITICAL ERROR: Infinite loop detected in RRBT Rewire!")
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
                    # Branch is now invalid due to parent change
                    u.children.remove(v)
