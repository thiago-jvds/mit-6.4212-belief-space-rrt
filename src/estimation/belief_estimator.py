"""
BeliefEstimatorSystem - A Drake LeafSystem for discrete Bayes Filter estimation.

This system maintains and updates the belief state (probability vector) based on
TPR/FPR sensor model from the LightDarkRegionSystem. It implements the estimation
layer in the Perception -> Estimation -> Visualization pipeline.

Single Source of Truth: Sensor parameters (TPR/FPR) are NOT duplicated here.
Instead, the system receives the current sensor model from the upstream
LightDarkRegionSystem, which is the single source of truth for sensor parameters.

Belief State: [P(A), P(B), P(C)] - probability that object is in each bin.
"""

import numpy as np
from pydrake.all import (
    LeafSystem,
    BasicVector,
)
from src.estimation.bayes_filter import bayes_update_all_bins


class BeliefEstimatorSystem(LeafSystem):
    """
    A Drake System that maintains discrete Bayes Filter belief state.
    
    The system:
    - Receives sensor model (TPR, FPR) from LightDarkRegionSystem
    - Maintains belief vector as discrete state
    - Updates belief via Bayes Filter when in light region
    - Outputs current belief for downstream systems (e.g., visualization)
    
    Inputs:
        sensor_model (2D): [TPR, FPR] from LightDarkRegionSystem
            - In light: informative (TPR=0.8, FPR=0.15)
            - In dark: uninformative (TPR=0.5, FPR=0.5)
        
    Outputs:
        belief (n_bins D): Probability vector [P(A), P(B), P(C)]
        
    Discrete State:
        belief (n_bins elements): Probability distribution over hypotheses
    """
    
    def __init__(
        self,
        n_bins: int = 3,
        true_bin: int = 0,
        update_period: float = 0.01,
    ):
        """
        Args:
            n_bins: Number of discrete hypothesis bins (default: 3)
            true_bin: Ground truth bin index for simulation (default: 0)
            update_period: Bayes filter update period in seconds
        """
        LeafSystem.__init__(self)
        
        self._n_bins = n_bins
        self._true_bin = true_bin
        
        # Input port: sensor model [TPR, FPR] from LightDarkRegionSystem
        self._sensor_port = self.DeclareVectorInputPort("sensor_model", 2)
        
        # Discrete state: belief vector (n_bins elements)
        # Initialize with uniform prior (maximum entropy)
        initial_belief = np.ones(n_bins) / n_bins
        self._belief_state_index = self.DeclareDiscreteState(initial_belief)
        
        # Output port: current belief vector
        self.DeclareVectorOutputPort(
            "belief",
            n_bins,
            self._CalcBelief,
        )
        
        # Periodic discrete update for Bayes Filter belief propagation
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=update_period,
            offset_sec=0.0,
            update=self._DoBayesUpdate,
        )
    
    def _CalcBelief(self, context, output):
        """Output the current belief vector."""
        belief = context.get_discrete_state(self._belief_state_index).get_value()
        output.SetFromVector(belief)
    
    def _DoBayesUpdate(self, context, discrete_state):
        """
        Bayes Filter discrete update step.
        
        Updates the belief vector using the TPR/FPR sensor model
        received from LightDarkRegionSystem (single source of truth).
        
        Only updates when in light region (TPR != 0.5).
        In dark region, measurements are uninformative, so belief unchanged.
        """
        # Get sensor model from LightDarkRegionSystem (single source of truth)
        sensor_model = self._sensor_port.Eval(context)
        tpr, fpr = sensor_model[0], sensor_model[1]
        
        # Get current belief from discrete state
        belief_vec = discrete_state.get_mutable_vector(self._belief_state_index)
        current_belief = belief_vec.get_mutable_value().copy()
        
        # Only update if sensor is informative (in light region)
        # In dark region, TPR = FPR = 0.5 (coin flip, no information)
        if abs(tpr - 0.5) > 1e-6:  # Not in dark
            # Update belief by measuring all bins
            updated_belief = bayes_update_all_bins(
                current_belief, 
                self._true_bin, 
                tpr, 
                fpr
            )
            belief_vec.set_value(updated_belief)
        # If in dark, belief unchanged (uninformative measurements)
