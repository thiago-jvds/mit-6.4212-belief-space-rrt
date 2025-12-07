"""
BeliefEstimatorSystem - A Drake LeafSystem for Kalman Filter belief estimation.

This system maintains and updates the belief state (covariance matrix) based on
measurement variance from the LightDarkRegionSystem. It implements the estimation
layer in the Perception -> Estimation -> Visualization pipeline.

Single Source of Truth: Measurement noise parameters are NOT duplicated here.
Instead, the system receives the current measurement variance from the upstream
LightDarkRegionSystem, which is the single source of truth for noise parameters.
"""

import numpy as np
from pydrake.all import (
    LeafSystem,
    BasicVector,
)


class BeliefEstimatorSystem(LeafSystem):
    """
    A Drake System that maintains Kalman Filter belief state.
    
    The system:
    - Receives measurement variance from LightDarkRegionSystem (single source of truth)
    - Maintains covariance matrix as discrete state
    - Updates belief via Kalman Filter using the received variance
    - Outputs current covariance for downstream systems (e.g., visualization)
    
    Inputs:
        measurement_variance (1D): Current measurement variance (sigma^2)
            from LightDarkRegionSystem - this is the single source of truth
        
    Outputs:
        covariance (49D): Flattened 7x7 covariance matrix
        
    Discrete State:
        sigma (7x7 flattened to 49D): Covariance matrix representing target uncertainty
    """
    
    def __init__(
        self,
        initial_sigma=None,
        update_period=0.01,
    ):
        """
        Args:
            initial_sigma: Initial 7x7 covariance matrix (default: I)
            update_period: Kalman filter update period in seconds
        """
        LeafSystem.__init__(self)
        
        # Input port: measurement variance from LightDarkRegionSystem
        # This is the single source of truth for noise - no R_light/R_dark stored here
        self._variance_port = self.DeclareVectorInputPort("measurement_variance", 1)
        
        # Discrete state: flattened covariance matrix (7x7 = 49 elements)
        if initial_sigma is None:
            initial_sigma = np.eye(7) * 1.0  # High initial uncertainty
        self._initial_sigma = initial_sigma
        self._sigma_state_index = self.DeclareDiscreteState(initial_sigma.flatten())
        
        # Output port: current covariance (flattened 7x7)
        self.DeclareVectorOutputPort(
            "covariance",
            49,  # 7x7 flattened
            self._CalcCovariance,
        )
        
        # Periodic discrete update for Kalman Filter belief propagation
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=update_period,
            offset_sec=0.0,
            update=self._DoKalmanUpdate,
        )
    
    def _CalcCovariance(self, context, output):
        """Output the current covariance matrix (flattened)."""
        sigma_flat = context.get_discrete_state(self._sigma_state_index).get_value()
        output.SetFromVector(sigma_flat)
    
    def _DoKalmanUpdate(self, context, discrete_state):
        """
        Kalman Filter discrete update step.
        
        Updates the covariance matrix using the measurement variance
        received from LightDarkRegionSystem (single source of truth).
        """
        # Get measurement variance from LightDarkRegionSystem (single source of truth)
        variance = self._variance_port.Eval(context)[0]
        
        # Build observation noise covariance R = variance * I
        R = np.eye(7) * variance
        
        # Get current covariance from discrete state
        sigma_flat = discrete_state.get_mutable_vector(self._sigma_state_index)
        sigma = sigma_flat.get_mutable_value().reshape(7, 7)
        
        # Kalman Filter update (static target: A=I, C=I, Q=0)
        # Prediction: sigma_pred = A @ sigma @ A.T + Q = sigma (no process noise)
        sigma_pred = sigma
        
        # Update step
        C = np.eye(7)
        S = C @ sigma_pred @ C.T + R  # Innovation covariance
        K = sigma_pred @ C.T @ np.linalg.inv(S)  # Kalman gain
        sigma_new = (np.eye(7) - K @ C) @ sigma_pred  # Updated covariance
        
        # Write back to state
        sigma_flat.set_value(sigma_new.flatten())
