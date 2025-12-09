"""
MustardPositionBeliefEstimatorSystem - A Drake LeafSystem for 2D (X-Y) position Kalman Filter.

This system maintains and updates the belief state (2D position covariance in X-Y plane)
based on measurement variance from the MustardPositionLightDarkRegionSensorSystem.

The Z position is fixed (bottle sits on bin floor), only X-Y uncertainty is tracked.

Belief State:
- Position mean (3D): X-Y from ICP (uncertain), Z fixed at bin surface
- Covariance (2x2): X-Y uncertainty, shrinks via Kalman updates

The covariance shrinks when the robot enters the light region (low measurement variance)
and stays constant in the dark region (high measurement variance).
"""

import numpy as np
from pydrake.all import (
    LeafSystem,
    BasicVector,
    AbstractValue,
    RigidTransform,
)


class MustardPositionBeliefEstimatorSystem(LeafSystem):
    """
    A Drake System that maintains 2D (X-Y) position Kalman Filter belief state.
    
    The system:
    - Receives measurement variance from MustardPositionLightDarkRegionSensorSystem
    - Receives initial position estimate from MustardPoseEstimatorSystem (ICP)
    - Maintains 2x2 covariance (X-Y only) as discrete state
    - Updates covariance via Kalman Filter (mean stays fixed)
    - Outputs current position mean and covariance for visualization
    
    Inputs:
        measurement_variance (1D): sigma^2 from perception system
            - In light: small (informative measurements)
            - In dark: huge (uninformative measurements)
        estimated_pose (RigidTransform): Initial pose estimate from ICP
        
    Outputs:
        position_mean (3D): [x, y, z] position estimate (Z fixed at bin surface)
        covariance (4D): Flattened 2x2 covariance matrix (X-Y only)
        
    Discrete State:
        covariance (4 elements): Flattened 2x2 covariance matrix
    """
    
    def __init__(
        self,
        initial_uncertainty: float = 0.1,
        update_period: float = 0.01,
    ):
        """
        Args:
            initial_uncertainty: Initial diagonal value for 2x2 covariance
            update_period: Kalman filter update period in seconds
        """
        LeafSystem.__init__(self)
        
        self._initial_uncertainty = initial_uncertainty
        self._position_mean = np.zeros(3)  # Will be set from ICP (Z fixed)
        self._initialized = False
        
        # Input port: measurement variance from perception system
        self._variance_port = self.DeclareVectorInputPort("measurement_variance", 1)
        
        # Input port: estimated pose from ICP (used once to initialize mean)
        self._pose_port = self.DeclareAbstractInputPort(
            "estimated_pose",
            AbstractValue.Make(RigidTransform())
        )
        
        # Discrete state: covariance matrix (flattened 2x2 = 4 elements, X-Y only)
        initial_cov = np.eye(2) * initial_uncertainty
        self._cov_state_index = self.DeclareDiscreteState(initial_cov.flatten())
        
        # Output port: position mean (3D, but Z is fixed)
        self.DeclareVectorOutputPort(
            "position_mean",
            3,
            self._CalcPositionMean,
        )
        
        # Output port: covariance (flattened 2x2 = 4D, X-Y only)
        self.DeclareVectorOutputPort(
            "covariance",
            4,
            self._CalcCovariance,
        )
        
        # Periodic discrete update for Kalman Filter covariance propagation
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=update_period,
            offset_sec=0.0,
            update=self._DoKalmanUpdate,
        )
    
    def _CalcPositionMean(self, context, output):
        """Output the position mean (from ICP, fixed after initialization)."""
        # Initialize from ICP estimate if not yet done
        if not self._initialized:
            try:
                pose = self._pose_port.Eval(context)
                position = pose.translation()
                # Only initialize if we got a valid (non-zero) position
                if np.linalg.norm(position) > 1e-6:
                    self._position_mean = position.copy()
                    self._initialized = True
            except Exception:
                pass  # Keep default zeros until ICP provides estimate
        
        output.SetFromVector(self._position_mean)
    
    def _CalcCovariance(self, context, output):
        """Output the current covariance (flattened 2x2, X-Y only)."""
        cov_flat = context.get_discrete_state(self._cov_state_index).get_value()
        output.SetFromVector(cov_flat)
    
    def _DoKalmanUpdate(self, context, discrete_state):
        """
        Kalman Filter covariance update step (2D, X-Y only).
        
        Updates the 2x2 covariance using the measurement variance received from
        MustardPositionLightDarkRegionSensorSystem.
        
        The position mean stays FIXED (we trust ICP). Only covariance shrinks
        when measurements are informative (low variance in light region).
        """
        # Initialize position mean from ICP if not yet done
        if not self._initialized:
            try:
                pose = self._pose_port.Eval(context)
                position = pose.translation()
                if np.linalg.norm(position) > 1e-6:
                    self._position_mean = position.copy()
                    self._initialized = True
            except Exception:
                pass
        
        # Get measurement variance from perception system
        variance = self._variance_port.Eval(context)[0]
        
        # Get current covariance from discrete state (2x2 flattened)
        cov_vec = discrete_state.get_mutable_vector(self._cov_state_index)
        cov_flat = cov_vec.get_mutable_value().copy()
        sigma = cov_flat.reshape(2, 2)
        
        # Only update if measurement variance is reasonable (not dark region's huge value)
        # In dark region, variance is ~1e9, so we skip update
        if variance < 1e6:
            # Kalman filter covariance update (2D, X-Y only)
            # R = measurement noise covariance (scalar * I for isotropic in X-Y)
            R = np.eye(2) * variance
            C = np.eye(2)  # Observation matrix (identity for direct X-Y observation)
            
            # S = C @ Σ @ C.T + R
            S = C @ sigma @ C.T + R
            
            # K = Σ @ C.T @ S^(-1)
            try:
                K = sigma @ C.T @ np.linalg.inv(S)
                
                # Σ_new = (I - K @ C) @ Σ
                sigma_new = (np.eye(2) - K @ C) @ sigma
                
                cov_vec.set_value(sigma_new.flatten())
            except np.linalg.LinAlgError:
                pass  # Keep current covariance if update fails
        # If in dark (huge variance), covariance unchanged
    
    def reset_covariance(self, initial_uncertainty: float = None):
        """
        Reset covariance to initial value.
        
        Called by planner when starting RRBT2 planning.
        """
        if initial_uncertainty is None:
            initial_uncertainty = self._initial_uncertainty
        self._initial_uncertainty = initial_uncertainty
        # Note: Actual reset happens through discrete state initialization
        # This method is for external configuration
    
    def set_position_mean(self, position: np.ndarray):
        """
        Manually set the position mean.
        
        Called by planner after ICP completes if direct setting is needed.
        """
        self._position_mean = np.array(position).copy()
        self._initialized = True
