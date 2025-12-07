"""
BeliefVisualizerSystem - A Drake LeafSystem for visualizing goal uncertainty.

This system renders an uncertainty ellipsoid in Meshcat based on covariance
data received from BeliefEstimatorSystem. It implements the visualization
layer in the Perception -> Estimation -> Visualization pipeline.

The system is purely for visualization - it contains no estimation logic.
"""

import numpy as np
from pydrake.all import (
    LeafSystem,
    Sphere,
    Ellipsoid,
    Rgba,
    JacobianWrtVariable,
    Meshcat,
    RigidTransform,
    RotationMatrix,
)


class BeliefVisualizerSystem(LeafSystem):
    """
    A Drake System that visualizes goal uncertainty as an ellipsoid in Meshcat.
    
    The system:
    - Receives covariance matrix from BeliefEstimatorSystem
    - Projects 7D joint-space covariance to 3D task-space at goal
    - Renders uncertainty ellipsoid in Meshcat during Publish events
    
    This system is stateless (no discrete state) - it only visualizes
    what it receives from upstream estimation systems.
    
    Inputs:
        covariance (49D): Flattened 7x7 covariance matrix from BeliefEstimatorSystem
    """
    
    def __init__(
        self,
        meshcat: Meshcat,
        plant,
        goal_config,
        publish_period=0.02,
    ):
        """
        Args:
            meshcat: Meshcat instance for visualization
            plant: MultibodyPlant for forward kinematics (goal position/Jacobian)
            goal_config: 7D array of goal joint configuration
            publish_period: Meshcat publish period in seconds
        """
        LeafSystem.__init__(self)
        
        # Store references
        self._meshcat: Meshcat | None = meshcat
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        
        # Problem parameters
        self._goal_config = np.array(goal_config)
        
        # Precompute goal position and Jacobian (fixed throughout simulation)
        self._setup_goal_kinematics()
        
        # Input port: covariance from BeliefEstimatorSystem
        self._covariance_port = self.DeclareVectorInputPort("covariance", 49)
        
        # Periodic publish for Meshcat visualization
        self.DeclarePeriodicPublishEvent(
            period_sec=publish_period,
            offset_sec=0.0,
            publish=self._DoPublishEllipsoid,
        )
        
        # Setup Meshcat visualization objects (only if meshcat is provided)
        if meshcat is not None:
            self._setup_meshcat_objects()
    
    def _setup_goal_kinematics(self):
        """Precompute goal position and Jacobian for task-space projection."""
        iiwa = self._plant.GetModelInstanceByName("iiwa")
        wsg_body = self._plant.GetBodyByName("body", self._plant.GetModelInstanceByName("wsg"))
        wsg_frame = self._plant.GetFrameByName("body", self._plant.GetModelInstanceByName("wsg"))
        
        # Set plant to goal configuration
        self._plant.SetPositions(self._plant_context, iiwa, self._goal_config)
        X_Goal = self._plant.EvalBodyPoseInWorld(self._plant_context, wsg_body)
        self._goal_position = X_Goal.translation()
        self._X_Goal = X_Goal
        
        # Compute Jacobian at goal: maps joint velocities to gripper velocity
        p_BoBi_B = np.array([[0.0], [0.0], [0.0]])  # Point at gripper frame origin
        J = self._plant.CalcJacobianTranslationalVelocity(
            self._plant_context,
            with_respect_to=JacobianWrtVariable.kQDot,
            frame_B=wsg_frame,
            p_BoBi_B=p_BoBi_B,
            frame_A=self._plant.world_frame(),
            frame_E=self._plant.world_frame(),
        )
        # Extract only iiwa columns (first 7)
        self._J_goal = J[:, :7]
    
    def _setup_meshcat_objects(self):
        """Create initial Meshcat objects for visualization."""
        # Small marker at true goal position (static, doesn't change)
        self._meshcat.SetObject(
            "goal_uncertainty/goal_marker",
            Sphere(0.02),
            Rgba(0.0, 1.0, 0.0, 0.8)  # Green
        )
        self._meshcat.SetTransform("goal_uncertainty/goal_marker", self._X_Goal)
        
        # Initial ellipsoid - will be recreated each frame with correct size
        initial_radius = 0.3
        self._meshcat.SetObject(
            "goal_uncertainty/ellipsoid",
            Ellipsoid(initial_radius, initial_radius, initial_radius),
            Rgba(1.0, 0.3, 0.0, 0.25)  # Orange, semi-transparent
        )
        self._meshcat.SetTransform(
            "goal_uncertainty/ellipsoid",
            RigidTransform(self._goal_position)
        )
    
    def _DoPublishEllipsoid(self, context):
        """
        Publish the uncertainty ellipsoid to Meshcat.
        
        Reads covariance from input port, projects to task-space,
        and renders as an ellipsoid at the goal position.
        """
        # Skip if no meshcat instance
        if self._meshcat is None:
            return
        
        # Get covariance from input port (from BeliefEstimatorSystem)
        sigma_flat = self._covariance_port.Eval(context)
        sigma = sigma_flat.reshape(7, 7)
        
        # Project to 3D task-space: Sigma_3D = J @ Sigma_7D @ J^T
        sigma_3d = self._J_goal @ sigma @ self._J_goal.T
        
        # Eigendecomposition for ellipsoid axes
        eigvals, eigvecs = np.linalg.eigh(sigma_3d)
        
        # Clamp eigenvalues to avoid numerical issues
        eigvals = np.maximum(eigvals, 1e-8)
        
        # Convert to ellipsoid radii (3-sigma = 99.7% confidence)
        radii = 3.0 * np.sqrt(eigvals)
        radii = np.clip(radii, 0.01, 0.5)  # Clamp for visibility
        
        # Create new Ellipsoid shape with the computed semi-axes
        ellipsoid_shape = Ellipsoid(radii[0], radii[1], radii[2])
        
        # Recreate the object with new dimensions
        self._meshcat.SetObject(
            "goal_uncertainty/ellipsoid",
            ellipsoid_shape,
            Rgba(1.0, 0.3, 0.0, 0.25)  # Orange, semi-transparent
        )
        
        # Create RigidTransform with rotation (principal axes) and translation
        # The eigenvectors form a rotation matrix (columns are principal axes)
        # Ensure it's a proper rotation matrix (det = +1)
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 0] *= -1  # Flip first column to ensure right-handed
        
        rotation = RotationMatrix(eigvecs)
        transform = RigidTransform(rotation, self._goal_position)
        
        # Set transform with time_in_recording for animation playback
        current_time = context.get_time()
        self._meshcat.SetTransform(
            "goal_uncertainty/ellipsoid",
            transform,
            time_in_recording=current_time
        )
