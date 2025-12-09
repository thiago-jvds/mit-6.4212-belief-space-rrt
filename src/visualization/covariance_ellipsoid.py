"""
CovarianceEllipsoidSystem - A Drake LeafSystem for visualizing 2D (X-Y) covariance as flat ellipsoid.

This system renders a flat ellipsoid (disk) in Meshcat centered at the estimated position,
with X-Y axes scaled by the eigenvalues of the 2x2 covariance matrix. As uncertainty decreases
(when in light region), the ellipsoid shrinks.

The ellipsoid is flat on the X-Y plane (small Z height), representing uncertainty in the
horizontal plane where the bottle sits on the bin floor.
"""

import numpy as np
from pydrake.all import (
    LeafSystem,
    Meshcat,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Ellipsoid,
)


class CovarianceEllipsoidSystem(LeafSystem):
    """
    A Drake System that visualizes 2D (X-Y) covariance as a flat ellipsoid in Meshcat.
    
    The system:
    - Receives position mean (3D) and covariance (4D flattened 2x2) from estimator
    - Computes eigendecomposition of 2x2 covariance
    - Renders flat ellipsoid with X-Y axes scaled by sqrt(eigenvalues) * scale_factor
    - Z axis has a small fixed height (flat disk on bin surface)
    - Color: Red with alpha transparency
    - Updates visualization during Publish events
    
    Inputs:
        position (3D): [x, y, z] position estimate (ellipsoid center)
        covariance (4D): Flattened 2x2 covariance matrix (X-Y only)
    """
    
    def __init__(
        self,
        meshcat: Meshcat,
        scale_factor: float = 3.0,  # 3-sigma ellipsoid
        color: Rgba = Rgba(1.0, 0.0, 0.0, 0.5),  # Red with 50% transparency
        min_radius: float = 0.01,  # Minimum radius for visibility
        max_radius: float = 1.0,   # Maximum radius to prevent huge ellipsoids
        z_height: float = 0.01,    # Fixed Z height for flat ellipsoid (thin disk)
        publish_period: float = 0.02,
        name: str = "covariance_ellipsoid",
    ):
        """
        Args:
            meshcat: Meshcat instance for visualization
            scale_factor: Multiplier for sqrt(eigenvalues) (default: 3.0 for 3-sigma)
            color: Ellipsoid color (default: red with 50% alpha)
            min_radius: Minimum axis radius for visibility
            max_radius: Maximum axis radius to prevent huge ellipsoids
            z_height: Fixed Z-axis height for flat ellipsoid (default: 0.01m = 1cm)
            publish_period: Meshcat publish period in seconds
            name: Meshcat path name for the ellipsoid
        """
        LeafSystem.__init__(self)
        
        self._meshcat = meshcat
        self._scale_factor = scale_factor
        self._color = color
        self._min_radius = min_radius
        self._max_radius = max_radius
        self._z_height = z_height
        self._name = name
        
        # Input port: position mean (3D)
        self._position_port = self.DeclareVectorInputPort("position", 3)
        
        # Input port: covariance (flattened 2x2 = 4D, X-Y only)
        self._covariance_port = self.DeclareVectorInputPort("covariance", 4)
        
        # Periodic publish for Meshcat visualization
        self.DeclarePeriodicPublishEvent(
            period_sec=publish_period,
            offset_sec=0.0,
            publish=self._DoPublishEllipsoid,
        )
        
        # Create initial ellipsoid object in Meshcat
        if meshcat is not None:
            self._setup_meshcat_object()
    
    def _setup_meshcat_object(self):
        """Create initial Meshcat ellipsoid object."""
        # Start with a small flat ellipsoid (will be updated on first publish)
        initial_radii = np.array([0.05, 0.05, self._z_height])
        self._meshcat.SetObject(
            self._name,
            Ellipsoid(initial_radii[0], initial_radii[1], initial_radii[2]),
            self._color
        )
        # Place at origin initially
        self._meshcat.SetTransform(
            self._name,
            RigidTransform()
        )
    
    def _DoPublishEllipsoid(self, context):
        """
        Publish the flat ellipsoid to Meshcat.
        
        Reads position and 2x2 covariance from input ports, computes eigendecomposition,
        and updates ellipsoid size and position. The ellipsoid is flat in the X-Y plane.
        """
        if self._meshcat is None:
            return
        
        # Get position (ellipsoid center)
        position = self._position_port.Eval(context)
        
        # Get covariance and reshape to 2x2 (X-Y only)
        cov_flat = self._covariance_port.Eval(context)
        covariance_2d = cov_flat.reshape(2, 2)
        
        # Skip update if position is all zeros (not yet initialized)
        if np.linalg.norm(position) < 1e-6:
            return
        
        # Ensure covariance is symmetric (numerical stability)
        covariance_2d = (covariance_2d + covariance_2d.T) / 2
        
        # Eigendecomposition for ellipsoid axes (2D in X-Y plane)
        try:
            eigenvalues_2d, eigenvectors_2d = np.linalg.eigh(covariance_2d)
            
            # Clamp eigenvalues to be positive (numerical stability)
            eigenvalues_2d = np.maximum(eigenvalues_2d, 1e-10)
            
            # Compute ellipsoid radii for X-Y (scale_factor * sqrt(eigenvalue))
            radii_xy = self._scale_factor * np.sqrt(eigenvalues_2d)
            
            # Clamp radii to reasonable range
            radii_xy = np.clip(radii_xy, self._min_radius, self._max_radius)
            
            # Build 3D rotation matrix from 2D eigenvectors
            # The 2D eigenvectors define rotation in X-Y plane
            # Z axis stays aligned with world Z
            R_2d = eigenvectors_2d
            
            # Ensure proper rotation (det = +1)
            if np.linalg.det(R_2d) < 0:
                R_2d[:, 0] *= -1
            
            # Construct 3D rotation matrix (rotation only in X-Y plane)
            R_3d = np.eye(3)
            R_3d[0:2, 0:2] = R_2d
            
            rotation = RotationMatrix(R_3d)
            
            # Full radii: [X, Y, Z_fixed]
            radii = np.array([radii_xy[0], radii_xy[1], self._z_height])
            
        except np.linalg.LinAlgError:
            # Fallback to small flat disk if eigendecomposition fails
            radii = np.array([0.1, 0.1, self._z_height])
            rotation = RotationMatrix()
        
        # Get current time for recording
        current_time = context.get_time()
        
        # Update ellipsoid shape
        self._meshcat.SetObject(
            self._name,
            Ellipsoid(radii[0], radii[1], radii[2]),
            self._color
        )
        
        # Update ellipsoid transform (position + rotation)
        X_WE = RigidTransform(rotation, position)
        self._meshcat.SetTransform(
            self._name,
            X_WE,
            time_in_recording=current_time
        )
