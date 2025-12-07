#!/usr/bin/env python3
"""
Observation Model Testing Script - Visualizes the robot scenario with wrist camera in Meshcat.

Starts Meshcat on port 7000, loads the obs_modeling_scenario with the wrist camera enabled,
and keeps running until you press Ctrl+C.

Usage:
    python obs_model_testing.py

Then open http://localhost:7000 in your browser.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass, field
from manipulation.station import MakeHardwareStation, LoadScenario
from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    Meshcat,
    MeshcatParams,
    ConstantVectorSource,
    Box,
    Rgba,
    RigidTransform,
    RotationMatrix,
    PiecewisePolynomial,
    TrajectorySource,
)
from pydrake.multibody.math import SpatialVelocity
from src.perception.light_and_dark import LightDarkRegionSystem
from src.utils.config_loader import load_rrbt_config
from src.planning.standard_rrt import rrt_planning
from src.simulation.simulation_tools import IiwaProblem
from src.utils.ik_solver import solve_ik_for_pose


def path_to_trajectory(path: list, time_per_segment: float = 0.1) -> PiecewisePolynomial:
    """
    Convert a list of joint configurations to a time-parameterized trajectory.
    
    Args:
        path: List of joint configurations (each is a 7-element array/tuple)
        time_per_segment: Time allocated per path segment in seconds
        
    Returns:
        PiecewisePolynomial trajectory (7D output matching iiwa.position port)
    """
    # Convert path to numpy array (n_points x 7)
    path_array = np.array([np.array(q) for q in path])
    
    # Create time breakpoints
    n_points = len(path)
    times = np.linspace(0, (n_points - 1) * time_per_segment, n_points)
    
    # Create trajectory using first-order hold (linear interpolation)
    # PiecewisePolynomial.FirstOrderHold expects:
    # - breaks: 1D array of times
    # - samples: 2D array where each column is a sample (7 x n_points)
    trajectory = PiecewisePolynomial.FirstOrderHold(times, path_array.T)
    
    return trajectory


@dataclass
class VoxelGrid:
    """
    3D Voxel Grid for occupancy mapping.
    
    Each voxel stores a probability of occupancy (0 = free, 1 = occupied, 0.5 = unknown).
    The grid is defined by its origin (corner), dimensions (number of voxels), and voxel size.
    
    Attributes:
        origin: (x, y, z) world coordinates of the grid's minimum corner
        dimensions: (nx, ny, nz) number of voxels in each dimension
        voxel_size: Size of each cubic voxel in meters
        occupancy: 3D numpy array of occupancy probabilities [0, 1]
        log_odds: 3D numpy array of log-odds representation for Bayesian updates
    """
    origin: np.ndarray
    dimensions: Tuple[int, int, int]
    voxel_size: float
    occupancy: np.ndarray = field(init=False)
    log_odds: np.ndarray = field(init=False)
    
    # Log-odds parameters for Bayesian update
    L_FREE: float = field(default=-0.4, repr=False)  # log-odds update when observed free
    L_OCC: float = field(default=0.85, repr=False)   # log-odds update when observed occupied
    L_MIN: float = field(default=-5.0, repr=False)   # minimum log-odds (clamp) -> prob ≈ 0.007
    L_MAX: float = field(default=5.0, repr=False)    # maximum log-odds (clamp) -> prob ≈ 0.993
    
    def __post_init__(self):
        """Initialize occupancy grid with unknown (0.5) probability."""
        self.origin = np.array(self.origin)
        nx, ny, nz = self.dimensions
        # Initialize with 0.5 probability (unknown)
        self.occupancy = np.ones((nx, ny, nz)) * 0.5
        # Log-odds of 0.5 probability is 0
        self.log_odds = np.zeros((nx, ny, nz))
    
    @property
    def total_voxels(self) -> int:
        """Total number of voxels in the grid."""
        return int(np.prod(self.dimensions))
    
    @property
    def grid_size(self) -> np.ndarray:
        """Physical size of the entire grid in meters."""
        return np.array(self.dimensions) * self.voxel_size
    
    @property
    def grid_center(self) -> np.ndarray:
        """World coordinates of the grid center."""
        return self.origin + self.grid_size / 2
    
    def index_to_world(self, i: int, j: int, k: int) -> np.ndarray:
        """
        Convert voxel indices to world coordinates (voxel center).
        
        Args:
            i, j, k: Voxel indices
            
        Returns:
            (x, y, z) world coordinates of voxel center
        """
        return self.origin + (np.array([i, j, k]) + 0.5) * self.voxel_size
    
    def world_to_index(self, point: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Convert world coordinates to voxel indices.
        
        Args:
            point: (x, y, z) world coordinates
            
        Returns:
            (i, j, k) voxel indices, or None if outside grid
        """
        relative = (point - self.origin) / self.voxel_size
        indices = np.floor(relative).astype(int)
        
        # Check bounds
        if np.any(indices < 0) or np.any(indices >= self.dimensions):
            return None
        
        return tuple(indices)
    
    def get_all_voxel_centers(self) -> np.ndarray:
        """
        Get world coordinates of all voxel centers.
        
        Returns:
            (N, 3) array of voxel center coordinates where N = nx * ny * nz
        """
        nx, ny, nz = self.dimensions
        centers = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    centers.append(self.index_to_world(i, j, k))
        return np.array(centers)
    
    def get_occupancy(self, i: int, j: int, k: int) -> float:
        """Get occupancy probability at voxel (i, j, k)."""
        return self.occupancy[i, j, k]
    
    def set_occupancy(self, i: int, j: int, k: int, prob: float):
        """Set occupancy probability at voxel (i, j, k)."""
        self.occupancy[i, j, k] = np.clip(prob, 0.0, 1.0)
        # Update log-odds to match
        self.log_odds[i, j, k] = self._prob_to_log_odds(self.occupancy[i, j, k])
    
    def _prob_to_log_odds(self, p: float) -> float:
        """Convert probability to log-odds."""
        p = np.clip(p, 1e-6, 1 - 1e-6)  # Avoid log(0)
        return np.log(p / (1 - p))
    
    def _log_odds_to_prob(self, l: float) -> float:
        """Convert log-odds to probability."""
        return 1.0 / (1.0 + np.exp(-l))
    
    def update_voxel(self, i: int, j: int, k: int, observed_occupied: bool):
        """
        Update voxel occupancy using Bayesian log-odds update.
        
        Args:
            i, j, k: Voxel indices
            observed_occupied: True if observation indicates occupied, False if free
        """
        # Log-odds update
        update = self.L_OCC if observed_occupied else self.L_FREE
        self.log_odds[i, j, k] += update
        
        # Clamp to prevent overconfidence
        self.log_odds[i, j, k] = np.clip(self.log_odds[i, j, k], self.L_MIN, self.L_MAX)
        
        # Update probability
        self.occupancy[i, j, k] = self._log_odds_to_prob(self.log_odds[i, j, k])
    
    def is_point_in_frustum(
        self,
        point: np.ndarray,
        camera_pose: RigidTransform,
        fov_y_deg: float,
        aspect_ratio: float,
        near: float,
        far: float,
    ) -> bool:
        """
        Check if a point is inside the camera view frustum.
        
        Args:
            point: (x, y, z) world coordinates
            camera_pose: Camera pose in world frame (X_WC)
            fov_y_deg: Vertical field of view in degrees
            aspect_ratio: Image width / height
            near: Near clipping plane distance
            far: Far clipping plane distance
            
        Returns:
            True if point is inside frustum
        """
        # Transform point to camera frame
        p_world = point.reshape(3, 1)
        R_WC = camera_pose.rotation().matrix()
        p_WC = camera_pose.translation().reshape(3, 1)
        
        # p_cam = R_CW @ (p_world - p_WC) = R_WC.T @ (p_world - p_WC)
        p_cam = R_WC.T @ (p_world - p_WC)
        p_cam = p_cam.flatten()
        
        # Check depth (Z in camera frame)
        z = p_cam[2]
        if z < near or z > far:
            return False
        
        # Check horizontal and vertical bounds
        fov_y = np.deg2rad(fov_y_deg)
        fov_x = 2 * np.arctan(np.tan(fov_y / 2) * aspect_ratio)
        
        half_width = z * np.tan(fov_x / 2)
        half_height = z * np.tan(fov_y / 2)
        
        if abs(p_cam[0]) > half_width or abs(p_cam[1]) > half_height:
            return False
        
        return True
    
    def get_voxels_in_frustum(
        self,
        camera_pose: RigidTransform,
        fov_y_deg: float,
        aspect_ratio: float,
        near: float,
        far: float,
    ) -> list:
        """
        Get indices of all voxels whose centers are inside the camera frustum.
        
        Returns:
            List of (i, j, k) tuples for voxels in frustum
        """
        in_frustum = []
        nx, ny, nz = self.dimensions
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    center = self.index_to_world(i, j, k)
                    if self.is_point_in_frustum(
                        center, camera_pose, fov_y_deg, aspect_ratio, near, far
                    ):
                        in_frustum.append((i, j, k))
        
        return in_frustum
    
    def visualize(
        self,
        meshcat: Meshcat,
        path_prefix: str = "voxel_grid",
        alpha: float = 0.4,
        show_grid_bounds: bool = True,
    ):
        """
        Visualize the voxel grid in Meshcat.
        
        Voxel colors:
        - Red: High occupancy probability (occupied)
        - White/Gray: Unknown (0.5 probability)
        - Green: Low occupancy probability (free)
        
        Args:
            meshcat: Meshcat instance
            path_prefix: Meshcat path prefix for voxel objects
            alpha: Transparency of voxels
            show_grid_bounds: Whether to show grid bounding box
        """
        nx, ny, nz = self.dimensions
        voxel_box = Box(self.voxel_size, self.voxel_size, self.voxel_size)
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    prob = self.occupancy[i, j, k]
                    center = self.index_to_world(i, j, k)
                    
                    # Color based on occupancy:
                    # prob = 1.0 (occupied) -> Red
                    # prob = 0.5 (unknown) -> Gray/White  
                    # prob = 0.0 (free) -> Green
                    color = self._prob_to_color(prob, alpha)
                    
                    voxel_path = f"{path_prefix}/voxel_{i}_{j}_{k}"
                    meshcat.SetObject(voxel_path, voxel_box, color)
                    meshcat.SetTransform(
                        voxel_path,
                        RigidTransform(RotationMatrix(), center)
                    )
        
        # Draw grid bounding box
        if show_grid_bounds:
            self._draw_grid_bounds(meshcat, f"{path_prefix}/bounds")
        
        print(f"  Voxel grid visualized: {nx}x{ny}x{nz} = {self.total_voxels} voxels")
        print(f"    Origin: {self.origin}")
        print(f"    Size: {self.grid_size}m, Voxel size: {self.voxel_size}m")
    
    def _prob_to_color(self, prob: float, alpha: float) -> Rgba:
        """
        Convert occupancy probability to RGBA color.
        
        Color gradient:
        - prob = 0.0 (free) -> Green
        - prob = 0.5 (unknown) -> Yellow  
        - prob = 1.0 (occupied) -> Red
        
        Transparency gradient:
        - prob = 0.0 (certain free) -> Transparent (invisible)
        - prob = 0.5 (uncertain) -> Opaque (visible, needs resolution)
        - prob = 1.0 (certain occupied) -> Translucent red (visible but see-through)
        """
        # Color interpolation: yellow at 0.5 (unknown), red at 1.0 (occupied), green at 0.0 (free)
        if prob < 0.5:
            # Interpolate between green (0.0) and yellow (0.5)
            t = prob * 2.0  # t in [0, 1] for prob in [0, 0.5]
            r = t * 1.0  # 0 -> 1.0 (green to yellow)
            g = 1.0  # Always 1.0 (green/yellow)
            b = (1.0 - t) * 0.2  # 0.2 -> 0 (green to yellow)
        else:
            # Interpolate between yellow (0.5) and red (1.0)
            t = (prob - 0.5) * 2.0  # t in [0, 1] for prob in [0.5, 1.0]
            r = 1.0  # Always 1.0 (yellow/red)
            g = 1.0 - t  # 1.0 -> 0 (yellow to red)
            b = 0.0  # No blue
        
        # Alpha mapping for both free AND occupied confidence:
        # - prob = 0.0 (certain free) -> transparent (invisible)
        # - prob = 0.5 (uncertain) -> opaque (visible, needs resolution)
        # - prob = 1.0 (certain occupied) -> translucent red (visible but see-through)
        
        if prob <= 0.01:
            # Completely transparent for highly confident FREE voxels
            actual_alpha = 0.0
        elif prob >= 0.99:
            # Translucent for highly confident OCCUPIED voxels (can see through)
            actual_alpha = 0.6
        elif prob <= 0.5:
            # Free side: fade from transparent (prob=0) to opaque (prob=0.5)
            actual_alpha = 2.0 * prob
        else:
            # Occupied side: fade from opaque (prob=0.5) to translucent (prob=1.0)
            # Linear: 1.0 at prob=0.5, 0.6 at prob=1.0
            actual_alpha = 1.0 - 0.4 * (prob - 0.5) * 2.0
        
        return Rgba(r, g, b, actual_alpha)
    
    def _draw_grid_bounds(self, meshcat: Meshcat, path: str):
        """Draw the bounding box of the voxel grid."""
        corners = self.origin
        size = self.grid_size
        
        # 8 corners of the bounding box
        c = [
            corners,
            corners + np.array([size[0], 0, 0]),
            corners + np.array([size[0], size[1], 0]),
            corners + np.array([0, size[1], 0]),
            corners + np.array([0, 0, size[2]]),
            corners + np.array([size[0], 0, size[2]]),
            corners + np.array([size[0], size[1], size[2]]),
            corners + np.array([0, size[1], size[2]]),
        ]
        
        # 12 edges of the box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
        ]
        
        starts = []
        ends = []
        for e in edges:
            starts.append(c[e[0]])
            ends.append(c[e[1]])
        
        starts = np.array(starts).T
        ends = np.array(ends).T
        
        meshcat.SetLineSegments(path, starts, ends, 2.0, Rgba(1, 1, 0, 0.8))
    
    def update_visualization(
        self,
        meshcat: Meshcat,
        path_prefix: str = "voxel_grid",
        alpha: float = 0.4,
        voxels_to_update: Optional[list] = None,
    ):
        """
        Update visualization for specific voxels (more efficient than full redraw).
        
        Args:
            meshcat: Meshcat instance
            path_prefix: Meshcat path prefix
            alpha: Transparency
            voxels_to_update: List of (i, j, k) tuples to update, or None for all
        """
        if voxels_to_update is None:
            # Update all voxels
            nx, ny, nz = self.dimensions
            voxels_to_update = [
                (i, j, k) 
                for i in range(nx) 
                for j in range(ny) 
                for k in range(nz)
            ]
        
        voxel_box = Box(self.voxel_size, self.voxel_size, self.voxel_size)
        
        for i, j, k in voxels_to_update:
            prob = self.occupancy[i, j, k]
            center = self.index_to_world(i, j, k)
            color = self._prob_to_color(prob, alpha)
            
            voxel_path = f"{path_prefix}/voxel_{i}_{j}_{k}"
            meshcat.SetObject(voxel_path, voxel_box, color)
            meshcat.SetTransform(
                voxel_path,
                RigidTransform(RotationMatrix(), center)
            )


@dataclass
class ObservationModel:
    """
    Probabilistic observation model for RGBD camera voxel occupancy sensing.
    
    This model computes distance-dependent observation likelihoods for voxel
    occupancy based on camera measurements. The accuracy degrades with distance,
    meaning both false positive and false negative rates increase at longer ranges.
    
    The observation model uses the depth image to check for occlusion - voxels
    behind collision geometry (like shelves) cannot be observed.
    
    The observation model is defined by:
    - P(z=occ | x=occ, d) = true positive rate (decreases with distance)
    - P(z=free | x=free, d) = true negative rate (decreases with distance)
    
    For Bayesian updates, we compute log-odds updates that depend on distance.
    
    Attributes:
        fov_y_deg: Vertical field of view in degrees
        aspect_ratio: Image width / height ratio
        image_width: Image width in pixels
        image_height: Image height in pixels
        near_distance: Minimum sensing range (meters)
        far_distance: Maximum sensing range (meters)
        occlusion_tolerance: Tolerance for occlusion check (meters)
        p_true_pos_near: P(z=occ | x=occ) at near range
        p_true_pos_far: P(z=occ | x=occ) at far range
        p_true_neg_near: P(z=free | x=free) at near range
        p_true_neg_far: P(z=free | x=free) at far range
    """
    fov_y_deg: float = 45.0
    aspect_ratio: float = 640.0 / 480.0
    image_width: int = 640
    image_height: int = 480
    near_distance: float = 0.02
    far_distance: float = 2.0
    occlusion_tolerance: float = 0.05  # 5cm tolerance for occlusion check
    
    # Detection probabilities at near and far range
    # These define how accuracy degrades with distance
    p_true_pos_near: float = 0.95  # P(z=occ | x=occ) at near range
    p_true_pos_far: float = 0.70   # P(z=occ | x=occ) at far range
    p_true_neg_near: float = 0.98  # P(z=free | x=free) at near range  
    p_true_neg_far: float = 0.80   # P(z=free | x=free) at far range
    
    # Update weight: scales the log-odds update to control convergence speed
    # 1.0 = full Bayesian update (fast convergence, 1-2 observations)
    # 0.1 = slow convergence (requires ~20+ observations to become confident)
    update_weight: float = 0.15
    
    def __post_init__(self):
        """Precompute FOV values and camera intrinsics for efficiency."""
        self.fov_y_rad = np.deg2rad(self.fov_y_deg)
        self.fov_x_rad = 2 * np.arctan(np.tan(self.fov_y_rad / 2) * self.aspect_ratio)
        self.half_fov_x = self.fov_x_rad / 2
        self.half_fov_y = self.fov_y_rad / 2
        
        # Compute camera intrinsics (pinhole model)
        # focal length in pixels: f = (height/2) / tan(fov_y/2)
        self.fy = (self.image_height / 2) / np.tan(self.fov_y_rad / 2)
        self.fx = (self.image_width / 2) / np.tan(self.fov_x_rad / 2)
        # Principal point at image center
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
    
    def get_true_positive_rate(self, distance: np.ndarray) -> np.ndarray:
        """
        Get P(z=occupied | x=occupied) as a function of distance.
        
        Linearly interpolates between near and far rates.
        
        Args:
            distance: Array of distances (meters)
            
        Returns:
            Array of true positive rates [0, 1]
        """
        # Normalize distance to [0, 1] range
        t = np.clip((distance - self.near_distance) / (self.far_distance - self.near_distance), 0, 1)
        # Linear interpolation
        return self.p_true_pos_near + t * (self.p_true_pos_far - self.p_true_pos_near)
    
    def get_true_negative_rate(self, distance: np.ndarray) -> np.ndarray:
        """
        Get P(z=free | x=free) as a function of distance.
        
        Linearly interpolates between near and far rates.
        
        Args:
            distance: Array of distances (meters)
            
        Returns:
            Array of true negative rates [0, 1]
        """
        t = np.clip((distance - self.near_distance) / (self.far_distance - self.near_distance), 0, 1)
        return self.p_true_neg_near + t * (self.p_true_neg_far - self.p_true_neg_near)
    
    def get_false_positive_rate(self, distance: np.ndarray) -> np.ndarray:
        """P(z=occupied | x=free) - false alarm rate."""
        return 1.0 - self.get_true_negative_rate(distance)
    
    def get_false_negative_rate(self, distance: np.ndarray) -> np.ndarray:
        """P(z=free | x=occupied) - miss rate."""
        return 1.0 - self.get_true_positive_rate(distance)
    
    def compute_log_odds_update(
        self, 
        distance: np.ndarray, 
        observed_occupied: np.ndarray
    ) -> np.ndarray:
        """
        Compute log-odds update for each observation.
        
        The log-odds update for a Bayesian occupancy filter is:
        - If z=occupied: log(P(z|occ) / P(z|free)) = log(TPR / FPR)
        - If z=free: log(P(z|occ) / P(z|free)) = log(FNR / TNR)
        
        But we want to add this to log-odds, so:
        - If z=occupied: delta_L = log(P(occ|z) / P(free|z)) contribution
        
        Using inverse sensor model:
        - L_update(z=occ) = log(P(occ|z=occ) / (1 - P(occ|z=occ)))
        
        For simplicity, we use the log-likelihood ratio:
        - z=occ: L_update = log(TPR / FPR)
        - z=free: L_update = log(FNR / TNR) = log((1-TPR) / (1-FPR))
        
        Args:
            distance: Array of distances to voxel centers
            observed_occupied: Boolean array, True if depth indicates occupied
            
        Returns:
            Array of log-odds updates (can be positive or negative)
        """
        tpr = self.get_true_positive_rate(distance)
        tnr = self.get_true_negative_rate(distance)
        fpr = 1.0 - tnr  # False positive rate
        fnr = 1.0 - tpr  # False negative rate
        
        # Clamp to avoid log(0)
        eps = 1e-6
        tpr = np.clip(tpr, eps, 1 - eps)
        tnr = np.clip(tnr, eps, 1 - eps)
        fpr = np.clip(fpr, eps, 1 - eps)
        fnr = np.clip(fnr, eps, 1 - eps)
        
        # Compute log-odds updates
        # For z=occupied: log(TPR / FPR) -> positive update (increases belief of occupied)
        # For z=free: log(FNR / TNR) -> negative update (decreases belief of occupied)
        l_occ = np.log(tpr / fpr)
        l_free = np.log(fnr / tnr)
        
        # Select based on observation and apply update weight for gradual convergence
        raw_update = np.where(observed_occupied, l_occ, l_free)
        return raw_update * self.update_weight
    
    def transform_points_to_camera_frame(
        self,
        points_world: np.ndarray,
        camera_pose: RigidTransform,
    ) -> np.ndarray:
        """
        Transform world points to camera frame (vectorized).
        
        Args:
            points_world: (N, 3) array of world coordinates
            camera_pose: Camera pose X_WC (world to camera transform)
            
        Returns:
            (N, 3) array of points in camera frame
        """
        R_WC = camera_pose.rotation().matrix()
        p_WC = camera_pose.translation()
        
        # p_cam = R_CW @ (p_world - p_WC) = R_WC.T @ (p_world - p_WC)
        points_cam = (points_world - p_WC) @ R_WC  # Equivalent to R_WC.T @ (p - p_WC).T
        return points_cam
    
    def project_to_image(
        self,
        points_cam: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D camera-frame points to 2D image coordinates (vectorized).
        
        Uses pinhole camera model: u = fx * X/Z + cx, v = fy * Y/Z + cy
        
        Args:
            points_cam: (N, 3) array of points in camera frame
            
        Returns:
            Tuple of:
            - pixel_coords: (N, 2) array of (u, v) pixel coordinates
            - depths: (N,) array of depths (Z values in camera frame)
        """
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]
        
        # Avoid division by zero
        z_safe = np.maximum(z, 1e-6)
        
        # Project to image plane
        u = self.fx * (x / z_safe) + self.cx
        v = self.fy * (y / z_safe) + self.cy
        
        pixel_coords = np.stack([u, v], axis=1)
        return pixel_coords, z
    
    def check_occlusion(
        self,
        points_cam: np.ndarray,
        depth_image: np.ndarray,
    ) -> np.ndarray:
        """
        Check which points are occluded using the depth image (vectorized).
        
        A point is occluded if the measured depth at its projected pixel
        is less than the point's depth (minus tolerance).
        
        Args:
            points_cam: (N, 3) array of points in camera frame
            depth_image: (H, W) or (H, W, 1) depth image in meters
            
        Returns:
            (N,) boolean array, True if point is NOT occluded (visible)
        """
        # Ensure depth image is 2D
        if depth_image.ndim == 3:
            depth_image = depth_image.squeeze()
        
        # Project points to image coordinates
        pixel_coords, point_depths = self.project_to_image(points_cam)
        
        # Round to nearest pixel
        u = np.round(pixel_coords[:, 0]).astype(int)
        v = np.round(pixel_coords[:, 1]).astype(int)
        
        # Check if pixels are within image bounds
        in_bounds = (
            (u >= 0) & (u < self.image_width) &
            (v >= 0) & (v < self.image_height)
        )
        
        # Initialize all as not visible
        not_occluded = np.zeros(len(points_cam), dtype=bool)
        
        # For points in bounds, check depth
        valid_indices = np.where(in_bounds)[0]
        if len(valid_indices) > 0:
            u_valid = u[valid_indices]
            v_valid = v[valid_indices]
            point_depths_valid = point_depths[valid_indices]
            
            # Get measured depths at pixel locations
            # Note: depth_image is (H, W) so index as [row, col] = [v, u]
            measured_depths = depth_image[v_valid, u_valid]
            
            # Handle invalid depth values (inf, nan, 0)
            valid_depth = np.isfinite(measured_depths) & (measured_depths > 0)
            
            # Point is visible if:
            # 1. Measured depth is invalid (nothing in the way), OR
            # 2. Measured depth is >= point depth - tolerance (point is at or in front of obstacle)
            visible_valid = (
                ~valid_depth |  # No valid depth measurement
                (measured_depths >= point_depths_valid - self.occlusion_tolerance)
            )
            
            not_occluded[valid_indices] = visible_valid
        
        return not_occluded
    
    def get_visible_voxels_vectorized(
        self,
        voxel_grid: VoxelGrid,
        camera_pose: RigidTransform,
        depth_image: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Efficiently find all voxels visible in the camera frustum (vectorized).
        
        If a depth image is provided, also checks for occlusion - voxels behind
        collision geometry (shelves, etc.) are excluded.
        
        Args:
            voxel_grid: VoxelGrid instance
            camera_pose: Camera pose X_WC
            depth_image: Optional (H, W) depth image for occlusion checking
            
        Returns:
            Tuple of:
            - visible_indices: (M, 3) array of (i, j, k) indices for visible voxels
            - visible_centers: (M, 3) array of world coordinates
            - distances: (M,) array of distances from camera to voxel centers
        """
        nx, ny, nz = voxel_grid.dimensions
        
        # Generate all voxel indices
        i_idx, j_idx, k_idx = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
        )
        all_indices = np.stack([i_idx.ravel(), j_idx.ravel(), k_idx.ravel()], axis=1)
        
        # Compute all voxel centers (vectorized)
        centers_world = voxel_grid.origin + (all_indices + 0.5) * voxel_grid.voxel_size
        
        # Transform to camera frame
        centers_cam = self.transform_points_to_camera_frame(centers_world, camera_pose)
        
        # Frustum check (vectorized)
        z = centers_cam[:, 2]  # Depth in camera frame
        x = centers_cam[:, 0]
        y = centers_cam[:, 1]
        
        # Check depth bounds
        in_depth = (z >= self.near_distance) & (z <= self.far_distance)
        
        # Check angular bounds (within FOV)
        half_width = z * np.tan(self.half_fov_x)
        half_height = z * np.tan(self.half_fov_y)
        in_fov = (np.abs(x) <= half_width) & (np.abs(y) <= half_height)
        
        # Combined frustum visibility mask
        in_frustum = in_depth & in_fov
        
        # If depth image is provided, check for occlusion
        if depth_image is not None:
            # Check occlusion for all points in frustum
            not_occluded = self.check_occlusion(centers_cam, depth_image)
            visible_mask = in_frustum & not_occluded
        else:
            visible_mask = in_frustum
        
        # Extract visible voxels
        visible_indices = all_indices[visible_mask]
        visible_centers = centers_world[visible_mask]
        
        # Compute distances (Euclidean from camera origin)
        camera_pos = camera_pose.translation()
        distances = np.linalg.norm(visible_centers - camera_pos, axis=1)
        
        return visible_indices, visible_centers, distances
    
    def generate_observations(
        self,
        voxel_grid: VoxelGrid,
        camera_pose: RigidTransform,
        depth_image: Optional[np.ndarray] = None,
        ground_truth_occupied: Optional[np.ndarray] = None,
        use_perfect_observations: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate probabilistic observations for all visible voxels.
        
        This simulates what the camera would observe, optionally with noise.
        Uses depth image for occlusion checking if provided.
        
        Args:
            voxel_grid: VoxelGrid instance
            camera_pose: Camera pose X_WC
            depth_image: Optional (H, W) depth image for occlusion checking
            ground_truth_occupied: Optional (nx, ny, nz) boolean array of true occupancy.
                                  If None, assumes all cells are free (empty space).
            use_perfect_observations: If True, observations match ground truth perfectly.
                                     If False, apply probabilistic noise model.
        
        Returns:
            Tuple of:
            - visible_indices: (M, 3) array of visible voxel indices
            - observations: (M,) boolean array of observed occupancy
            - log_odds_updates: (M,) array of log-odds updates for each voxel
        """
        # Get visible voxels (with occlusion checking if depth image provided)
        visible_indices, visible_centers, distances = self.get_visible_voxels_vectorized(
            voxel_grid, camera_pose, depth_image
        )
        
        if len(visible_indices) == 0:
            return np.array([]).reshape(0, 3), np.array([]), np.array([])
        
        # Determine ground truth for visible voxels
        if ground_truth_occupied is None:
            # Assume all free (no obstacles)
            gt_occupied = np.zeros(len(visible_indices), dtype=bool)
        else:
            # Extract ground truth for visible voxels
            idx = visible_indices.astype(int)
            gt_occupied = ground_truth_occupied[idx[:, 0], idx[:, 1], idx[:, 2]]
        
        if use_perfect_observations:
            # Perfect observations match ground truth
            observations = gt_occupied.copy()
        else:
            # Apply probabilistic noise model
            tpr = self.get_true_positive_rate(distances)
            tnr = self.get_true_negative_rate(distances)
            
            # For occupied cells: observe occupied with probability TPR
            # For free cells: observe free with probability TNR
            random_vals = np.random.random(len(visible_indices))
            
            observations = np.where(
                gt_occupied,
                random_vals < tpr,  # True occupied -> observe occupied with P=TPR
                random_vals >= tnr,  # True free -> observe occupied with P=FPR=(1-TNR)
            )
        
        # Compute log-odds updates based on observations and distances
        log_odds_updates = self.compute_log_odds_update(distances, observations)
        
        return visible_indices, observations, log_odds_updates
    
    def update_voxel_grid(
        self,
        voxel_grid: VoxelGrid,
        camera_pose: RigidTransform,
        depth_image: Optional[np.ndarray] = None,
        ground_truth_occupied: Optional[np.ndarray] = None,
        use_perfect_observations: bool = True,
        l_min: float = -5.0,
        l_max: float = 5.0,
    ) -> np.ndarray:
        """
        Update voxel grid occupancy based on camera observations.
        
        This is the main interface for updating beliefs based on observations.
        Uses depth image for occlusion checking if provided.
        
        Args:
            voxel_grid: VoxelGrid instance to update (modified in-place)
            camera_pose: Camera pose X_WC
            depth_image: Optional (H, W) depth image for occlusion checking
            ground_truth_occupied: Optional ground truth occupancy grid
            use_perfect_observations: Whether to use perfect or noisy observations
            l_min: Minimum log-odds value (clamping), -5.0 -> prob ≈ 0.007
            l_max: Maximum log-odds value (clamping), +5.0 -> prob ≈ 0.993
            
        Returns:
            visible_indices: (M, 3) array of updated voxel indices
        """
        visible_indices, observations, log_odds_updates = self.generate_observations(
            voxel_grid, camera_pose, depth_image, ground_truth_occupied, use_perfect_observations
        )
        
        if len(visible_indices) == 0:
            return visible_indices
        
        # Apply log-odds updates (vectorized)
        idx = visible_indices.astype(int)
        for n in range(len(idx)):
            i, j, k = idx[n]
            voxel_grid.log_odds[i, j, k] += log_odds_updates[n]
            voxel_grid.log_odds[i, j, k] = np.clip(
                voxel_grid.log_odds[i, j, k], l_min, l_max
            )
            voxel_grid.occupancy[i, j, k] = 1.0 / (1.0 + np.exp(-voxel_grid.log_odds[i, j, k]))
        
        return visible_indices
    
    def generate_observations_from_depth(
        self,
        voxel_grid: VoxelGrid,
        camera_pose: RigidTransform,
        depth_image: np.ndarray,
        surface_thickness: float = 0.03,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate observations for visible voxels using actual depth image.
        
        Instead of using a ground truth occupancy array, this method infers
        whether each voxel is occupied or free based on the measured depth.
        
        A voxel is observed as OCCUPIED if the measured depth at its projected
        pixel is approximately equal to the voxel's depth (± surface_thickness),
        meaning the voxel is at or near the observed surface.
        
        A voxel is observed as FREE if the measured depth is significantly
        greater than the voxel's depth, meaning the observed surface is
        behind the voxel (the voxel is in empty space).
        
        Args:
            voxel_grid: VoxelGrid instance
            camera_pose: Camera pose X_WC
            depth_image: (H, W) depth image in meters
            surface_thickness: Depth tolerance for considering a voxel "at the surface".
                              Should be roughly voxel_size or slightly larger.
        
        Returns:
            Tuple of:
            - visible_indices: (M, 3) array of observable voxel indices
            - observations: (M,) boolean array (True = observed occupied)
            - log_odds_updates: (M,) array of log-odds updates
        """
        # Ensure depth image is 2D
        if depth_image.ndim == 3:
            depth_image = depth_image.squeeze()
        
        # Get all voxels in frustum (occlusion check excludes voxels behind surfaces)
        visible_indices, visible_centers, distances = self.get_visible_voxels_vectorized(
            voxel_grid, camera_pose, depth_image
        )
        
        if len(visible_indices) == 0:
            return np.array([]).reshape(0, 3), np.array([]), np.array([])
        
        # Transform visible voxel centers to camera frame
        points_cam = self.transform_points_to_camera_frame(visible_centers, camera_pose)
        voxel_depths = points_cam[:, 2]  # Z coordinate in camera frame = depth
        
        # Project to image coordinates
        pixel_coords, _ = self.project_to_image(points_cam)
        
        # Get pixel indices (round to nearest pixel)
        u = np.round(pixel_coords[:, 0]).astype(int)
        v = np.round(pixel_coords[:, 1]).astype(int)
        
        # Clamp to image bounds (should already be in bounds from frustum check)
        u = np.clip(u, 0, self.image_width - 1)
        v = np.clip(v, 0, self.image_height - 1)
        
        # Query measured depths at projected pixels
        measured_depths = depth_image[v, u]
        
        # Handle invalid depth values (inf, nan, 0 = no return / too far)
        valid_depth = np.isfinite(measured_depths) & (measured_depths > 0)
        
        # Initialize observations array (default to FREE)
        observations = np.zeros(len(visible_indices), dtype=bool)
        
        # For voxels with valid depth measurements:
        # OCCUPIED if voxel is AT the measured surface
        # FREE if surface is BEHIND voxel (voxel is in empty space in front of the surface)
        
        # Depth difference: positive means surface is BEHIND voxel, negative means IN FRONT
        # depth_diff = measured_depth - voxel_depth
        #   depth_diff > 0: surface behind voxel (voxel in free space)
        #   depth_diff ≈ 0: surface at voxel (voxel occupied)
        #   depth_diff < 0: surface in front of voxel (voxel occluded/behind obstacle)
        depth_diff = measured_depths - voxel_depths
        
        # ASYMMETRIC condition: only mark as occupied if surface is AT the voxel
        # Allow small positive depth_diff for noise, but NOT if surface is significantly behind
        # - depth_diff >= -surface_thickness: surface not too far in front (would mean voxel is behind obstacle)
        # - depth_diff <= surface_thickness * 0.3: surface not significantly behind voxel
        #   (if surface is behind, voxel is in free space looking at back wall/floor)
        surface_at_voxel = (depth_diff >= -surface_thickness) & (depth_diff <= surface_thickness * 0.3)
        
        # Mark voxels at the surface as OCCUPIED
        observations[valid_depth & surface_at_voxel] = True
        
        # Voxels with valid depth where surface is behind them (depth_diff > surface_thickness)
        # are FREE - observations already initialized to False, so nothing to do
        
        # For invalid depth (no return), treat as FREE (no obstacle detected in that direction)
        # observations already False, so nothing to do
        
        # Compute log-odds updates based on observations and distances
        log_odds_updates = self.compute_log_odds_update(distances, observations)
        
        return visible_indices, observations, log_odds_updates

    def update_voxel_grid_from_depth(
        self,
        voxel_grid: VoxelGrid,
        camera_pose: RigidTransform,
        depth_image: np.ndarray,
        surface_thickness: float = 0.03,
        l_min: float = -5.0,
        l_max: float = 5.0,
    ) -> Tuple[np.ndarray, int, int]:
        """
        Update voxel grid occupancy using actual depth image measurements.
        
        This is the realistic observation model that infers occupancy
        from the depth image rather than using simulated ground truth.
        
        Args:
            voxel_grid: VoxelGrid instance to update (modified in-place)
            camera_pose: Camera pose X_WC
            depth_image: (H, W) depth image in meters
            surface_thickness: Depth tolerance for surface detection
            l_min: Minimum log-odds clamp
            l_max: Maximum log-odds clamp
            
        Returns:
            Tuple of:
            - visible_indices: (M, 3) array of updated voxel indices
            - num_observed_occupied: Count of voxels observed as occupied
            - num_observed_free: Count of voxels observed as free
        """
        visible_indices, observations, log_odds_updates = self.generate_observations_from_depth(
            voxel_grid, camera_pose, depth_image, surface_thickness
        )
        
        if len(visible_indices) == 0:
            return visible_indices, 0, 0
        
        # Apply log-odds updates (vectorized)
        idx = visible_indices.astype(int)
        for n in range(len(idx)):
            i, j, k = idx[n]
            voxel_grid.log_odds[i, j, k] += log_odds_updates[n]
            voxel_grid.log_odds[i, j, k] = np.clip(
                voxel_grid.log_odds[i, j, k], l_min, l_max
            )
            voxel_grid.occupancy[i, j, k] = 1.0 / (1.0 + np.exp(-voxel_grid.log_odds[i, j, k]))
        
        num_occupied = int(np.sum(observations))
        num_free = len(observations) - num_occupied
        
        return visible_indices, num_occupied, num_free

    def get_observation_stats(
        self,
        voxel_grid: VoxelGrid,
        camera_pose: RigidTransform,
        depth_image: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Get statistics about what the camera can observe.
        Uses depth image for occlusion checking if provided.
        
        Returns:
            Dictionary with observation statistics
        """
        visible_indices, visible_centers, distances = self.get_visible_voxels_vectorized(
            voxel_grid, camera_pose, depth_image
        )
        
        if len(visible_indices) == 0:
            return {
                "num_visible": 0,
                "total_voxels": voxel_grid.total_voxels,
                "visibility_fraction": 0.0,
                "min_distance": None,
                "max_distance": None,
                "mean_distance": None,
            }
        
        return {
            "num_visible": len(visible_indices),
            "total_voxels": voxel_grid.total_voxels,
            "visibility_fraction": len(visible_indices) / voxel_grid.total_voxels,
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
            "mean_distance": float(np.mean(distances)),
            "min_tpr": float(self.get_true_positive_rate(np.array([np.max(distances)]))[0]),
            "max_tpr": float(self.get_true_positive_rate(np.array([np.min(distances)]))[0]),
            "min_tnr": float(self.get_true_negative_rate(np.array([np.max(distances)]))[0]),
            "max_tnr": float(self.get_true_negative_rate(np.array([np.min(distances)]))[0]),
        }


def depth_to_color_image(
    depth_image: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 2.0,
    colormap: str = "turbo",
    invalid_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Convert a depth image to a color image for visualization.
    
    Args:
        depth_image: (H, W) depth image in meters
        min_depth: Minimum depth for colormap scaling (meters)
        max_depth: Maximum depth for colormap scaling (meters)
        colormap: Colormap name ("turbo", "jet", "viridis", "grayscale")
        invalid_color: RGB color for invalid depth values (inf, nan, 0)
        
    Returns:
        (H, W, 3) uint8 RGB image
    """
    # Ensure 2D
    if depth_image.ndim == 3:
        depth_image = depth_image.squeeze()
    
    H, W = depth_image.shape
    
    # Identify valid depth values
    valid_mask = np.isfinite(depth_image) & (depth_image > 0)
    
    # Normalize depth to [0, 1] range
    depth_normalized = np.zeros_like(depth_image)
    depth_normalized[valid_mask] = np.clip(
        (depth_image[valid_mask] - min_depth) / (max_depth - min_depth),
        0.0, 1.0
    )
    
    # Apply colormap
    if colormap == "grayscale":
        # Grayscale: near = white, far = black
        gray = ((1.0 - depth_normalized) * 255).astype(np.uint8)
        color_image = np.stack([gray, gray, gray], axis=-1)
    elif colormap == "turbo":
        # Turbo colormap (perceptually uniform, good for depth)
        color_image = _apply_turbo_colormap(depth_normalized)
    elif colormap == "jet":
        # Jet colormap
        color_image = _apply_jet_colormap(depth_normalized)
    else:
        # Default to grayscale
        gray = ((1.0 - depth_normalized) * 255).astype(np.uint8)
        color_image = np.stack([gray, gray, gray], axis=-1)
    
    # Set invalid pixels to invalid_color
    color_image[~valid_mask] = invalid_color
    
    return color_image


def _apply_turbo_colormap(normalized: np.ndarray) -> np.ndarray:
    """Apply turbo colormap to normalized [0,1] values."""
    # Simplified turbo colormap approximation
    # Based on Google's Turbo colormap
    r = np.clip(np.where(normalized < 0.5,
                         0.5 + normalized * 3.0,
                         2.5 - normalized * 3.0), 0, 1)
    g = np.clip(np.where(normalized < 0.25,
                         normalized * 4.0,
                         np.where(normalized < 0.75,
                                  1.0,
                                  (1.0 - normalized) * 4.0)), 0, 1)
    b = np.clip(np.where(normalized < 0.5,
                         1.5 - normalized * 3.0,
                         normalized * 3.0 - 1.5), 0, 1)
    
    # Invert so near is warm (red/yellow), far is cool (blue)
    r, b = b, r
    
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _apply_jet_colormap(normalized: np.ndarray) -> np.ndarray:
    """Apply jet colormap to normalized [0,1] values."""
    # Jet colormap: blue -> cyan -> green -> yellow -> red
    r = np.clip(1.5 - np.abs(normalized - 0.75) * 4, 0, 1)
    g = np.clip(1.5 - np.abs(normalized - 0.5) * 4, 0, 1)
    b = np.clip(1.5 - np.abs(normalized - 0.25) * 4, 0, 1)
    
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


class CameraVisualizer:
    """
    Real-time camera feed visualizer using matplotlib.
    Displays depth and RGB images in separate windows that update in real-time.
    
    Note: For macOS, you need:
    1. Install XQuartz: brew install --cask xquartz
    2. Enable X11 forwarding in XQuartz preferences
    3. Restart XQuartz and rebuild the devcontainer
    """
    
    def __init__(self, show_depth: bool = True, show_rgb: bool = True):
        self.show_depth = show_depth
        self.show_rgb = show_rgb
        self.depth_fig = None
        self.depth_ax = None
        self.depth_im = None
        self.rgb_fig = None
        self.rgb_ax = None
        self.rgb_im = None
        self.display_available = True
        
        # Check if display is available
        import os
        display = os.environ.get('DISPLAY')
        if not display:
            print("⚠ WARNING: DISPLAY environment variable not set.")
            print("  Matplotlib windows will not be displayed.")
            print("  For macOS: Install XQuartz and set DISPLAY in devcontainer.json")
            self.display_available = False
            return
        
        try:
            # Try to initialize matplotlib with a non-GUI backend first to test
            # If this fails, we'll catch it and use a fallback
            import matplotlib
            # Force TkAgg backend for better compatibility
            matplotlib.use('TkAgg', force=True)
            
            # Initialize matplotlib in interactive mode
            plt.ion()
            
            if self.show_depth:
                self.depth_fig, self.depth_ax = plt.subplots(1, 1, figsize=(8, 6))
                self.depth_ax.set_title("Depth Camera Feed", fontsize=14)
                self.depth_ax.axis('off')
                self.depth_fig.canvas.manager.set_window_title("Depth Camera")
                plt.tight_layout()
            
            if self.show_rgb:
                self.rgb_fig, self.rgb_ax = plt.subplots(1, 1, figsize=(8, 6))
                self.rgb_ax.set_title("RGB Camera Feed", fontsize=14)
                self.rgb_ax.axis('off')
                self.rgb_fig.canvas.manager.set_window_title("RGB Camera")
                plt.tight_layout()
        except Exception as e:
            print(f"⚠ WARNING: Could not initialize matplotlib display: {e}")
            print("  Camera feeds will not be displayed.")
            print("  Check X11 forwarding configuration.")
            self.display_available = False
    
    def update_depth(self, depth_image: np.ndarray, min_depth: float = 0.1, max_depth: float = 2.0):
        """Update the depth image display."""
        if not self.show_depth or not self.display_available:
            return
        
        # Ensure 2D
        if depth_image.ndim == 3:
            depth_image = depth_image.squeeze()
        
        if self.depth_im is None:
            # First time: create image
            self.depth_im = self.depth_ax.imshow(
                depth_image,
                cmap='turbo',
                vmin=min_depth,
                vmax=max_depth,
                interpolation='nearest'
            )
            self.depth_fig.colorbar(self.depth_im, ax=self.depth_ax, label='Depth (m)')
        else:
            # Update existing image
            self.depth_im.set_data(depth_image)
        
        # Refresh display
        self.depth_fig.canvas.draw()
        self.depth_fig.canvas.flush_events()
    
    def update_rgb(self, rgb_image: np.ndarray):
        """Update the RGB image display."""
        if not self.show_rgb or not self.display_available:
            return
        
        # Ensure correct shape and type
        if rgb_image.ndim == 3:
            if rgb_image.shape[-1] == 4:
                rgb_image = rgb_image[:, :, :3]  # Remove alpha channel
        else:
            return
        
        if rgb_image.dtype != np.uint8:
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            else:
                rgb_image = rgb_image.astype(np.uint8)
        
        if self.rgb_im is None:
            # First time: create image
            self.rgb_im = self.rgb_ax.imshow(rgb_image, interpolation='nearest')
        else:
            # Update existing image
            self.rgb_im.set_data(rgb_image)
        
        # Refresh display
        self.rgb_fig.canvas.draw()
        self.rgb_fig.canvas.flush_events()
    
    def close(self):
        """Close all figure windows."""
        if self.depth_fig:
            plt.close(self.depth_fig)
        if self.rgb_fig:
            plt.close(self.rgb_fig)
        plt.ioff()


def visualize_camera_frustum(
    meshcat: Meshcat,
    plant,
    plant_context,
    camera_body_name: str = "base",
    camera_model_name: str = "wrist_camera",
    fov_y_deg: float = 45.0,
    aspect_ratio: float = 640.0 / 480.0,
    near_distance: float = 0.01,
    far_distance: float = 1.0,
    frustum_name: str = "camera_frustum",
    color: Rgba = Rgba(0.2, 0.6, 1.0, 0.7),
    verbose: bool = True,
):
    """
    Visualize the view frustum of an RGBD camera in Meshcat.
    
    Args:
        meshcat: Meshcat instance
        plant: MultibodyPlant containing the camera body
        plant_context: Context for the plant
        camera_body_name: Name of the camera body frame
        camera_model_name: Name of the camera model instance
        fov_y_deg: Vertical field of view in degrees
        aspect_ratio: Width / Height of the image
        near_distance: Near clipping plane distance
        far_distance: Far clipping plane distance (for visualization)
        frustum_name: Name for the meshcat path
        color: RGBA color for the frustum lines
    """
    # Get camera body pose in world frame
    camera_model = plant.GetModelInstanceByName(camera_model_name)
    camera_body = plant.GetBodyByName(camera_body_name, camera_model)
    X_WC = plant.EvalBodyPoseInWorld(plant_context, camera_body)
    
    # Calculate frustum corners
    fov_y = np.deg2rad(fov_y_deg)
    fov_x = 2 * np.arctan(np.tan(fov_y / 2) * aspect_ratio)
    
    # Half-angles
    half_fov_x = fov_x / 2
    half_fov_y = fov_y / 2
    
    # Near plane corners (in camera frame, looking down +Z)
    # Drake camera convention: +Z is optical axis (looking forward)
    near_half_h = near_distance * np.tan(half_fov_y)
    near_half_w = near_distance * np.tan(half_fov_x)
    
    far_half_h = far_distance * np.tan(half_fov_y)
    far_half_w = far_distance * np.tan(half_fov_x)
    
    # Corners in camera frame (X-right, Y-down, Z-forward for typical camera)
    # But we'll adjust based on how the camera is mounted
    near_corners_cam = np.array([
        [-near_half_w, -near_half_h, near_distance],  # top-left
        [ near_half_w, -near_half_h, near_distance],  # top-right
        [ near_half_w,  near_half_h, near_distance],  # bottom-right
        [-near_half_w,  near_half_h, near_distance],  # bottom-left
    ]).T  # Shape: (3, 4)
    
    far_corners_cam = np.array([
        [-far_half_w, -far_half_h, far_distance],  # top-left
        [ far_half_w, -far_half_h, far_distance],  # top-right
        [ far_half_w,  far_half_h, far_distance],  # bottom-right
        [-far_half_w,  far_half_h, far_distance],  # bottom-left
    ]).T  # Shape: (3, 4)
    
    # Transform to world frame
    R_WC = X_WC.rotation().matrix()
    p_WC = X_WC.translation().reshape(3, 1)
    
    near_corners_world = R_WC @ near_corners_cam + p_WC
    far_corners_world = R_WC @ far_corners_cam + p_WC
    origin_world = p_WC
    
    # Build line segments for the frustum
    # Lines from origin to far corners (4 lines)
    # Lines connecting far corners (4 lines for the rectangle)
    # Lines connecting near corners (4 lines for the rectangle)
    
    starts = []
    ends = []
    
    # Lines from origin to far corners
    for i in range(4):
        starts.append(origin_world[:, 0])
        ends.append(far_corners_world[:, i])
    
    # Near plane rectangle
    for i in range(4):
        starts.append(near_corners_world[:, i])
        ends.append(near_corners_world[:, (i + 1) % 4])
    
    # Far plane rectangle
    for i in range(4):
        starts.append(far_corners_world[:, i])
        ends.append(far_corners_world[:, (i + 1) % 4])
    
    # Convert to numpy arrays
    starts = np.array(starts).T  # Shape: (3, N)
    ends = np.array(ends).T      # Shape: (3, N)
    
    # Draw the frustum lines
    meshcat.SetLineSegments(
        frustum_name,
        starts,
        ends,
        2.0,  # line width
        color,
    )
    
    if verbose:
        print(f"  Camera frustum visualized: FOV={fov_y_deg}°, range=[{near_distance}, {far_distance}]m")


def main():
    print("=" * 60)
    print("MIT 6.4212 - Observation Model Testing")
    print("Robot Scenario Visualization with Wrist Camera")
    print("=" * 60)

    # Load configuration
    config = load_rrbt_config()
    print("Loaded Configuration:")
    print(f"    > Physics: Q_scale={config.physics.process_noise_scale}")
    print(
        f"    > Planner: MaxUncert={config.planner.max_uncertainty}, LightBias={config.planner.prob_sample_light}"
    )
    print()

    # Start Meshcat
    try:
        params = MeshcatParams()
        params.port = 7000
        meshcat = Meshcat(params=params)
        print("✓ Meshcat started on http://localhost:7000")
    except RuntimeError as e:
        print("\n✗ ERROR: Could not start Meshcat on port 7000")
        print(f"  {e}")
        print("\n  Port 7000 is likely already in use.")
        print(
            "  Please stop any other Meshcat servers or Python processes using that port."
        )
        raise

    # Load the observation modeling scenario (with wrist camera enabled)
    scenario_path = Path(__file__).parent / "config" / "obs_modeling_scenario.yaml"
    print(f"Loading scenario from: {scenario_path}")

    with open(scenario_path, "r") as f:
        scenario = LoadScenario(data=f.read())

    # ====== GOAL POSE CONFIGURATION ======
    # Option 1: Use joint angles from config file
    # Option 2: Use a world-frame transform (X_WG) and solve IK
    #
    # Set USE_TRANSFORM_GOAL = True to use a world-frame transform
    # Set USE_TRANSFORM_GOAL = False to use joint angles from config
    
    USE_TRANSFORM_GOAL = True  # <-- Change this to switch modes
    
    q_home = config.simulation.q_home
    
    # Compute what the config q_goal corresponds to in world frame
    # This helps us understand what transform to use
    print("\n" + "=" * 60)
    print("COMPUTING CONFIG Q_GOAL WORLD TRANSFORM")
    print("=" * 60)
    temp_builder_for_config = DiagramBuilder()
    temp_station_for_config = temp_builder_for_config.AddSystem(MakeHardwareStation(scenario=scenario, meshcat=None))
    temp_diagram_for_config = temp_builder_for_config.Build()
    temp_context_for_config = temp_diagram_for_config.CreateDefaultContext()
    temp_plant_for_config = temp_station_for_config.GetSubsystemByName("plant")
    
    config_q_goal = np.array(config.simulation.q_goal)
    temp_plant_context_for_config = temp_plant_for_config.CreateDefaultContext()
    iiwa_model_for_config = temp_plant_for_config.GetModelInstanceByName("iiwa")
    temp_plant_for_config.SetPositions(temp_plant_context_for_config, iiwa_model_for_config, config_q_goal)
    wsg_body_for_config = temp_plant_for_config.GetBodyByName("body", temp_plant_for_config.GetModelInstanceByName("wsg"))
    X_WG_config_goal = temp_plant_for_config.EvalBodyPoseInWorld(temp_plant_context_for_config, wsg_body_for_config)
    
    print(f"Config q_goal: {config_q_goal}")
    print(f"Corresponding world transform:")
    print(f"  Position (x, y, z): {X_WG_config_goal.translation()}")
    print(f"  Rotation matrix:\n{X_WG_config_goal.rotation().matrix()}")
    print("=" * 60)
    
    # Clean up
    del temp_builder_for_config, temp_station_for_config, temp_diagram_for_config, temp_context_for_config, temp_plant_for_config
    
    if USE_TRANSFORM_GOAL:
        # ====== Define goal as a world-frame transform ======
        # X_WG_goal is the desired pose of the gripper body frame in world coordinates
        # The gripper frame has:
        #   - Z pointing forward (along fingers)
        #   - X pointing to the right (between fingers)
        #   - Y pointing up (perpendicular to palm)
        
        # ====== Translation Offset ======
        # Apply a translation offset to the config goal position
        # Set this to [0, 0, 0] to use the exact config goal position
        # Or modify to shift the goal relative to the config position
        # Example: [0.1, 0.0, 0.05] moves 10cm in X, 5cm in Z
        translation_offset = np.array([-0.15, 0.0, -0.1])  # (dx, dy, dz) in meters
        
        # Use the config q_goal transform (computed above) with optional offset
        goal_position = X_WG_config_goal.translation() + translation_offset
        goal_rotation = X_WG_config_goal.rotation()
        
        # You can also override completely by uncommenting these:
        # goal_position = np.array([0.4, 0.0, 0.5])  # (x, y, z) in meters
        # goal_rotation = RotationMatrix.MakeXRotation(np.pi)  # Example rotation
        
        X_WG_goal = RigidTransform(goal_rotation, goal_position)
        
        print("\n" + "=" * 60)
        print("GOAL POSE (World Frame Transform)")
        print("=" * 60)
        print(f"  Config goal position: {X_WG_config_goal.translation()}")
        print(f"  Translation offset: {translation_offset}")
        print(f"  Final position (x, y, z): {goal_position}")
        print(f"  Rotation matrix:\n{goal_rotation.matrix()}")
        
        # Create a temporary plant for IK (we need this before building the main diagram)
        temp_builder = DiagramBuilder()
        temp_station = temp_builder.AddSystem(MakeHardwareStation(scenario=scenario, meshcat=None))
        temp_diagram = temp_builder.Build()
        temp_context = temp_diagram.CreateDefaultContext()
        temp_plant = temp_station.GetSubsystemByName("plant")
        
        print("\nSolving IK for goal pose...")
        print(f"  Target position: {goal_position}")
        print(f"  Target rotation:\n{goal_rotation.matrix()}")
        try:
            q_goal = solve_ik_for_pose(
                plant=temp_plant,
                X_WG_target=X_WG_goal,
                q_nominal=tuple(q_home),
                theta_bound=0.05,  # Orientation tolerance (radians)
                pos_tol=0.01,      # Position tolerance (meters)
            )
            print(f"✓ IK Success!")
            print(f"  q_goal: {np.round(q_goal, 3)}")
            
            # Verify the IK solution actually achieves the desired pose
            temp_plant_context = temp_plant.CreateDefaultContext()
            iiwa_model = temp_plant.GetModelInstanceByName("iiwa")
            temp_plant.SetPositions(temp_plant_context, iiwa_model, q_goal)
            wsg_body = temp_plant.GetBodyByName("body", temp_plant.GetModelInstanceByName("wsg"))
            X_WG_achieved = temp_plant.EvalBodyPoseInWorld(temp_plant_context, wsg_body)
            
            position_error = np.linalg.norm(X_WG_achieved.translation() - X_WG_goal.translation())
            print(f"  Desired position: {X_WG_goal.translation()}")
            print(f"  Achieved position: {X_WG_achieved.translation()}")
            print(f"  Position error: {position_error:.4f}m")
            
            if position_error > 0.02:  # More than 2cm error
                print(f"  ⚠ Warning: Large position error! IK may not have converged properly.")
            
            # Store the desired goal for visualization
            X_WG_desired = X_WG_goal
            
        except RuntimeError as e:
            print(f"✗ IK Failed: {e}")
            print("  Falling back to config q_goal")
            print(f"  This usually means the target pose is unreachable or the IK solver")
            print(f"  couldn't find a solution. Try adjusting the goal position or rotation.")
            q_goal = config.simulation.q_goal
            X_WG_desired = None  # No desired pose to visualize
        
        # Clean up temporary resources
        del temp_builder, temp_station, temp_diagram, temp_context, temp_plant
        
    else:
        # ====== Use joint angles from config ======
        q_goal = config.simulation.q_goal
        X_WG_desired = None  # No desired transform when using config
        print("\n" + "=" * 60)
        print("GOAL POSE (Joint Angles from Config)")
        print("=" * 60)
    
    print("=" * 60)

    # ====== RRT Planning ======
    print("\n" + "=" * 60)
    print("RRT PLANNING")
    print("=" * 60)
    print(f"  Start: {q_home}")
    print(f"  Goal: {q_goal}")
    
    # Create IiwaProblem for RRT planning (uses its own internal collision checker)
    rrt_problem = IiwaProblem(
        q_start=q_home,
        q_goal=q_goal,
        gripper_setpoint=0.1,
        meshcat=meshcat,
        is_visualizing=False,  # Don't visualize during planning for speed
    )
    
    # Run RRT planning
    print("\nRunning RRT planning...")
    rrt_path, rrt_iterations = rrt_planning(
        rrt_problem,
        max_iterations=1000,
        prob_sample_q_goal=0.05,
    )
    
    if rrt_path:
        print(f"✓ RRT found path in {rrt_iterations} iterations")
        print(f"  Path length: {len(rrt_path)} waypoints")
        # Convert path to trajectory for execution
        TIME_PER_SEGMENT = 0.1  # Time per waypoint in seconds
        trajectory = path_to_trajectory(rrt_path, time_per_segment=TIME_PER_SEGMENT)
        trajectory_duration = trajectory.end_time()
        print(f"  Trajectory duration: {trajectory_duration:.1f} seconds")
    else:
        print(f"✗ RRT failed to find path after {rrt_iterations} iterations")
        print("  Robot will stay at home position.")
        rrt_path = None
        trajectory = None
        trajectory_duration = 0.0
    
    print("=" * 60)

    # Build the diagram
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario=scenario, meshcat=meshcat))

    plant = station.GetSubsystemByName("plant")

    # ====== Perception system ======
    perception_sys = builder.AddSystem(
        LightDarkRegionSystem(
            plant=plant,
            light_region_center=config.simulation.light_center,
            light_region_size=config.simulation.light_size,
            sigma_light=np.sqrt(float(config.physics.meas_noise_light)),
            sigma_dark=np.sqrt(float(config.physics.meas_noise_dark)),
        )
    )

    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        perception_sys.GetInputPort("iiwa.position"),
    )

    # ====== Robot Position Source ======
    # If RRT found a path, use TrajectorySource to execute it
    # Otherwise, use ConstantVectorSource at home position
    if trajectory is not None:
        iiwa_position_source = builder.AddSystem(TrajectorySource(trajectory))
        iiwa_position_source.set_name("PlannedTrajectorySource")
        print("✓ Using TrajectorySource for robot motion")
    else:
        iiwa_position_source = builder.AddSystem(ConstantVectorSource(q_home))
        print("✓ Using ConstantVectorSource (robot stays at home)")
    
    builder.Connect(
        iiwa_position_source.get_output_port(), station.GetInputPort("iiwa.position")
    )

    wsg_position_source = builder.AddSystem(ConstantVectorSource([0.1]))
    builder.Connect(
        wsg_position_source.get_output_port(), station.GetInputPort("wsg.position")
    )

    # Build diagram and create simulator
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    
    # Check if plant is in continuous mode (required for physics)
    time_step = plant.time_step()
    if time_step > 0:
        print(f"⚠ Warning: Plant is in discrete mode (time_step={time_step}).")
        print("  Free-floating bodies may not experience physics properly.")
        print("  Consider using a continuous plant for full physics simulation.")
    else:
        print("✓ Plant is in continuous mode - physics should work")
    
    # Ensure physics is enabled for free-floating bodies
    # Get the plant context from the simulator
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)
    
    # Check if the target object is a free body and set its pose explicitly
    try:
        target_model = plant.GetModelInstanceByName("target_object")
        target_body = plant.GetBodyByName("base_link_mustard", target_model)
        
        # Verify it's a free body
        if plant.IsBodyFreeFloating(target_body):
            print("✓ Target object is a free-floating body")
            
            # Set the initial pose of the free body in the context
            initial_pose = RigidTransform(
                RotationMatrix.MakeXRotation(-np.pi/2) @ RotationMatrix.MakeZRotation(-np.pi/2),
                [0.1, 0, 0.45]
            )
            plant.SetFreeBodyPose(plant_context, target_body, initial_pose)
            
            # Set velocities to zero
            zero_velocity = SpatialVelocity()
            plant.SetFreeBodySpatialVelocity(plant_context, target_body, zero_velocity)
            
            print("✓ Configured free-floating target object for physics simulation")
        else:
            print("⚠ Warning: Target object is not a free-floating body")
    except Exception as e:
        print(f"⚠ Warning: Could not configure free body: {e}")

    # Visualize light region indicator
    meshcat.SetObject(
        "light_region_indicator",
        Box(*config.simulation.light_size),
        Rgba(0, 1, 0, 0.3),  # Green, 0.3 Alpha
    )
    meshcat.SetTransform(
        "light_region_indicator",
        RigidTransform(RotationMatrix(), config.simulation.light_center),
    )

    # Visualize goal pose
    iiwa = plant.GetModelInstanceByName("iiwa")
    goal_plant_context = plant.CreateDefaultContext()
    plant.SetPositions(goal_plant_context, iiwa, q_goal)

    wsg_body = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))
    X_Goal_achieved = plant.EvalBodyPoseInWorld(goal_plant_context, wsg_body)

    # Visualize achieved goal pose (where robot will actually go)
    AddMeshcatTriad(meshcat, "goal_pose_achieved", length=0.2, radius=0.005)
    meshcat.SetTransform("goal_pose_achieved", X_Goal_achieved)
    # Add green sphere to mark achieved position
    meshcat.SetObject("goal_pose_achieved/sphere", Box(0.02, 0.02, 0.02), Rgba(0, 1, 0, 0.8))
    
    # If using transform mode, also visualize desired pose
    if USE_TRANSFORM_GOAL and X_WG_desired is not None:
        AddMeshcatTriad(meshcat, "goal_pose_desired", length=0.15, radius=0.003)
        meshcat.SetTransform("goal_pose_desired", X_WG_desired)
        # Add red sphere to mark desired position
        meshcat.SetObject("goal_pose_desired/sphere", Box(0.02, 0.02, 0.02), Rgba(1, 0, 0, 0.8))
        print(f"\nGoal visualization:")
        print(f"  Red sphere/triad = Desired pose (from transform)")
        print(f"  Green sphere/triad = Achieved pose (from IK solution)")

    # Advance simulation briefly to initialize
    print("\nInitializing simulation...")
    simulator.AdvanceTo(0.01)

    # Get the context after simulation to get current robot pose
    context = simulator.get_context()
    plant_sim_context = plant.GetMyContextFromRoot(context)

    # Visualize the wrist camera frustum
    print("\nVisualizing camera frustum...")
    visualize_camera_frustum(
        meshcat=meshcat,
        plant=plant,
        plant_context=plant_sim_context,
        camera_body_name="base",
        camera_model_name="wrist_camera",
        fov_y_deg=45.0,  # Typical RGBD camera vertical FOV
        aspect_ratio=640.0 / 480.0,
        near_distance=0.02,
        far_distance=2.0,  # Visualize up to 0.8m for clarity
        frustum_name="wrist_camera_frustum",
        color=Rgba(0.2, 0.6, 1.0, 0.8),  # Light blue
    )

    # Also add a triad at the camera frame for reference
    camera_model = plant.GetModelInstanceByName("wrist_camera")
    camera_body = plant.GetBodyByName("base", camera_model)
    X_WCamera = plant.EvalBodyPoseInWorld(plant_sim_context, camera_body)
    AddMeshcatTriad(meshcat, "camera_frame", length=0.1, radius=0.003)
    meshcat.SetTransform("camera_frame", X_WCamera)
    
    # ====== Voxel Grid Configuration ======
    # Edit these values to change the voxel grid position and size
    # Origin is in world frame coordinates (x, y, z) in meters
    # Kuka base is at x=-0.5, shelf is at x=0.5
    VOXEL_GRID_ORIGIN = np.array([0.0, -0.24, 0.0])  # World frame (x, y, z) in meters
    # VOXEL_GRID_DIMENSIONS = (16, 12, 12)  # (nx, ny, nz) number of voxels
    VOXEL_GRID_DIMENSIONS = (5, 10, 4)  # (nx, ny, nz) number of voxels
    VOXEL_SIZE = 0.0568  # Size of each cubic voxel in meters
    
    # Get shelf_lower_origin frame transform from the plant
    # This frame is defined in shelves.sdf and represents the top surface center of shelf_lower
    shelves_model = plant.GetModelInstanceByName("shelves")
    shelf_lower_origin_frame = plant.GetFrameByName("shelf_lower_origin", shelves_model)
    world_frame = plant.world_frame()
    X_WShelfLowerOrigin = plant.CalcRelativeTransform(
        plant_sim_context, world_frame, shelf_lower_origin_frame
    )
        
    AddMeshcatTriad(meshcat, "voxel_grid_shelf_lower_origin", length=0.1, radius=0.003)
    meshcat.SetTransform("voxel_grid_shelf_lower_origin", X_WShelfLowerOrigin)
    
    # ====== Create and visualize voxel grid ======
    print("\nCreating voxel grid for occupancy mapping...")
    print(f"  Origin (world frame): {VOXEL_GRID_ORIGIN}")
    print(f"  Dimensions: {VOXEL_GRID_DIMENSIONS}")
    print(f"  Voxel size: {VOXEL_SIZE}m")
    
    voxel_grid = VoxelGrid(
        origin=X_WShelfLowerOrigin.translation(),
        dimensions=VOXEL_GRID_DIMENSIONS,
        voxel_size=VOXEL_SIZE,
    )
    
    # Visualize the voxel grid (initially all red/unknown)
    voxel_grid.visualize(
        meshcat=meshcat,
        path_prefix="voxel_grid",
        alpha=0.1,  # Translucent
        show_grid_bounds=True,
    )

    # ====== Observation Model Configuration ======
    # Configure the probabilistic observation model
    # These parameters define how accuracy degrades with distance
    obs_model = ObservationModel(
        fov_y_deg=45.0,
        aspect_ratio=640.0 / 480.0,
        near_distance=0.02,
        far_distance=2.0,
        # Detection rates at near range (high accuracy)
        p_true_pos_near=0.95,  # P(z=occ | x=occ) - 95% detection at close range
        p_true_neg_near=0.98,  # P(z=free | x=free) - 98% correct rejection at close range
        # Detection rates at far range (degraded accuracy)
        p_true_pos_far=0.70,   # P(z=occ | x=occ) - 70% detection at max range
        p_true_neg_far=0.80,   # P(z=free | x=free) - 80% correct rejection at max range
    )
    
    print("\nObservation Model configured:")
    print(f"  FOV: {obs_model.fov_y_deg}° (vertical)")
    print(f"  Image size: {obs_model.image_width}x{obs_model.image_height}")
    print(f"  Range: [{obs_model.near_distance}, {obs_model.far_distance}]m")
    print(f"  Occlusion tolerance: {obs_model.occlusion_tolerance}m")
    print(f"  True Positive Rate: {obs_model.p_true_pos_near:.0%} (near) -> {obs_model.p_true_pos_far:.0%} (far)")
    print(f"  True Negative Rate: {obs_model.p_true_neg_near:.0%} (near) -> {obs_model.p_true_neg_far:.0%} (far)")
    print(f"  False Positive Rate: {1-obs_model.p_true_neg_near:.0%} (near) -> {1-obs_model.p_true_neg_far:.0%} (far)")
    print(f"  False Negative Rate: {1-obs_model.p_true_pos_near:.0%} (near) -> {1-obs_model.p_true_pos_far:.0%} (far)")
    
    # Get initial depth image for occlusion-aware stats
    init_depth_port = station.GetOutputPort("wrist_camera_sensor.depth_image")
    init_context = simulator.get_context()
    init_station_context = station.GetMyContextFromRoot(init_context)
    init_depth_obj = init_depth_port.Eval(init_station_context)
    init_depth_image = np.array(init_depth_obj.data, copy=True)
    if init_depth_image.ndim == 1:
        init_depth_image = init_depth_image.reshape(obs_model.image_height, obs_model.image_width)
    elif init_depth_image.ndim == 3:
        init_depth_image = init_depth_image.squeeze()
    
    # Get observation statistics for current camera pose (with occlusion)
    obs_stats = obs_model.get_observation_stats(voxel_grid, X_WCamera, init_depth_image)
    print(f"\nCurrent observation statistics (with occlusion checking):")
    print(f"  Visible voxels: {obs_stats['num_visible']}/{obs_stats['total_voxels']} ({obs_stats['visibility_fraction']:.1%})")
    if obs_stats['num_visible'] > 0:
        print(f"  Distance range: [{obs_stats['min_distance']:.2f}, {obs_stats['max_distance']:.2f}]m")
        print(f"  TPR range: [{obs_stats['min_tpr']:.1%}, {obs_stats['max_tpr']:.1%}]")
        print(f"  TNR range: [{obs_stats['min_tnr']:.1%}, {obs_stats['max_tnr']:.1%}]")

    # ====== Observation Configuration ======
    # Using realistic depth-based observations: occupancy is inferred from
    # the actual depth image, not from a simulated ground truth.
    # Voxels at measured surfaces will be marked OCCUPIED (red, translucent)
    # Voxels in front of surfaces will be marked FREE (transparent)
    
    # Surface thickness determines how close a voxel must be to the measured
    # depth to be considered "at the surface" (occupied)
    SURFACE_THICKNESS = voxel_grid.voxel_size * 1.5  # Slightly larger than voxel size
    
    print("\n" + "=" * 60)
    print("SCENARIO SETUP COMPLETE")
    print("=" * 60)
    print(f"  Home position: {q_home}")
    print(f"  Goal position: {q_goal}")
    print(f"  Light region center: {config.simulation.light_center}")
    print(f"  Light region size: {config.simulation.light_size}")
    print(f"  Voxel grid: {voxel_grid.dimensions} @ {voxel_grid.voxel_size}m resolution")
    print(f"  Observation mode: DEPTH-BASED (realistic)")
    print(f"  Surface thickness: {SURFACE_THICKNESS:.3f}m")
    print("=" * 60)

    print("\n✓ Visualization ready!")
    print("  Open http://localhost:7000 in your browser to view the scenario.")
    print("  The wrist camera is mounted on the robot's end-effector.")
    print("  Camera frustum is shown in light blue.")
    print("  Camera feeds will open in separate matplotlib windows.")
    print("  Voxel grid will update based on DEPTH measurements:")
    print("    - Yellow (0.5) = unknown/uncertain")
    print("    - Transparent = confident FREE (empty space)")
    print("    - Translucent Red = confident OCCUPIED (surface detected)")
    print("\n" + "=" * 60)
    if trajectory is not None:
        print("EXECUTING RRT TRAJECTORY WITH OBSERVATION LOOP")
        print(f"  Trajectory duration: {trajectory_duration:.1f}s")
    else:
        print("STARTING DEPTH-BASED OBSERVATION LOOP")
    print("=" * 60)
    print("  Observations inferred from actual depth image...")
    print("  Press Ctrl+C to exit.\n")

    # ====== Dynamic Observation Loop ======
    # Execute trajectory while observing and updating voxel beliefs
    observation_count = 0
    sim_time_step = 0.1  # Simulation time step in seconds
    
    # Get image output ports from station
    depth_port = station.GetOutputPort("wrist_camera_sensor.depth_image")
    
    # Try to get RGB image port (may not exist depending on camera config)
    try:
        rgb_port = station.GetOutputPort("wrist_camera_sensor.rgb_image")
        has_rgb = True
        print("  RGB camera stream available.")
    except RuntimeError:
        rgb_port = None
        has_rgb = False
        print("  RGB camera stream not available (depth only).")
    
    # Initialize camera visualizer
    camera_visualizer = CameraVisualizer(show_depth=True, show_rgb=has_rgb)
    print("  Camera visualizer initialized - matplotlib windows will open.")
    
    # Track trajectory execution state
    trajectory_complete = False
    
    try:
        while True:
            # Get current simulation time
            current_time = simulator.get_context().get_time()
            
            # Check if trajectory is complete
            if trajectory is not None and not trajectory_complete:
                if current_time >= trajectory_duration:
                    trajectory_complete = True
                    print(f"\n✓ Trajectory complete at t={current_time:.1f}s")
                    print("  Robot is now at goal position.")
                    print("  Continuing observations... (Press Ctrl+C to exit)")
            
            # Get current context and camera pose from simulation
            context = simulator.get_context()
            station_context = station.GetMyContextFromRoot(context)
            plant_sim_context = plant.GetMyContextFromRoot(context)
            X_WCamera = plant.EvalBodyPoseInWorld(plant_sim_context, camera_body)
            
            # Update camera frustum visualization as robot moves
            visualize_camera_frustum(
                meshcat=meshcat,
                plant=plant,
                plant_context=plant_sim_context,
                camera_body_name="base",
                camera_model_name="wrist_camera",
                fov_y_deg=45.0,
                aspect_ratio=640.0 / 480.0,
                near_distance=0.02,
                far_distance=2.0,
                frustum_name="wrist_camera_frustum",
                color=Rgba(0.2, 0.6, 1.0, 0.8),
                verbose=False,  # Don't print every iteration
            )
            
            # Update camera frame triad as robot moves
            meshcat.SetTransform("camera_frame", X_WCamera)
            
            # Get depth image from camera
            depth_image_obj = depth_port.Eval(station_context)
            depth_image = np.array(depth_image_obj.data, copy=True)
            # Reshape to (H, W) if needed - Drake depth images are typically (H, W, 1)
            if depth_image.ndim == 1:
                depth_image = depth_image.reshape(obs_model.image_height, obs_model.image_width)
            elif depth_image.ndim == 3:
                depth_image = depth_image.squeeze()
            
            # Update depth visualization
            camera_visualizer.update_depth(depth_image, min_depth=0.1, max_depth=2.0)
            
            # Get and visualize RGB image if available
            if has_rgb:
                rgb_image_obj = rgb_port.Eval(station_context)
                rgb_image = np.array(rgb_image_obj.data, copy=True)
                # Reshape if needed - Drake RGB images are typically (H, W, 4) RGBA
                if rgb_image.ndim == 1:
                    rgb_image = rgb_image.reshape(obs_model.image_height, obs_model.image_width, -1)
                # Convert RGBA to RGB if needed
                if rgb_image.shape[-1] == 4:
                    rgb_image = rgb_image[:, :, :3]
                camera_visualizer.update_rgb(rgb_image)
            
            # Update voxel grid using DEPTH-BASED observations (realistic)
            # Occupancy is inferred from the actual depth image, not ground truth
            visible_indices, num_occ, num_free = obs_model.update_voxel_grid_from_depth(
                voxel_grid,
                X_WCamera,
                depth_image=depth_image,
                surface_thickness=SURFACE_THICKNESS,
            )
            
            # Update visualization for observed voxels
            if len(visible_indices) > 0:
                # Convert to list of tuples for visualization update
                voxels_to_update = [tuple(idx) for idx in visible_indices]
                voxel_grid.update_visualization(
                    meshcat=meshcat,
                    path_prefix="voxel_grid",
                    alpha=0.3,  # Base alpha (actual alpha computed by _prob_to_color)
                    voxels_to_update=voxels_to_update,
                )
            
            observation_count += 1
            
            # Advance simulation to move the robot along trajectory
            next_time = current_time + sim_time_step
            simulator.AdvanceTo(next_time)
            
            # Print progress every 10 observations
            if observation_count % 10 == 0:
                # Compute statistics
                mean_occupancy = np.mean(voxel_grid.occupancy)
                num_confident_free = np.sum(voxel_grid.occupancy < 0.2)
                num_confident_occupied = np.sum(voxel_grid.occupancy > 0.8)
                num_unknown = voxel_grid.total_voxels - num_confident_free - num_confident_occupied
                
                # Show trajectory progress if executing
                if trajectory is not None and not trajectory_complete:
                    progress = min(100.0, (current_time / trajectory_duration) * 100)
                    print(f"[t={current_time:5.1f}s/{trajectory_duration:.0f}s ({progress:5.1f}%)] "
                          f"Obs #{observation_count:3d} | "
                          f"Grid: free={num_confident_free}, occ={num_confident_occupied}, unk={num_unknown}")
                else:
                    print(f"[t={current_time:5.1f}s] Obs #{observation_count:3d} | "
                          f"Visible: {len(visible_indices):3d} (occ={num_occ:2d}, free={num_free:3d}) | "
                          f"Grid: free={num_confident_free}, occ={num_confident_occupied}, unk={num_unknown}")
            
    except KeyboardInterrupt:
        final_time = simulator.get_context().get_time()
        print("\n\n" + "=" * 60)
        print("OBSERVATION LOOP STOPPED")
        print("=" * 60)
        print(f"  Simulation time: {final_time:.1f}s")
        print(f"  Total observations: {observation_count}")
        if trajectory is not None:
            print(f"  Trajectory progress: {min(100, final_time/trajectory_duration*100):.1f}%")
        print(f"  Final mean occupancy: {np.mean(voxel_grid.occupancy):.4f}")
        print(f"  Voxels confident FREE (P<0.2): {np.sum(voxel_grid.occupancy < 0.2)}/{voxel_grid.total_voxels}")
        print(f"  Voxels confident OCCUPIED (P>0.8): {np.sum(voxel_grid.occupancy > 0.8)}/{voxel_grid.total_voxels}")
        print(f"  Voxels uncertain (0.2≤P≤0.8): {np.sum((voxel_grid.occupancy >= 0.2) & (voxel_grid.occupancy <= 0.8))}/{voxel_grid.total_voxels}")
        print("=" * 60)
    finally:
        # Close camera visualizer windows
        camera_visualizer.close()
        print("  Camera visualizer closed.")


if __name__ == "__main__":
    main()
