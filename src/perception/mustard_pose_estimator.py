"""
MustardPoseEstimatorSystem - A Drake LeafSystem for ICP-based pose estimation.

This system receives 6 camera point clouds and belief as input, dynamically 
selects cameras based on MAP estimate, fuses point clouds, segments yellow 
points, runs ICP to estimate mustard bottle pose, and visualizes the result.

Architecture:
    - Inputs: 6 camera point clouds (camera0-5) + belief vector
    - Processing: Camera selection based on MAP, point cloud fusion, yellow 
                  segmentation, ICP pose estimation
    - Outputs: Estimated RigidTransform of mustard bottle in world frame
"""

import numpy as np
from pydrake.all import (
    LeafSystem,
    AbstractValue,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    PointCloud,
    Concatenate,
    Fields,
    BaseField,
    Rgba,
)
from manipulation.icp import IterativeClosestPoint
from manipulation.mustard_depth_camera_example import MustardPointCloud
from manipulation.meshcat_utils import AddMeshcatTriad


def ToPointCloud(xyzs, rgbs=None):
    """Convert numpy arrays to Drake PointCloud."""
    if rgbs is not None:
        cloud = PointCloud(xyzs.shape[1], Fields(BaseField.kXYZs | BaseField.kRGBs))
        cloud.mutable_rgbs()[:] = rgbs
    else:
        cloud = PointCloud(xyzs.shape[1])
    cloud.mutable_xyzs()[:] = xyzs
    return cloud


def segment_by_yellow(scene_xyzs, scene_rgbs):
    """
    Segment yellow points (mustard bottle) from point cloud.
    
    Uses inclusive thresholds to capture yellow/gold colors under varying
    lighting conditions. Focuses on color ratios rather than absolute values.
    
    Args:
        scene_xyzs: (3, N) numpy array of XYZ coordinates
        scene_rgbs: (3, N) numpy array of RGB values (0-255)
        
    Returns:
        xyz_yellow: (3, M) segmented XYZ coordinates
        rgb_yellow: (3, M) segmented RGB values
    """
    rgb = scene_rgbs.astype(np.float32)
    r, g, b = rgb[0], rgb[1], rgb[2]
    
    # Normalize to fractions (more robust to lighting variations)
    sum_rgb = r + g + b + 1e-6
    r_frac = r / sum_rgb
    g_frac = g / sum_rgb
    b_frac = b / sum_rgb
    
    # Yellow detection: inclusive thresholds for mustard bottle
    # Yellow/gold colors have high red and green, low blue
    mask_yellow_ratio = (
        (r_frac > 0.25) &      # Red is significant
        (g_frac > 0.20) &      # Green is significant
        (b_frac < 0.40) &      # Blue is low
        (r > b) &              # Red greater than blue
        (g > b * 0.6)          # Green significantly greater than blue
    )
    
    # Brightness filter to remove very dark/black points
    mask_brightness = (r + g + b) > 85
    
    # Height filter: above floor level
    mask_z = scene_xyzs[2, :] > 0.005
    
    # Combined mask
    mask = mask_yellow_ratio & mask_brightness & mask_z
    
    xyz_yellow = scene_xyzs[:, mask]
    rgb_yellow = scene_rgbs[:, mask]
    
    # Fallback if no points found with primary thresholds
    if xyz_yellow.shape[1] == 0:
        mask_fallback = (
            (r_frac > 0.20) &
            (g_frac > 0.15) &
            (b_frac < 0.50) &
            (r > b * 0.8) &
            (g > b * 0.5) &
            (r + g + b) > 30 &
            mask_z
        )
        
        xyz_yellow = scene_xyzs[:, mask_fallback]
        rgb_yellow = scene_rgbs[:, mask_fallback]
    
    # Outlier removal: keep main cluster (98th percentile)
    if xyz_yellow.shape[1] > 10:
        center = np.mean(xyz_yellow, axis=1, keepdims=True)
        diff = xyz_yellow - center
        d2 = np.sum(diff**2, axis=0)
        thr = np.quantile(d2, 0.98)
        mask_cluster = d2 < thr
        xyz_yellow = xyz_yellow[:, mask_cluster]
        rgb_yellow = rgb_yellow[:, mask_cluster]
    
    return xyz_yellow, rgb_yellow


class MustardPoseEstimatorSystem(LeafSystem):
    """
    LeafSystem that estimates mustard bottle pose from camera point clouds.
    
    Takes 6 camera point clouds, belief, and a trigger signal as input. 
    Based on MAP of belief, selects cameras 0-2 (bin0) or cameras 3-5 (bin1), 
    fuses point clouds, segments yellow points, runs ICP, and outputs estimated pose.
    
    IMPORTANT: This system only runs estimation when:
    1. The estimation_trigger input is 1.0 (from BinBeliefEstimatorSystem)
    2. Estimation has not already been completed
    
    Inputs:
        camera0_point_cloud through camera5_point_cloud: PointCloud from each camera
        belief: n_bins vector of bin probabilities
        estimation_trigger: 1.0 when bin belief is confident, 0.0 otherwise
        
    Outputs:
        estimated_pose: RigidTransform of mustard bottle in world frame
    """
    
    def __init__(self, meshcat, n_bins=2, visualize=True):
        """
        Initialize the MustardPoseEstimatorSystem.
        
        Args:
            meshcat: Meshcat visualizer instance
            n_bins: Number of bins in the belief vector (default: 2)
            visualize: Whether to visualize results in Meshcat (default: True)
        """
        LeafSystem.__init__(self)
        
        self._meshcat = meshcat
        self._n_bins = n_bins
        self._visualize = visualize
        self._estimation_complete = False
        self._cached_pose = RigidTransform()
        
        # Load mustard model once
        self._mustard_model = MustardPointCloud()
        
        # Declare 6 point cloud input ports (one per camera)
        model_point_cloud = AbstractValue.Make(PointCloud(0))
        self._pcl_ports = []
        for i in range(6):
            port = self.DeclareAbstractInputPort(
                f"camera{i}_point_cloud", 
                model_point_cloud
            )
            self._pcl_ports.append(port)
        
        # Declare belief input port
        self._belief_port = self.DeclareVectorInputPort("belief", n_bins)
        
        # Declare estimation trigger input port (from BinBeliefEstimatorSystem)
        self._trigger_port = self.DeclareVectorInputPort("estimation_trigger", 1)
        
        # Declare estimated pose output port
        self.DeclareAbstractOutputPort(
            "estimated_pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcEstimatedPose,
        )
        
        # Create triad for visualization
        if self._visualize and meshcat is not None:
            AddMeshcatTriad(meshcat, "estimated_mustard_pose", length=0.1, radius=0.003)
    
    def CalcEstimatedPose(self, context, output):
        """
        Calculate estimated pose based on MAP-selected cameras.
        
        Only runs estimation when estimation_trigger input is 1.0.
        Returns cached/identity pose otherwise to avoid expensive computation.
        """
        # Return cached result if estimation already completed
        if self._estimation_complete:
            output.set_value(self._cached_pose)
            return
        
        # Check trigger from BinBeliefEstimatorSystem
        trigger = self._trigger_port.Eval(context)[0]
        
        # Only run estimation when triggered (bin belief is confident)
        if trigger < 0.5:
            # Not triggered - return identity pose without running estimation
            output.set_value(self._cached_pose)
            return
        
        # Triggered - run estimation
        belief = self._belief_port.Eval(context)
        map_bin = np.argmax(belief)
        
        print(f"\n{'='*60}")
        print("POSE ESTIMATION (MustardPoseEstimatorSystem)")
        print(f"{'='*60}")
        print(f"  Belief: {belief}")
        print(f"  MAP estimate: bin{map_bin}")
        
        # Select camera indices based on MAP
        # bin0 (negative Y): cameras 0, 1, 2
        # bin1 (positive X): cameras 3, 4, 5
        if map_bin == 0:
            camera_indices = [0, 1, 2]
        else:
            camera_indices = [3, 4, 5]
        
        print(f"  Selected cameras: {camera_indices}")
        
        # Get and fuse selected point clouds
        point_clouds = []
        for idx in camera_indices:
            pcl = self._pcl_ports[idx].Eval(context)
            # Crop to reasonable workspace bounds
            pcl_cropped = pcl.Crop(lower_xyz=[-2, -2, -2], upper_xyz=[2, 2, 2])
            point_clouds.append(pcl_cropped)
            print(f"    camera{idx}: {pcl_cropped.size()} points")
        
        fused_pcl = Concatenate(point_clouds)
        print(f"  Fused point cloud: {fused_pcl.size()} points")
        
        # Run perception pipeline
        X_WM = self._run_perception_pipeline(fused_pcl, map_bin)
        
        self._cached_pose = X_WM
        self._estimation_complete = True
        output.set_value(X_WM)
    
    def _run_perception_pipeline(self, fused_pcl, map_bin):
        """
        Run yellow segmentation and ICP on fused point cloud.
        
        Args:
            fused_pcl: Drake PointCloud of fused camera data
            map_bin: MAP estimate of which bin (for logging)
            
        Returns:
            X_WM: RigidTransform of estimated mustard pose in world frame
        """
        # Extract numpy arrays
        scene_xyzs = fused_pcl.xyzs()
        scene_rgbs = fused_pcl.rgbs()
        
        # Segment yellow points
        segmented_xyz, segmented_rgb = segment_by_yellow(scene_xyzs, scene_rgbs)
        print(f"  Segmented {segmented_xyz.shape[1]} yellow points")
        
        if segmented_xyz.shape[1] < 10:
            print("  WARNING: Not enough yellow points found!")
            print("    Returning identity pose as fallback")
            return RigidTransform()
        
        # Visualize segmented points
        if self._visualize and self._meshcat is not None:
            self._meshcat.SetObject(
                "pose_estimator/segmented_pcl",
                ToPointCloud(segmented_xyz),
                rgba=Rgba(0, 1, 0, 1)  # Green
            )
        
        # Compute initial guess from centroid of segmented points
        centroid = np.mean(segmented_xyz, axis=1)
        initial_guess = RigidTransform()
        initial_guess.set_translation(centroid)
        
        print(f"  Running ICP...")
        print(f"    Initial guess (centroid): {centroid}")
        
        # Run ICP
        try:
            X_WM, correspondences = IterativeClosestPoint(
                p_Om=self._mustard_model.xyzs(),
                p_Ws=segmented_xyz,
                X_Ohat=initial_guess,
                max_iterations=100,
            )
            
            print(f"  ICP converged!")
            print(f"    Estimated position: {X_WM.translation()}")
            rpy = RollPitchYaw(X_WM.rotation())
            print(f"    Estimated RPY: [{rpy.roll_angle():.3f}, {rpy.pitch_angle():.3f}, {rpy.yaw_angle():.3f}]")
            
            # Visualize estimated pose
            if self._visualize and self._meshcat is not None:
                self._meshcat.SetTransform("estimated_mustard_pose", X_WM)
                
                # Also show the ICP-aligned model
                model_world_xyz = X_WM @ self._mustard_model.xyzs()
                self._meshcat.SetObject(
                    "pose_estimator/icp_model",
                    ToPointCloud(model_world_xyz),
                    rgba=Rgba(1, 0, 1, 1)  # Magenta
                )
            
            return X_WM
            
        except Exception as e:
            print(f"  ICP failed: {e}")
            print("    Returning centroid-based pose as fallback")
            fallback_pose = RigidTransform()
            fallback_pose.set_translation(centroid)
            return fallback_pose
    
    def reset_estimation(self):
        """Reset to allow re-estimation (call before new RRBT cycle)."""
        self._estimation_complete = False
        self._cached_pose = RigidTransform()
