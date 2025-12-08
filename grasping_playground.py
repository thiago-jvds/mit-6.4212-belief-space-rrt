#!/usr/bin/env python3
"""
Grasping Playground - Simulation with point cloud segmentation and ICP.

Sets up:
- Kuka iiwa robot with wrist camera
- WSG gripper
- Bin with mustard bottle
- Point cloud capture, yellow segmentation, and ICP pose estimation

Usage:
    python grasping_playground.py

Then open http://localhost:7000 in your browser.
"""

import numpy as np
import matplotlib
# Configure matplotlib backend for X11 or headless environments
import os
if 'DISPLAY' in os.environ:
    # X11 is available, try TkAgg backend
    try:
        matplotlib.use('TkAgg')
    except:
        # Fallback to Agg if TkAgg fails
        matplotlib.use('Agg')
else:
    # No display, use non-interactive backend
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from manipulation.station import MakeHardwareStation, LoadScenario, AddPointClouds
from manipulation.icp import IterativeClosestPoint
from manipulation.mustard_depth_camera_example import MustardPointCloud
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    Meshcat,
    MeshcatParams,
    ConstantVectorSource,
    PointCloud,
    Fields,
    BaseField,
    Rgba,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    Parser,
    AddMultibodyPlantSceneGraph,
)


def ToPointCloud(xyzs, rgbs=None):
    """Convert numpy arrays to Drake PointCloud."""
    if rgbs is not None:
        cloud = PointCloud(xyzs.shape[1], Fields(BaseField.kXYZs | BaseField.kRGBs))
        cloud.mutable_rgbs()[:] = rgbs
    else:
        cloud = PointCloud(xyzs.shape[1])
    cloud.mutable_xyzs()[:] = xyzs
    return cloud


def create_yellow_mask_from_rgb(rgb_image):
    """
    Create a boolean mask for yellow pixels in an RGB image.
    Uses the same color thresholds as segment_by_yellow.
    
    Args:
        rgb_image: (H, W, 3) or (H, W, 4) numpy array of uint8 RGB/RGBA image
        
    Returns:
        mask: (H, W) boolean array where True indicates yellow pixels
    """
    # Handle RGBA images
    if rgb_image.shape[-1] == 4:
        rgb = rgb_image[:, :, :3].astype(np.float32)
    else:
        rgb = rgb_image.astype(np.float32)
    
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    
    # Normalize to fractions
    sum_rgb = r + g + b + 1e-6
    r_frac = r / sum_rgb
    g_frac = g / sum_rgb
    b_frac = b / sum_rgb
    
    # Same thresholds as segment_by_yellow
    mask_yellow_ratio = (
        (r_frac > 0.25) &
        (g_frac > 0.20) &
        (b_frac < 0.40) &
        (r > b) &
        (g > b * 0.6)
    )
    
    mask_brightness = (r + g + b) > 50
    
    mask = mask_yellow_ratio & mask_brightness
    
    # Fallback if too few points
    if np.sum(mask) < 100:
        mask_fallback = (
            (r_frac > 0.20) &
            (g_frac > 0.15) &
            (b_frac < 0.50) &
            (r > b * 0.8) &
            (g > b * 0.5) &
            (r + g + b) > 30
        )
        mask = mask_fallback
    
    return mask


def plot_segmentation_results(rgb_image, depth_image, mask, save_path=None, title_prefix=""):
    """
    Plot RGB image, segmented RGB, and segmented depth.
    
    Args:
        rgb_image: (H, W, 3) or (H, W, 4) RGB image
        depth_image: (H, W) or (H, W, 1) depth image in meters
        mask: (H, W) boolean mask for segmentation
        save_path: Optional path to save the figure
        title_prefix: Optional prefix for plot titles (e.g., "Left Camera - ")
    """
    # Prepare RGB image
    if rgb_image.shape[-1] == 4:
        rgb_display = rgb_image[:, :, :3]
    else:
        rgb_display = rgb_image.copy()
    
    # Prepare depth image
    if depth_image.ndim == 3:
        depth_2d = depth_image.squeeze()
    else:
        depth_2d = depth_image.copy()
    
    # Create segmented RGB (set non-yellow pixels to black)
    segmented_rgb = rgb_display.copy()
    segmented_rgb[~mask] = 0
    
    # Create segmented depth (set non-yellow pixels to NaN/black)
    segmented_depth = depth_2d.copy()
    segmented_depth[~mask] = np.nan
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original RGB
    axes[0].imshow(rgb_display)
    axes[0].set_title(f"{title_prefix}Original RGB Image", fontsize=14)
    axes[0].axis('off')
    
    # Plot 2: Segmented RGB
    axes[1].imshow(segmented_rgb)
    axes[1].set_title(f"{title_prefix}Segmented RGB (Yellow Only)", fontsize=14)
    axes[1].axis('off')
    
    # Plot 3: Segmented Depth (grayscale)
    depth_plot = axes[2].imshow(segmented_depth, cmap='gray', vmin=0, vmax=2.0)
    axes[2].set_title(f"{title_prefix}Segmented Depth (B&W)", fontsize=14)
    axes[2].axis('off')
    plt.colorbar(depth_plot, ax=axes[2], label='Depth (m)')
    
    plt.tight_layout()
    
    # Always save the plot to a file
    if save_path is None:
        save_path = "segmentation_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {save_path}")
    
    # Try to show interactively if display is available
    try:
        if 'DISPLAY' in os.environ:
            plt.show(block=False)  # Non-blocking so script continues
            print("  Plot displayed (close window to continue)")
        else:
            print("  No DISPLAY available, plot saved to file only")
    except Exception as e:
        print(f"  Could not display plot interactively: {e}")
        print("  Plot saved to file only")


def segment_by_yellow(scene_pcl_np, scene_rgb_np):
    """
    Segment points that are yellow (mustard bottle color).
    
    Uses inclusive thresholds to capture yellow/gold colors under varying
    lighting conditions. Focuses on color ratios rather than absolute values.
    """
    rgb = scene_rgb_np.astype(np.float32)
    r, g, b = rgb[0], rgb[1], rgb[2]
    
    # Normalize to fractions (more robust to lighting variations)
    sum_rgb = r + g + b + 1e-6
    r_frac = r / sum_rgb
    g_frac = g / sum_rgb
    b_frac = b / sum_rgb
    
    # Yellow detection: More inclusive thresholds
    # Yellow/gold colors have:
    # - High red and green fractions (both > blue)
    # - Low blue fraction
    # - R and G are both significant (typically R >= G or close)
    # - Mustard can be darker (lower absolute values) or lighter
    
    # Primary mask: color ratios indicate yellow/gold
    # More inclusive: allow wider range of yellow shades
    mask_yellow_ratio = (
        (r_frac > 0.25) &      # Red is significant
        (g_frac > 0.20) &      # Green is significant
        (b_frac < 0.40) &      # Blue is low
        (r > b) &              # Red greater than blue
        (g > b * 0.6)          # Green significantly greater than blue
    )
    
    # Secondary mask: absolute values to filter out very dark/black points
    # But be more lenient - mustard can appear darker in shadows
    mask_brightness = (
        (r + g + b) > 50       # Total brightness threshold
    )
    
    # Combined mask
    mask_yellow = mask_yellow_ratio & mask_brightness
    
    xyz_yellow = scene_pcl_np[:, mask_yellow]
    rgb_yellow = scene_rgb_np[:, mask_yellow]
    
    # If no points found, try even more inclusive thresholds
    if xyz_yellow.shape[1] == 0:
        print("Warning: No yellow points found with primary thresholds!")
        print("  Trying more inclusive fallback thresholds...")
        
        # Fallback: even more lenient thresholds
        mask_yellow_fallback = (
            (r_frac > 0.20) &      # Very low red threshold
            (g_frac > 0.15) &      # Very low green threshold
            (b_frac < 0.50) &      # Allow more blue
            (r > b * 0.8) &        # Red at least 80% of blue
            (g > b * 0.5) &        # Green at least 50% of blue
            (r + g + b) > 30       # Very low brightness threshold
        )
        
        xyz_yellow = scene_pcl_np[:, mask_yellow_fallback]
        rgb_yellow = scene_rgb_np[:, mask_yellow_fallback]
        
        if xyz_yellow.shape[1] == 0:
            print("  ✗ Still no points found with fallback thresholds")
            print("    Check camera view, lighting, or color thresholds")
            return xyz_yellow, rgb_yellow
        else:
            print(f"  ✓ Found {xyz_yellow.shape[1]} points with fallback thresholds")
    
    # Outlier removal: keep main cluster
    # Use a more inclusive threshold (98th percentile instead of 95th)
    center = np.mean(xyz_yellow, axis=1, keepdims=True)
    diff = xyz_yellow - center
    d2 = np.sum(diff**2, axis=0)
    
    # Keep points within 98th percentile distance (more inclusive)
    thr = np.quantile(d2, 0.98)
    mask_cluster = d2 < thr
    
    return xyz_yellow[:, mask_cluster], rgb_yellow[:, mask_cluster]


def get_mustard_bottle_model_pointcloud():
    """
    Get the point cloud model of the mustard bottle from the actual mesh.
    
    Uses the manipulation library's MustardPointCloud which loads the actual
    mesh from the SDF file and samples points from it. This ensures the model
    matches the actual geometry and coordinate frame.
    """
    # MustardPointCloud() returns a PointCloud object
    model_pcl_drake = MustardPointCloud()
    # Extract the xyzs as a 3xN numpy array
    model_pcl_np = model_pcl_drake.xyzs()
    return model_pcl_np


def main():
    print("=" * 60)
    print("Grasping Playground")
    print("Point Cloud Segmentation & ICP Pose Estimation")
    print("=" * 60)

    # Start Meshcat
    try:
        params = MeshcatParams()
        params.port = 7000
        meshcat = Meshcat(params=params)
        print(f"✓ Meshcat started at http://localhost:7000")
    except RuntimeError as e:
        print(f"\n✗ ERROR: Could not start Meshcat on port 7000")
        print(f"  {e}")
        raise

    # Load scenario
    scenario_path = Path(__file__).parent / "config" / "grasp_scenario.yaml"
    print(f"Loading scenario: {scenario_path}")
    
    with open(scenario_path, "r") as f:
        scenario = LoadScenario(data=f.read())

    # Build the diagram
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario=scenario, meshcat=meshcat))
    
    # Get references
    plant = station.GetSubsystemByName("plant")

    # Use q_home to match the default joint positions in the scenario
    # This prevents the robot from slumping
    q_home = np.array([0, 0.1, 0, -1.2, 0, 1.6, 0])
    
    iiwa_position_source = builder.AddSystem(ConstantVectorSource(q_home))
    builder.Connect(
        iiwa_position_source.get_output_port(), 
        station.GetInputPort("iiwa.position")
    )

    # Set gripper to open position
    wsg_position_source = builder.AddSystem(ConstantVectorSource([0.1]))
    builder.Connect(
        wsg_position_source.get_output_port(), 
        station.GetInputPort("wsg.position")
    )

    # Add point cloud generation from camera
    to_point_cloud = AddPointClouds(
        scenario=scenario,
        station=station,
        builder=builder,
        meshcat=meshcat,
    )
    
    # Export both camera point cloud outputs
    if isinstance(to_point_cloud, dict):
        builder.ExportOutput(
            to_point_cloud["left_wrist_camera_sensor"].get_output_port(),
            "left_camera_point_cloud"
        )
        builder.ExportOutput(
            to_point_cloud["right_wrist_camera_sensor"].get_output_port(),
            "right_camera_point_cloud"
        )
    else:
        # Fallback for list format
        builder.ExportOutput(
            to_point_cloud[0].get_output_port(),
            "left_camera_point_cloud"
        )
        if len(to_point_cloud) > 1:
            builder.ExportOutput(
                to_point_cloud[1].get_output_port(),
                "right_camera_point_cloud"
            )

    # Build diagram and create simulator
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    print("\n" + "-" * 40)
    print("Phase 1: Letting mustard bottle fall into bin...")
    print("-" * 40)

    # Let the mustard bottle fall and settle
    simulator.AdvanceTo(2.0)
    
    print("✓ Bottle settled")

    # Get the current context for point cloud capture
    context = simulator.get_context()
    
    # Capture the point cloud
    print("\n" + "-" * 40)
    print("Phase 2: Capturing point cloud from wrist camera...")
    print("-" * 40)
    
    # Force publish to update visualization
    diagram.ForcedPublish(context)
    
    # Get the point clouds from both cameras
    left_pcl_drake = (
        diagram.GetOutputPort("left_camera_point_cloud")
        .Eval(context)
        .Crop(lower_xyz=[-2, -2, -2], upper_xyz=[2, 2, 2])
    )
    right_pcl_drake = (
        diagram.GetOutputPort("right_camera_point_cloud")
        .Eval(context)
        .Crop(lower_xyz=[-2, -2, -2], upper_xyz=[2, 2, 2])
    )
    
    print(f"  Left camera: {left_pcl_drake.xyzs().shape[1]} points")
    print(f"  Right camera: {right_pcl_drake.xyzs().shape[1]} points")
    
    # Visualize both raw point clouds in meshcat
    meshcat.SetObject("pcl_left_raw", left_pcl_drake)
    meshcat.SetObject("pcl_right_raw", right_pcl_drake)
    
    # Fuse both point clouds together
    left_xyzs = left_pcl_drake.xyzs()
    left_rgbs = left_pcl_drake.rgbs()
    right_xyzs = right_pcl_drake.xyzs()
    right_rgbs = right_pcl_drake.rgbs()
    
    # Concatenate point clouds
    scene_xyzs = np.hstack([left_xyzs, right_xyzs])
    scene_rgbs = np.hstack([left_rgbs, right_rgbs])
    
    print(f"  Fused point cloud: {scene_xyzs.shape[1]} points")
    
    # Visualize fused point cloud
    fused_pcl = ToPointCloud(scene_xyzs, scene_rgbs)
    meshcat.SetObject("pcl_fused_raw", fused_pcl)
    
    # Get RGB and depth images from BOTH cameras for visualization
    print("\n  Getting RGB and depth images from both cameras...")
    station_context = station.GetMyContextFromRoot(context)
    
    # Left camera images
    left_rgb_image = None
    left_depth_image = None
    left_yellow_mask = None
    try:
        left_rgb_obj = station.GetOutputPort("left_wrist_camera_sensor.rgb_image").Eval(station_context)
        left_depth_obj = station.GetOutputPort("left_wrist_camera_sensor.depth_image").Eval(station_context)
        
        left_rgb_image = np.array(left_rgb_obj.data, copy=True)
        left_depth_image = np.array(left_depth_obj.data, copy=True)
        
        if left_rgb_image.ndim == 1:
            left_rgb_image = left_rgb_image.reshape(left_rgb_obj.height(), left_rgb_obj.width(), -1)
        if left_depth_image.ndim == 1:
            left_depth_image = left_depth_image.reshape(left_depth_obj.height(), left_depth_obj.width())
        elif left_depth_image.ndim == 3:
            left_depth_image = left_depth_image.squeeze()
        
        left_yellow_mask = create_yellow_mask_from_rgb(left_rgb_image)
        print(f"  Left camera - RGB: {left_rgb_image.shape}, Yellow: {np.sum(left_yellow_mask)} pixels")
    except Exception as e:
        print(f"  ⚠ Could not get left camera images: {e}")
    
    # Right camera images
    right_rgb_image = None
    right_depth_image = None
    right_yellow_mask = None
    try:
        right_rgb_obj = station.GetOutputPort("right_wrist_camera_sensor.rgb_image").Eval(station_context)
        right_depth_obj = station.GetOutputPort("right_wrist_camera_sensor.depth_image").Eval(station_context)
        
        right_rgb_image = np.array(right_rgb_obj.data, copy=True)
        right_depth_image = np.array(right_depth_obj.data, copy=True)
        
        if right_rgb_image.ndim == 1:
            right_rgb_image = right_rgb_image.reshape(right_rgb_obj.height(), right_rgb_obj.width(), -1)
        if right_depth_image.ndim == 1:
            right_depth_image = right_depth_image.reshape(right_depth_obj.height(), right_depth_obj.width())
        elif right_depth_image.ndim == 3:
            right_depth_image = right_depth_image.squeeze()
        
        right_yellow_mask = create_yellow_mask_from_rgb(right_rgb_image)
        print(f"  Right camera - RGB: {right_rgb_image.shape}, Yellow: {np.sum(right_yellow_mask)} pixels")
    except Exception as e:
        print(f"  ⚠ Could not get right camera images: {e}")
    
    # Segment by yellow color on the FUSED point cloud
    print("\n" + "-" * 40)
    print("Phase 3: Segmenting yellow points (mustard bottle) from fused point cloud...")
    print("-" * 40)
    
    segmented_xyz, segmented_rgb = segment_by_yellow(scene_xyzs, scene_rgbs)
    print(f"  Found {segmented_xyz.shape[1]} yellow points in fused cloud")
    
    # Plot segmentation results for BOTH cameras
    print("\n" + "-" * 40)
    print("Plotting segmentation results for both cameras...")
    print("-" * 40)
    print(f"  Matplotlib backend: {matplotlib.get_backend()}")
    print(f"  DISPLAY env var: {os.environ.get('DISPLAY', 'Not set')}")
    
    if left_rgb_image is not None and left_depth_image is not None and left_yellow_mask is not None:
        plot_segmentation_results(left_rgb_image, left_depth_image, left_yellow_mask, 
                                  save_path="segmentation_results_left.png",
                                  title_prefix="Left Camera - ")
        print("  Saved left camera segmentation to segmentation_results_left.png")
    
    if right_rgb_image is not None and right_depth_image is not None and right_yellow_mask is not None:
        plot_segmentation_results(right_rgb_image, right_depth_image, right_yellow_mask, 
                                  save_path="segmentation_results_right.png",
                                  title_prefix="Right Camera - ")
        print("  Saved right camera segmentation to segmentation_results_right.png")
    
    if segmented_xyz.shape[1] > 0:
        # Visualize segmented point cloud in green
        meshcat.SetObject(
            "pcl_segmented",
            ToPointCloud(segmented_xyz),
            rgba=Rgba(0, 1, 0, 1)  # Green
        )
        
        # Generate model point cloud
        print("\n" + "-" * 40)
        print("Phase 4: Running ICP pose estimation...")
        print("-" * 40)
        
        model_pcl = get_mustard_bottle_model_pointcloud()
        print(f"  Model has {model_pcl.shape[1]} points (from actual mesh)")
        
        # Visualize model point cloud at origin (blue)
        meshcat.SetObject(
            "pcl_model_origin",
            ToPointCloud(model_pcl),
            rgba=Rgba(0, 0, 1, 0.5)  # Blue, semi-transparent
        )
        
        # Initial guess: center of segmented points, with some reasonable orientation
        # The bottle is lying on its side in the bin
        centroid = np.mean(segmented_xyz, axis=1)
        initial_guess = RigidTransform()
        initial_guess.set_translation(centroid)
        # Start with bottle lying on its side (rotated 90 deg about X)
        initial_guess.set_rotation(RotationMatrix(RollPitchYaw(np.pi/2, np.pi/2, np.pi/2)))
        
        print(f"  Initial guess centroid: {centroid}")
        
        # Run ICP
        try:
            X_WM_estimated, correspondences = IterativeClosestPoint(
                p_Om=model_pcl,
                p_Ws=segmented_xyz,
                X_Ohat=initial_guess,
                meshcat=meshcat,
                meshcat_scene_path="icp_iterations",
                max_iterations=100,
            )
            
            print(f"  ✓ ICP converged!")
            print(f"    Estimated position: {X_WM_estimated.translation()}")
            rpy = RollPitchYaw(X_WM_estimated.rotation())
            print(f"    Estimated RPY: [{rpy.roll_angle():.3f}, {rpy.pitch_angle():.3f}, {rpy.yaw_angle():.3f}]")
            
            # Visualize the estimated model pose (magenta)
            meshcat.SetObject(
                "pcl_estimated",
                ToPointCloud(model_pcl),
                rgba=Rgba(1, 0, 1, 1)  # Magenta
            )
            meshcat.SetTransform("pcl_estimated", X_WM_estimated)
            
            # Get ground truth pose from simulation
            plant_context = plant.GetMyContextFromRoot(context)
            mustard_body = plant.GetBodyByName("base_link_mustard")
            X_WM_true = plant.EvalBodyPoseInWorld(plant_context, mustard_body)
            
            print(f"\n  Ground truth position: {X_WM_true.translation()}")
            rpy_true = RollPitchYaw(X_WM_true.rotation())
            print(f"  Ground truth RPY: [{rpy_true.roll_angle():.3f}, {rpy_true.pitch_angle():.3f}, {rpy_true.yaw_angle():.3f}]")
            
            # Compute error
            pos_error = np.linalg.norm(X_WM_estimated.translation() - X_WM_true.translation())
            print(f"\n  Position error: {pos_error*1000:.1f} mm")
            
        except Exception as e:
            print(f"  ✗ ICP failed: {e}")
    else:
        print("  ✗ No yellow points found - cannot run ICP")
        print("    Try adjusting the robot pose or color thresholds")

    print("\n" + "=" * 60)
    print("Visualization Guide:")
    print("  - pcl_left_raw: Raw point cloud from left camera")
    print("  - pcl_right_raw: Raw point cloud from right camera")
    print("  - pcl_fused_raw: Fused point cloud from both cameras")
    print("  - pcl_segmented: Yellow-segmented points (green)")
    print("  - pcl_model_origin: Model point cloud at origin (blue)")
    print("  - pcl_estimated: ICP result - model at estimated pose (magenta)")
    print("=" * 60)
    
    print("\nPress Ctrl+C to exit.")

    # Keep running so user can interact with Meshcat
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
