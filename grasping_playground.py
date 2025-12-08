#!/usr/bin/env python3
"""
Grasping Playground - Simulation with point cloud segmentation and ICP.

Sets up:
- Kuka iiwa robot with WSG gripper
- Two bins with fixed cameras (from manipulation package)
- Mustard bottle randomly positioned above bin0
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
from manipulation.utils import ConfigureParser
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Concatenate,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
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
    RandomGenerator,
    UniformlyRandomRotationMatrix,
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


# ============================================================================
# GRASP SELECTION FUNCTIONS
# ============================================================================

def make_internal_model():
    """
    Create an internal model for grasp planning/collision checking.
    This contains just the gripper, cameras, and bins - no objects.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/clutter_planning.dmd.yaml")
    plant.Finalize()
    return builder.Build()


def draw_grasp_candidate(meshcat, X_G, prefix="gripper", draw_frames=True):
    """
    Draw a gripper at the given pose in Meshcat.
    
    Args:
        meshcat: The Meshcat instance
        X_G: RigidTransform for the gripper pose
        prefix: Meshcat path prefix for the visualization
        draw_frames: Whether to draw coordinate frames (not used currently)
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/schunk_wsg_50_welded_fingers.sdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body"), X_G)
    plant.Finalize()

    params = MeshcatVisualizerParams()
    params.prefix = prefix
    params.delete_prefix_on_initialization_event = False
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)


def GraspCandidateCost(
    diagram,
    context,
    cloud,
    wsg_body_index=None,
    plant_system_name="plant",
    scene_graph_system_name="scene_graph",
    adjust_X_G=False,
    verbose=False,
):
    """
    Compute the cost of a grasp candidate.
    
    Args:
        diagram: A diagram containing a MultibodyPlant+SceneGraph with a free gripper
        context: The diagram context
        cloud: A PointCloud representing the object to grasp
        wsg_body_index: The body index of the gripper
        adjust_X_G: If True, adjust gripper position to center on grasped points
        verbose: Print debug info
    
    Returns:
        cost: The grasp cost (lower is better, inf if invalid)
    """
    plant = diagram.GetSubsystemByName(plant_system_name)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
    
    if wsg_body_index:
        wsg = plant.get_body(wsg_body_index)
    else:
        wsg = plant.GetBodyByName("body")
        wsg_body_index = wsg.index()

    X_G = plant.GetFreeBodyPose(plant_context, wsg)

    # Transform cloud into gripper frame
    X_GW = X_G.inverse()
    p_GC = X_GW @ cloud.xyzs()

    # Crop to a region inside of the finger box
    crop_min = [-0.05, 0.1, -0.00625]
    crop_max = [0.05, 0.1125, 0.00625]
    indices = np.all(
        (
            crop_min[0] <= p_GC[0, :],
            p_GC[0, :] <= crop_max[0],
            crop_min[1] <= p_GC[1, :],
            p_GC[1, :] <= crop_max[1],
            crop_min[2] <= p_GC[2, :],
            p_GC[2, :] <= crop_max[2],
        ),
        axis=0,
    )

    if adjust_X_G and np.sum(indices) > 0:
        p_GC_x = p_GC[0, indices]
        p_Gcenter_x = (p_GC_x.min() + p_GC_x.max()) / 2.0
        X_G.set_translation(X_G @ np.array([p_Gcenter_x, 0, 0]))
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        X_GW = X_G.inverse()

    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)

    # Check collisions between the gripper and the environment
    if query_object.HasCollisions():
        if verbose:
            print("Gripper is colliding with environment!")
        return np.inf

    # Check collisions between the gripper and the point cloud
    margin = 0.0
    for i in range(cloud.size()):
        distances = query_object.ComputeSignedDistanceToPoint(
            cloud.xyz(i), threshold=margin
        )
        if distances:
            if verbose:
                print("Gripper is colliding with point cloud!")
            return np.inf

    # Need normals for cost computation
    if not cloud.has_normals():
        return 0.0  # Can't compute orientation cost without normals
    
    n_GC = X_GW.rotation().multiply(cloud.normals()[:, indices])

    # Penalize deviation of the gripper from vertical
    cost = 20.0 * X_G.rotation().matrix()[2, 1]

    # Reward sum |dot product of normals with gripper x|^2
    if n_GC.shape[1] > 0:
        cost -= np.sum(n_GC[0, :] ** 2)
    
    if verbose:
        print(f"cost: {cost}")
    return cost


def GenerateAntipodalGraspCandidate(
    diagram,
    context,
    cloud,
    rng,
    wsg_body_index=None,
    plant_system_name="plant",
    scene_graph_system_name="scene_graph",
):
    """
    Generate an antipodal grasp candidate by picking a random point and aligning
    the gripper with its normal.
    """
    cost, X_G, _ = GenerateAntipodalGraspCandidateDebug(
        diagram, context, cloud, rng, wsg_body_index, 
        plant_system_name, scene_graph_system_name, verbose=False
    )
    return cost, X_G


def GenerateAntipodalGraspCandidateDebug(
    diagram,
    context,
    cloud,
    rng,
    wsg_body_index=None,
    plant_system_name="plant",
    scene_graph_system_name="scene_graph",
    verbose=False,
):
    """
    Generate an antipodal grasp candidate with debug info.
    
    Returns:
        cost: The grasp cost
        X_G: The grasp pose (RigidTransform), or None if no valid grasp found
        reject_reason: String describing why grasp was rejected, or None if valid
    """
    plant = diagram.GetSubsystemByName(plant_system_name)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
    
    if wsg_body_index:
        wsg = plant.get_body(wsg_body_index)
    else:
        wsg = plant.GetBodyByName("body")
        wsg_body_index = wsg.index()

    if cloud.size() < 2:
        return np.inf, None, "empty_cloud"
        
    index = rng.integers(0, cloud.size() - 1)

    # Use S for sample point/frame
    p_WS = cloud.xyz(index)
    n_WS = cloud.normal(index)

    norm = np.linalg.norm(n_WS)
    if not np.isclose(norm, 1.0):
        if norm < 1e-6:
            return np.inf, None, "zero_normal"
        n_WS = n_WS / norm

    if verbose:
        print(f"    DEBUG: Point {index}, pos={p_WS}, normal={n_WS}")

    Gx = n_WS  # gripper x axis aligns with normal
    # make orthonormal y axis, aligned with world down
    y = np.array([0.0, 0.0, -1.0])
    dot_y_gx = np.dot(y, Gx)
    
    if verbose:
        print(f"    DEBUG: dot(y, Gx) = {dot_y_gx:.4f}")
    
    if np.abs(dot_y_gx) > 0.999:
        # normal was pointing straight down/up, reject this sample
        return np.inf, None, "normal_down"

    Gy = y - dot_y_gx * Gx
    Gy = Gy / np.linalg.norm(Gy)
    Gz = np.cross(Gx, Gy)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    p_GS_G = [0.054 - 0.01, 0.10625, 0]

    # Try orientations from the center out
    min_roll = -np.pi / 3.0
    max_roll = np.pi / 3.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    
    collision_count = 0
    for theta in min_roll + (max_roll - min_roll) * alpha:
        # Rotate the object in the hand by a random rotation (around the normal)
        R_WG2 = R_WG.multiply(RotationMatrix.MakeXRotation(theta))

        # Use G for gripper frame
        p_SG_W = -R_WG2.multiply(p_GS_G)
        p_WG = p_WS + p_SG_W

        X_G = RigidTransform(R_WG2, p_WG)
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        
        # Check collisions to determine reason
        query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
        if query_object.HasCollisions():
            collision_count += 1
            if verbose:
                print(f"    DEBUG: theta={theta:.2f}, collision with environment")
            continue
        
        cost = GraspCandidateCost(diagram, context, cloud, adjust_X_G=True, verbose=verbose)
        X_G = plant.GetFreeBodyPose(plant_context, wsg)
        if np.isfinite(cost):
            return cost, X_G, None
    
    # All orientations failed
    if collision_count == len(alpha):
        return np.inf, None, "collision_env"
    else:
        return np.inf, None, "collision_cloud"


def select_best_grasp(meshcat, cloud, rng, num_candidates=100, debug=False):
    """
    Sample grasp candidates and select the best one.
    
    Args:
        meshcat: Meshcat instance for visualization
        cloud: PointCloud with normals (the object point cloud in world frame)
        rng: numpy random generator
        num_candidates: Number of grasp candidates to sample
        debug: If True, print debug info for first few candidates
    
    Returns:
        best_X_G: The best grasp pose, or None if no valid grasp found
        best_cost: The cost of the best grasp
    """
    print(f"\n  Sampling {num_candidates} grasp candidates...")
    
    # Debug: check normals
    if debug and cloud.has_normals():
        normals = cloud.normals()
        print(f"  DEBUG: Cloud has {cloud.size()} points with normals")
        # Check how many normals are pointing roughly up vs sideways
        z_component = np.abs(normals[2, :])
        up_count = np.sum(z_component > 0.9)
        side_count = np.sum(z_component < 0.3)
        print(f"  DEBUG: Normals pointing up (|z|>0.9): {up_count}")
        print(f"  DEBUG: Normals pointing sideways (|z|<0.3): {side_count}")
    
    # Create internal model for collision checking
    internal_model = make_internal_model()
    internal_model_context = internal_model.CreateDefaultContext()
    
    costs = []
    X_Gs = []
    
    # Debug counters
    reject_normal_down = 0
    reject_collision_env = 0
    reject_collision_cloud = 0
    reject_other = 0
    
    for i in range(num_candidates):
        cost, X_G, reject_reason = GenerateAntipodalGraspCandidateDebug(
            internal_model, internal_model_context, cloud, rng,
            verbose=(debug and i < 5)  # Verbose for first 5 candidates
        )
        if np.isfinite(cost):
            costs.append(cost)
            X_Gs.append(X_G)
        else:
            if reject_reason == "normal_down":
                reject_normal_down += 1
            elif reject_reason == "collision_env":
                reject_collision_env += 1
            elif reject_reason == "collision_cloud":
                reject_collision_cloud += 1
            else:
                reject_other += 1
        
        # Progress update every 20 candidates
        if (i + 1) % 20 == 0:
            print(f"    Processed {i+1}/{num_candidates}, found {len(costs)} valid grasps")
    
    # Print rejection summary
    print(f"\n  Rejection summary:")
    print(f"    Normal pointing down: {reject_normal_down}")
    print(f"    Collision with environment: {reject_collision_env}")
    print(f"    Collision with point cloud: {reject_collision_cloud}")
    print(f"    Other/no valid orientation: {reject_other}")
    
    if len(costs) == 0:
        print("  ✗ No valid grasp candidates found!")
        return None, np.inf
    
    # Sort by cost and get best candidates
    indices = np.asarray(costs).argsort()
    
    # Draw top 5 grasps (or fewer if we have less)
    num_to_draw = min(5, len(indices))
    print(f"\n  Drawing top {num_to_draw} grasp candidates:")
    
    for rank in range(num_to_draw):
        idx = indices[rank]
        print(f"    {rank+1}. Cost: {costs[idx]:.3f}")
        draw_grasp_candidate(meshcat, X_Gs[idx], prefix=f"grasp_{rank+1}_best")
    
    best_idx = indices[0]
    return X_Gs[best_idx], costs[best_idx]


def main():
    print("=" * 60)
    print("Grasping Playground")
    print("Two Bins with Cameras + Mustard Bottle ICP")
    print("=" * 60)

    # Random seed for reproducible results (change for different runs)
    rng = np.random.default_rng(seed=None)  # None = random seed each run

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

    # Add point cloud generation from cameras
    # The cameras (camera0, camera1, camera2) are defined in two_bins_w_cameras.dmd.yaml
    to_point_cloud = AddPointClouds(
        scenario=scenario,
        station=station,
        builder=builder,
        meshcat=meshcat,
    )
    
    # Export point cloud output ports
    # to_point_cloud is a dict mapping camera name -> DepthImageToPointCloud system
    for camera_name, converter in to_point_cloud.items():
        builder.ExportOutput(
            converter.get_output_port(),
            f"{camera_name}_point_cloud"
        )

    # Build diagram and create simulator
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    # Get context and plant context
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # ========== Randomize mustard bottle position above bin0 ==========
    print("\n" + "-" * 40)
    print("Phase 0: Randomizing mustard bottle position above bin0...")
    print("-" * 40)
    
    # Get bin0's pose in world frame
    bin0_instance = plant.GetModelInstanceByName("bin0")
    bin0_body = plant.GetBodyByName("bin_base", bin0_instance)
    X_WB = plant.EvalBodyPoseInWorld(plant_context, bin0_body)
    
    # Generate random position and orientation for mustard bottle
    generator = RandomGenerator(rng.integers(1000))  # C++ random generator
    random_rotation = UniformlyRandomRotationMatrix(generator)
    
    # Random XY position within bin bounds, Z height above bin
    random_x = rng.uniform(-0.15, 0.15)
    random_y = rng.uniform(-0.2, 0.2)
    random_z = 0.25  # Height above bin
    
    # Create transform relative to bin, then convert to world frame
    X_BM = RigidTransform(random_rotation, [random_x, random_y, random_z])
    X_WM = X_WB.multiply(X_BM)
    
    # Set mustard bottle pose
    mustard_body = plant.GetBodyByName("base_link_mustard")
    plant.SetFreeBodyPose(plant_context, mustard_body, X_WM)
    
    print(f"  Bin0 position: {X_WB.translation()}")
    print(f"  Mustard random offset: [{random_x:.3f}, {random_y:.3f}, {random_z:.3f}]")
    print(f"  Mustard world position: {X_WM.translation()}")

    # ========== Let bottle fall and settle ==========
    print("\n" + "-" * 40)
    print("Phase 1: Letting mustard bottle fall into bin0...")
    print("-" * 40)

    # Let the mustard bottle fall and settle
    simulator.AdvanceTo(2.0)
    
    print("✓ Bottle settled")

    # Get the current context for point cloud capture
    context = simulator.get_context()
    
    # Capture the point cloud
    print("\n" + "-" * 40)
    print("Phase 2: Capturing point clouds from fixed cameras...")
    print("-" * 40)
    
    # Force publish to update visualization
    diagram.ForcedPublish(context)
    
    # Camera names from two_bins_w_cameras.dmd.yaml
    camera_names = ["camera0", "camera1", "camera2"]
    
    # Get point clouds from all 3 cameras
    point_clouds = []
    for cam_name in camera_names:
        try:
            pcl = (
                diagram.GetOutputPort(f"{cam_name}_point_cloud")
                .Eval(context)
                .Crop(lower_xyz=[-2, -2, -2], upper_xyz=[2, 2, 2])
            )
            point_clouds.append(pcl)
            print(f"  {cam_name}: {pcl.xyzs().shape[1]} points")
            
            # Visualize each camera's raw point cloud
            meshcat.SetObject(f"pcl_{cam_name}_raw", pcl)
        except Exception as e:
            print(f"  ⚠ Could not get {cam_name} point cloud: {e}")
    
    # Fuse all point clouds using Drake's Concatenate
    if len(point_clouds) > 0:
        fused_pcl = Concatenate(point_clouds)
        print(f"  Fused point cloud: {fused_pcl.xyzs().shape[1]} points")
        
        # Visualize fused point cloud
        meshcat.SetObject("pcl_fused_raw", fused_pcl)
        
        scene_xyzs = fused_pcl.xyzs()
        scene_rgbs = fused_pcl.rgbs()
    else:
        print("  ⚠ No point clouds captured!")
        scene_xyzs = np.array([[], [], []])
        scene_rgbs = np.array([[], [], []])
    
    # Get RGB and depth images from cameras for visualization
    print("\n  Getting RGB and depth images from cameras...")
    
    # Store images from first two cameras for visualization
    camera_images = {}
    for i, cam_name in enumerate(camera_names[:2]):  # Just first 2 cameras for plots
        try:
            rgb_port = diagram.GetOutputPort(f"{cam_name}.rgb_image")
            depth_port = diagram.GetOutputPort(f"{cam_name}.depth_image")
            
            rgb_obj = rgb_port.Eval(context)
            depth_obj = depth_port.Eval(context)
            
            rgb_image = np.array(rgb_obj.data, copy=True)
            depth_image = np.array(depth_obj.data, copy=True)
            
            if rgb_image.ndim == 1:
                rgb_image = rgb_image.reshape(rgb_obj.height(), rgb_obj.width(), -1)
            if depth_image.ndim == 1:
                depth_image = depth_image.reshape(depth_obj.height(), depth_obj.width())
            elif depth_image.ndim == 3:
                depth_image = depth_image.squeeze()
            
            yellow_mask = create_yellow_mask_from_rgb(rgb_image)
            
            camera_images[cam_name] = {
                'rgb': rgb_image,
                'depth': depth_image,
                'mask': yellow_mask
            }
            print(f"  {cam_name} - RGB: {rgb_image.shape}, Yellow: {np.sum(yellow_mask)} pixels")
        except Exception as e:
            print(f"  ⚠ Could not get {cam_name} images: {e}")
    
    # Segment by yellow color on the FUSED point cloud
    print("\n" + "-" * 40)
    print("Phase 3: Segmenting yellow points (mustard bottle) from fused point cloud...")
    print("-" * 40)
    
    segmented_xyz, segmented_rgb = segment_by_yellow(scene_xyzs, scene_rgbs)
    print(f"  Found {segmented_xyz.shape[1]} yellow points in fused cloud")
    
    # Plot segmentation results for cameras
    print("\n" + "-" * 40)
    print("Plotting segmentation results...")
    print("-" * 40)
    print(f"  Matplotlib backend: {matplotlib.get_backend()}")
    print(f"  DISPLAY env var: {os.environ.get('DISPLAY', 'Not set')}")
    
    for cam_name, images in camera_images.items():
        plot_segmentation_results(
            images['rgb'], images['depth'], images['mask'], 
            save_path=f"segmentation_results_{cam_name}.png",
            title_prefix=f"{cam_name.upper()} - "
        )
        print(f"  Saved {cam_name} segmentation to segmentation_results_{cam_name}.png")
    
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
        initial_guess.set_rotation(RotationMatrix(RollPitchYaw(np.pi/2, np.pi/2, 0)))
        
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
            
            # ========== GRASP SELECTION ==========
            print("\n" + "-" * 40)
            print("Phase 5: Grasp Selection on ICP-fitted model...")
            print("-" * 40)
            
            # Transform model point cloud to estimated world pose
            model_world_xyz = X_WM_estimated @ model_pcl
            
            # Create a Drake PointCloud with the transformed model
            grasp_cloud = PointCloud(model_world_xyz.shape[1], Fields(BaseField.kXYZs | BaseField.kNormals))
            grasp_cloud.mutable_xyzs()[:] = model_world_xyz
            
            # Estimate normals on the model point cloud
            grasp_cloud.EstimateNormals(radius=0.05, num_closest=30)
            
            # Flip normals outward (away from centroid)
            centroid_model = np.mean(model_world_xyz, axis=1)
            grasp_cloud.FlipNormalsTowardPoint(centroid_model + np.array([0, 0, 1]))  # Flip away from center
            
            print(f"  Model cloud in world frame: {grasp_cloud.size()} points with normals")
            
            # Visualize the model with normals
            meshcat.SetObject("pcl_grasp_cloud", grasp_cloud, point_size=0.003, rgba=Rgba(0, 1, 1, 1))
            
            # Select best grasp (with debug=True to see rejection reasons)
            best_X_G, best_cost = select_best_grasp(meshcat, grasp_cloud, rng, num_candidates=10000, debug=True)
            
            if best_X_G is not None:
                print(f"\n  ✓ Best grasp found!")
                print(f"    Cost: {best_cost:.3f}")
                print(f"    Position: {best_X_G.translation()}")
                rpy_grasp = RollPitchYaw(best_X_G.rotation())
                print(f"    RPY: [{rpy_grasp.roll_angle():.3f}, {rpy_grasp.pitch_angle():.3f}, {rpy_grasp.yaw_angle():.3f}]")
                
                # ========== PRE-GRASP POSE CALCULATION ==========
                print("\n" + "-" * 40)
                print("Phase 6: Computing Pre-Grasp Pose...")
                print("-" * 40)
                
                # Get the best grasp position
                grasp_pos = best_X_G.translation()
                
                # Pre-grasp position: same x, y, but z + 0.3 (30cm above grasp)
                pre_grasp_pos = np.array([
                    grasp_pos[0],       # Same X
                    grasp_pos[1],       # Same Y
                    grasp_pos[2] + 0.3  # 0.3m above the grasp
                ])
                
                # Pre-grasp orientation: gripper pointing straight down
                # We want gripper Y-axis (approach direction) to align with world -Z (down)
                # This is achieved by rotating -90 degrees about the world X-axis
                #
                # The rotation maps:
                #   - Gripper X → World X:  [1, 0, 0] (fingers close horizontally)
                #   - Gripper Y → World -Z: [0, 0, -1] (approach direction points down)
                #   - Gripper Z → World Y:  [0, 1, 0] (finger length horizontal)
                R_pregrasp = RotationMatrix(RollPitchYaw(-np.pi/2, 0, 0))
                
                # Create the pre-grasp RigidTransform
                X_pregrasp = RigidTransform(R_pregrasp, pre_grasp_pos)
                
                print(f"  Grasp position:     {grasp_pos}")
                print(f"  Pre-grasp position: {pre_grasp_pos}")
                rpy_pregrasp = RollPitchYaw(R_pregrasp)
                print(f"  Pre-grasp RPY:      [{rpy_pregrasp.roll_angle():.3f}, {rpy_pregrasp.pitch_angle():.3f}, {rpy_pregrasp.yaw_angle():.3f}]")
                
                # Visualize as a triad (coordinate frame) in Meshcat
                # Red = X (finger close direction), Green = Y (approach, pointing down), Blue = Z
                meshcat.SetTriad(
                    path="pre_grasp_pose",
                    length=0.1,      # 10cm axes
                    radius=0.005     # 5mm thick
                )
                meshcat.SetTransform("pre_grasp_pose", X_pregrasp)
                
                # Also draw a gripper at the pre-grasp pose for visualization
                draw_grasp_candidate(meshcat, X_pregrasp, prefix="pre_grasp_gripper")
                
                print(f"\n  ✓ Pre-grasp pose visualized in Meshcat")
                print(f"    - 'pre_grasp_pose': Coordinate frame triad")
                print(f"    - 'pre_grasp_gripper': Gripper visualization")
                
            else:
                print("\n  ✗ No valid grasp found")
            
        except Exception as e:
            print(f"  ✗ ICP failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ✗ No yellow points found - cannot run ICP")
        print("    Try adjusting the robot pose or color thresholds")

    print("\n" + "=" * 60)
    print("Visualization Guide:")
    print("  - pcl_camera0_raw: Raw point cloud from camera0")
    print("  - pcl_camera1_raw: Raw point cloud from camera1")
    print("  - pcl_camera2_raw: Raw point cloud from camera2")
    print("  - pcl_fused_raw: Fused point cloud from all cameras")
    print("  - pcl_segmented: Yellow-segmented points (green)")
    print("  - pcl_model_origin: Model point cloud at origin (blue)")
    print("  - pcl_estimated: ICP result - model at estimated pose (magenta)")
    print("  - pcl_grasp_cloud: Model with normals for grasp selection (cyan)")
    print("  - grasp_N_best: Top N grasp candidates (gripper visualizations)")
    print("  - pre_grasp_pose: Coordinate triad for pre-grasp approach pose")
    print("  - pre_grasp_gripper: Gripper at pre-grasp pose (pointing down)")
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
