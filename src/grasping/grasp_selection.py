"""
Grasp Selection Module - Functions for antipodal grasp selection and pre-grasp computation.

Extracted and adapted from grasping_playground.py for use in the belief-space planning pipeline.

Key functions:
- make_internal_model(): Creates internal model for collision checking
- select_best_grasp(): Samples grasp candidates and returns the best one
- compute_pregrasp_pose(): Computes pre-grasp pose 30cm above grasp
- sample_position_from_covariance(): Samples position from 3D Gaussian
"""

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PointCloud,
    Fields,
    BaseField,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
)
from manipulation.utils import ConfigureParser


def make_internal_model():
    """
    Create an internal model for grasp planning/collision checking.

    This is the model in the robot's "head" - contains gripper, cameras, bins.
    Used to check collisions between the gripper and the environment.
    (Based on grasp_selection_example.py)
    
    Returns:
        Built diagram with plant and scene_graph
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


def select_best_grasp(meshcat, cloud, rng, num_candidates=100, num_to_draw=5, debug=False):
    """
    Sample grasp candidates and select the best one.

    Args:
        meshcat: Meshcat instance for visualization
        cloud: PointCloud with normals (the object point cloud in world frame)
        rng: numpy random generator
        num_candidates: Number of grasp candidates to sample
        num_to_draw: Number of top grasps to visualize
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
        if len(costs) > 250:
            break

    # Print rejection summary
    print(f"\n  Rejection summary:")
    print(f"    Normal pointing down: {reject_normal_down}")
    print(f"    Collision with environment: {reject_collision_env}")
    print(f"    Collision with point cloud: {reject_collision_cloud}")
    print(f"    Other/no valid orientation: {reject_other}")

    if len(costs) == 0:
        print("  No valid grasp candidates found!")
        return None, np.inf

    # Sort by cost and get best candidates
    indices = np.asarray(costs).argsort()

    # Draw top N grasps (or fewer if we have less)
    top_k_to_draw = min(num_to_draw, len(indices))
    if top_k_to_draw > 0:
        print(f"\n  Drawing top {top_k_to_draw} grasp candidates:")
        for rank in range(top_k_to_draw):
            idx = indices[rank]
            print(f"    {rank+1}. Cost: {costs[idx]:.3f}")
            draw_grasp_candidate(meshcat, X_Gs[idx], prefix=f"grasp_{rank+1}_best")

    best_idx = indices[0]
    return X_Gs[best_idx], costs[best_idx]


def compute_pregrasp_pose(grasp_pose: RigidTransform, offset_z: float = 0.3) -> RigidTransform:
    """
    Compute the pre-grasp pose from a grasp pose.
    
    The pre-grasp pose is positioned above the grasp with a STRAIGHT-DOWN
    orientation (gripper pointing directly down). This allows a clean vertical
    approach before rotating into the grasp orientation.

    Args:
        grasp_pose: The grasp pose (RigidTransform)
        offset_z: Height offset above grasp in meters (default: 0.3m = 30cm)

    Returns:
        X_pregrasp: The pre-grasp RigidTransform
    """
    grasp_pos = grasp_pose.translation()

    # Pre-grasp position: same x, y, but z + offset
    pre_grasp_pos = np.array([
        grasp_pos[0],       # Same X
        grasp_pos[1],       # Same Y
        grasp_pos[2] + offset_z  # Above the grasp
    ])

    # Pre-grasp orientation: Use the same orientation as q_home (known to be reachable)
    # This is approximately "straight down" with a slight forward tilt
    # RPY: [-103.8°, 0°, 90°] = [-1.8124, 0, 1.5708] radians
    R_straight_down = RollPitchYaw(-1.8124, 0, np.pi/2).ToRotationMatrix()

    # Create the pre-grasp RigidTransform
    X_pregrasp = RigidTransform(R_straight_down, pre_grasp_pos)

    return X_pregrasp


def sample_position_from_covariance(
    mean: np.ndarray,
    covariance: np.ndarray,
    rng: np.random.Generator = None,
    max_sigma: float = 2.0,
) -> np.ndarray:
    """
    Sample a 3D position from a truncated 2D Gaussian distribution (X-Y only).
    
    The X-Y position is sampled from a 2D Gaussian with the given 2x2 covariance.
    The Z position is kept fixed at the mean's Z value (bottle sits on bin floor).
    
    Samples are restricted to be within max_sigma standard deviations
    of the mean (measured by Mahalanobis distance in X-Y). If a sample falls
    outside this region, it is projected onto the ellipsoid boundary.

    Args:
        mean: 3D mean position [x, y, z] (Z is fixed)
        covariance: 2x2 covariance matrix (X-Y only) - flattened to 4 elements
        rng: numpy random generator (optional, uses default if None)
        max_sigma: Maximum Mahalanobis distance (default: 2.0 for ~95% probability region)

    Returns:
        sampled_position: 3D position with sampled X-Y and fixed Z
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Handle both flattened (4 elements) and 2x2 matrix inputs
    if covariance.size == 4:
        covariance_2d = covariance.reshape(2, 2)
    elif covariance.shape == (2, 2):
        covariance_2d = covariance
    else:
        raise ValueError(f"Expected 2x2 covariance, got shape {covariance.shape}")
    
    # Extract X-Y mean
    mean_xy = mean[:2]
    z_fixed = mean[2]
    
    # Ensure covariance is symmetric and positive semi-definite
    covariance_sym = (covariance_2d + covariance_2d.T) / 2
    
    # Add small regularization for numerical stability
    covariance_reg = covariance_sym + np.eye(2) * 1e-10
    
    # Sample X-Y from 2D Gaussian
    sample_xy = rng.multivariate_normal(mean_xy, covariance_reg)
    
    # Compute Mahalanobis distance in X-Y: d = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
    deviation_xy = sample_xy - mean_xy
    try:
        cov_inv = np.linalg.inv(covariance_reg)
        mahalanobis_dist = np.sqrt(deviation_xy @ cov_inv @ deviation_xy)
    except np.linalg.LinAlgError:
        # If inversion fails, use Euclidean distance scaled by trace
        mahalanobis_dist = np.linalg.norm(deviation_xy) / np.sqrt(np.trace(covariance_reg) / 2)
    
    # If sample is outside the max_sigma ellipse, project to boundary
    if mahalanobis_dist > max_sigma:
        # Scale deviation to lie on the ellipse boundary
        scale_factor = max_sigma / mahalanobis_dist
        sample_xy = mean_xy + deviation_xy * scale_factor
    
    # Return 3D position with sampled X-Y and fixed Z
    return np.array([sample_xy[0], sample_xy[1], z_fixed])
