# %% [markdown]
# This notebook provides examples to go along with the [textbook](http://manipulation.csail.mit.edu/clutter.html).  I recommend having both windows open, side-by-side!

# %%
import numpy as np
from IPython.display import clear_output
from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    Concatenate,
    DiagramBuilder,
    JointSliders,
    LeafSystem,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PointCloud,
    RandomGenerator,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
)

from manipulation import running_as_notebook
from manipulation.scenarios import AddFloatingRpyJoint, AddRgbdSensors, ycb
from manipulation.utils import ConfigureParser

# %%
# Start the visualizer.
meshcat = StartMeshcat()

# %% [markdown]
# # Scoring grasp candidates

# %%
def GraspCandidateCost(
    diagram,
    context,
    cloud,
    wsg_body_index=None,
    plant_system_name="plant",
    scene_graph_system_name="scene_graph",
    adjust_X_G=False,
    verbose=False,
    meshcat_path=None,
):
    """
    Args:
        diagram: A diagram containing a MultibodyPlant+SceneGraph that contains
            a free body gripper and any obstacles in the environment that we
            want to check collisions against. It should not include the objects
            in the point cloud; those are handled separately.
        context: The diagram context.  All positions in the context will be
            held fixed *except* the gripper free body pose.
        cloud: a PointCloud in world coordinates which represents candidate
            grasps.
        wsg_body_index: The body index of the gripper in plant.  If None, then
            a body named "body" will be searched for in the plant.

    Returns:
        cost: The grasp cost

    If adjust_X_G is True, then it also updates the gripper pose in the plant
    context.
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

    # Crop to a region inside of the finger box.
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

    if meshcat_path:
        pc = PointCloud(np.sum(indices))
        pc.mutable_xyzs()[:] = cloud.xyzs()[:, indices]
        meshcat.SetObject("planning/points", pc, rgba=Rgba(1.0, 0, 0), point_size=0.01)

    if adjust_X_G and np.sum(indices) > 0:
        p_GC_x = p_GC[0, indices]
        p_Gcenter_x = (p_GC_x.min() + p_GC_x.max()) / 2.0
        X_G.set_translation(X_G @ np.array([p_Gcenter_x, 0, 0]))
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        X_GW = X_G.inverse()

    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)

    # Check collisions between the gripper and the sink
    if query_object.HasCollisions():
        cost = np.inf
        if verbose:
            print("Gripper is colliding with the sink!\n")
            print(f"cost: {cost}")
        return cost

    # Check collisions between the gripper and the point cloud
    # must be smaller than the margin used in the point cloud preprocessing.
    margin = 0.0
    for i in range(cloud.size()):
        distances = query_object.ComputeSignedDistanceToPoint(
            cloud.xyz(i), threshold=margin
        )
        if distances:
            cost = np.inf
            if verbose:
                print("Gripper is colliding with the point cloud!\n")
                print(f"cost: {cost}")
            return cost

    n_GC = X_GW.rotation().multiply(cloud.normals()[:, indices])

    # Penalize deviation of the gripper from vertical.
    # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
    cost = 20.0 * X_G.rotation().matrix()[2, 1]

    # Reward sum |dot product of normals with gripper x|^2
    cost -= np.sum(n_GC[0, :] ** 2)
    if verbose:
        print(f"cost: {cost}")
        print(f"normal terms: {n_GC[0,:]**2}")
    return cost


class ScoreSystem(LeafSystem):
    def __init__(self, diagram, cloud, wsg_pose_index):
        LeafSystem.__init__(self)
        self._diagram = diagram
        self._context = diagram.CreateDefaultContext()
        self._plant = diagram.GetSubsystemByName("plant")
        self._plant_context = self._plant.GetMyMutableContextFromRoot(self._context)
        wsg = self._plant.GetBodyByName("body")
        self._wsg_body_index = wsg.index()
        self._wsg_pose_index = wsg_pose_index
        self._cloud = cloud
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareForcedPublishEvent(self.Publish)

    def Publish(self, context):
        X_WG = self.get_input_port(0).Eval(context)[self._wsg_pose_index]
        self._plant.SetFreeBodyPose(
            self._plant_context,
            self._plant.get_body(self._wsg_body_index),
            X_WG,
        )
        GraspCandidateCost(
            self._diagram,
            self._context,
            self._cloud,
            verbose=True,
            meshcat_path="planning/cost",
        )
        clear_output(wait=True)


def process_point_cloud(diagram, context, cameras, bin_name):
    plant = diagram.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    # Compute crop box.
    bin_instance = plant.GetModelInstanceByName(bin_name)
    bin_body = plant.GetBodyByName("bin_base", bin_instance)
    X_B = plant.EvalBodyPoseInWorld(plant_context, bin_body)
    margin = 0.001  # only because simulation is perfect!
    a = X_B.multiply([-0.22 + 0.025 + margin, -0.29 + 0.025 + margin, 0.015 + margin])
    b = X_B.multiply([0.22 - 0.1 - margin, 0.29 - 0.025 - margin, 2.0])
    crop_min = np.minimum(a, b)
    crop_max = np.maximum(a, b)

    pcd = []
    for i in range(3):
        cloud = diagram.GetOutputPort(f"{cameras[i]}_point_cloud").Eval(context)

        # Crop to region of interest.
        pcd.append(cloud.Crop(lower_xyz=crop_min, upper_xyz=crop_max))
        # Estimate normals
        pcd[i].EstimateNormals(radius=0.1, num_closest=30)

        # Flip normals toward camera
        camera = plant.GetModelInstanceByName(f"camera{i}")
        body = plant.GetBodyByName("base", camera)
        X_C = plant.EvalBodyPoseInWorld(plant_context, body)
        pcd[i].FlipNormalsTowardPoint(X_C.translation())

    # Merge point clouds.
    merged_pcd = Concatenate(pcd)

    # Voxelize down-sample.  (Note that the normals still look reasonable)
    return merged_pcd.VoxelizedDownSample(voxel_size=0.005)


def make_environment_model(
    directive: str = None,
    draw: bool = False,
    rng: np.random.Generator = None,
    num_ycb_objects: int = 0,
    bin_name: str = "bin0",
):
    # Make one model of the environment, but the robot only gets to see the sensor outputs.
    if not directive:
        directive = "package://manipulation/two_bins_w_cameras.dmd.yaml"

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.SetAutoRenaming(True)
    parser.AddModelsFromUrl(directive)

    for i in range(num_ycb_objects):
        object_num = rng.integers(len(ycb))
        parser.AddModelsFromUrl(f"package://manipulation/hydro/{ycb[object_num]}")

    plant.Finalize()
    AddRgbdSensors(builder, plant, scene_graph)

    if draw:
        MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(prefix="environment"),
        )

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    if num_ycb_objects > 0:
        generator = RandomGenerator(rng.integers(1000))  # this is for c++
        plant_context = plant.GetMyContextFromRoot(context)
        bin_instance = plant.GetModelInstanceByName(bin_name)
        bin_body = plant.GetBodyByName("bin_base", bin_instance)
        X_B = plant.EvalBodyPoseInWorld(plant_context, bin_body)
        z = 0.2
        for body_index in plant.GetFloatingBaseBodies():
            tf = RigidTransform(
                UniformlyRandomRotationMatrix(generator),
                [rng.uniform(-0.15, 0.15), rng.uniform(-0.2, 0.2), z],
            )
            plant.SetFreeBodyPose(
                plant_context, plant.get_body(body_index), X_B.multiply(tf)
            )
            z += 0.1

        simulator = Simulator(diagram, context)
        simulator.AdvanceTo(2.0 if running_as_notebook else 0.1)
    elif draw:
        diagram.ForcedPublish(context)

    return diagram, context


# Another diagram for the objects the robot "knows about": gripper, cameras, bins.  Think of this as the model in the robot's head.
def make_internal_model():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/clutter_planning.dmd.yaml")
    plant.Finalize()
    return builder.Build()


def grasp_score_inspector():
    meshcat.Delete()
    environment, environment_context = make_environment_model(
        directive="package://manipulation/clutter_mustard.dmd.yaml", draw=True
    )

    internal_model = make_internal_model()

    # Finally, we'll build a diagram for running our visualization
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/clutter_planning.dmd.yaml")
    AddFloatingRpyJoint(
        plant,
        plant.GetFrameByName("body"),
        plant.GetModelInstanceByName("gripper"),
    )
    plant.Finalize()

    meshcat.DeleteAddedControls()
    params = MeshcatVisualizerParams()
    params.prefix = "planning"
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)

    cloud = process_point_cloud(
        environment,
        environment_context,
        ["camera0", "camera1", "camera2"],
        "bin0",
    )
    meshcat.SetObject("planning/cloud", cloud, point_size=0.003)

    score = builder.AddSystem(
        ScoreSystem(internal_model, cloud, plant.GetBodyByName("body").index())
    )
    builder.Connect(plant.get_body_poses_output_port(), score.get_input_port())

    lower_limit = [-1, -1, 0, -np.pi, -np.pi / 4.0, -np.pi / 4.0]
    upper_limit = [1, 1, 1, 0, np.pi / 4.0, np.pi / 4.0]
    q0 = [-0.05, -0.5, 0.25, -np.pi / 2.0, 0, 0]
    default_interactive_timeout = None if running_as_notebook else 1.0
    sliders = builder.AddSystem(
        JointSliders(
            meshcat,
            plant,
            initial_value=q0,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            decrement_keycodes=[
                "KeyQ",
                "KeyS",
                "KeyA",
                "KeyJ",
                "KeyK",
                "KeyU",
            ],
            increment_keycodes=[
                "KeyE",
                "KeyW",
                "KeyD",
                "KeyL",
                "KeyI",
                "KeyO",
            ],
        )
    )
    diagram = builder.Build()
    sliders.Run(diagram, default_interactive_timeout)
    meshcat.DeleteAddedControls()


grasp_score_inspector()

# %% [markdown]
# # Generating grasp candidates

# %%
def draw_grasp_candidate(X_G, prefix="gripper", draw_frames=True):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/schunk_wsg_50_welded_fingers.sdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body"), X_G)
    plant.Finalize()

    # frames_to_draw = {"gripper": {"body"}} if draw_frames else {}
    params = MeshcatVisualizerParams()
    params.prefix = prefix
    params.delete_prefix_on_initialization_event = False
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)


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
    Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel.
    The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].
    Args:
        diagram: A diagram containing a MultibodyPlant+SceneGraph that contains
            a free body gripper and any obstacles in the environment that we
            want to check collisions against. It should not include the objects
            in the point cloud; those are handled separately.
        context: The diagram context.  All positions in the context will be
            held fixed *except* the gripper free body pose.
        cloud: a PointCloud in world coordinates which represents candidate
            grasps.
        rng: a np.random.default_rng()
        wsg_body_index: The body index of the gripper in plant.  If None, then
            a body named "body" will be searched for in the plant.

    Returns:
        cost: The grasp cost
        X_G: The grasp candidate
    """
    plant = diagram.GetSubsystemByName(plant_system_name)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
    scene_graph.GetMyMutableContextFromRoot(context)
    if wsg_body_index:
        wsg = plant.get_body(wsg_body_index)
    else:
        wsg = plant.GetBodyByName("body")
        wsg_body_index = wsg.index()

    index = rng.integers(0, cloud.size() - 1)

    # Use S for sample point/frame.
    p_WS = cloud.xyz(index)
    n_WS = cloud.normal(index)

    assert np.isclose(
        np.linalg.norm(n_WS), 1.0
    ), f"Normal has magnitude: {np.linalg.norm(n_WS)}"

    Gx = n_WS  # gripper x axis aligns with normal
    # make orthonormal y axis, aligned with world down
    y = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(y, Gx)) < 1e-6:
        # normal was pointing straight down.  reject this sample.
        return np.inf, None

    Gy = y - np.dot(y, Gx) * Gx
    Gz = np.cross(Gx, Gy)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    p_GS_G = [0.054 - 0.01, 0.10625, 0]

    # Try orientations from the center out
    min_roll = -np.pi / 3.0
    max_roll = np.pi / 3.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    for theta in min_roll + (max_roll - min_roll) * alpha:
        # Rotate the object in the hand by a random rotation (around the normal).
        R_WG2 = R_WG.multiply(RotationMatrix.MakeXRotation(theta))

        # Use G for gripper frame.
        p_SG_W = -R_WG2.multiply(p_GS_G)
        p_WG = p_WS + p_SG_W

        X_G = RigidTransform(R_WG2, p_WG)
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        cost = GraspCandidateCost(diagram, context, cloud, adjust_X_G=True)
        X_G = plant.GetFreeBodyPose(plant_context, wsg)
        if np.isfinite(cost):
            return cost, X_G

        # draw_grasp_candidate(X_G, f"collision/{theta:.1f}")

    return np.inf, None


def sample_grasps_example():
    meshcat.Delete()
    rng = np.random.default_rng()

    environment, environment_context = make_environment_model(
        rng=rng, num_ycb_objects=5, draw=False
    )

    internal_model = make_internal_model()
    internal_model_context = internal_model.CreateDefaultContext()

    # Finally a model for the visualization.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/clutter_planning.dmd.yaml")
    plant.Finalize()

    params = MeshcatVisualizerParams()
    params.prefix = "planning"
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

    # Hide the planning gripper
    meshcat.SetProperty("planning/gripper", "visible", False)

    cloud = process_point_cloud(
        environment,
        environment_context,
        ["camera0", "camera1", "camera2"],
        "bin0",
    )
    meshcat.SetObject("planning/cloud", cloud, point_size=0.003)

    plant.GetMyContextFromRoot(context)
    scene_graph.GetMyContextFromRoot(context)

    costs = []
    X_Gs = []
    for i in range(100 if running_as_notebook else 2):
        cost, X_G = GenerateAntipodalGraspCandidate(
            internal_model, internal_model_context, cloud, rng
        )
        if np.isfinite(cost):
            costs.append(cost)
            X_Gs.append(X_G)

    indices = np.asarray(costs).argsort()[:5]
    for rank, index in enumerate(indices):
        draw_grasp_candidate(X_Gs[index], prefix=f"{rank}th best", draw_frames=False)


sample_grasps_example()

# %%


# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=094905d9-dcb3-4fa9-9be0-1d0e62891bc4' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>


