# %% [markdown]
# ## Pose Estimation with ICP

# %%
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BaseField,
    DiagramBuilder,
    Fields,
    MeshcatVisualizer,
    Parser,
    PointCloud,
    Rgba,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    StartMeshcat,
)

from manipulation import running_as_notebook
from manipulation.exercises.grader import Grader
from manipulation.exercises.pose.test_pose_estimation import TestPoseEstimation
from manipulation.icp import IterativeClosestPoint
from manipulation.scenarios import AddMultibodyTriad
from manipulation.station import AddPointClouds, LoadScenario, MakeHardwareStation

# %%
# Start the visualizer.
meshcat = StartMeshcat()

# %% [markdown]
# ## Problem Description
# Last lecture, we designed pick and place trajectories **assuming** that the object pose ${}^W X^O$ was known. With all the tools we have learned for goemetric perception, it is time to relax this assumption and finally do pose estimation from sensor data.
# 
# The goal of the exercise is to give you some real-world experience into what dealing with depth cameras, and what it takes to go from a real depth image to the clean ICP formulation we learned.
# 
# **These are the main steps of the exercise:**
# 1. Perform Segmentation on the raw pointcloud of the scene to extract pointcloud from the object.
# 2. Tune an off-the-shelf ICP solver and estimate the pose of the object.

# %% [markdown]
# Before jumping into the main exercise, how should we computationally represent a pointcloud? If we say that pointcloud has $N$ points, then each point has a position in 3D, ${}^Cp^i$, as well as an associated color. Throughout this exercise, we will tend to store them as separate arrays of:
# - `3xN` numpy array where each row stores the XYZ position of the point in meters.
# - `3xN` numpy array where each row stores the RGB information of the point in `uint8` format.
# 
# Unfortunately, numpy prefers a rowwise representation, so you might find yourself using the `.T` transpose operator to make numpy operations more natural/efficient.

# %%
def ToPointCloud(xyzs, rgbs=None):
    if rgbs:
        cloud = PointCloud(xyzs.shape[1], Fields(BaseField.kXYZs | BaseField.kRGBs))
        cloud.mutable_rgbs()[:] = rgbs
    else:
        cloud = PointCloud(xyzs.shape[1])
    cloud.mutable_xyzs()[:] = xyzs
    return cloud

# %% [markdown]
# ## Getting a Pointcloud of the Model ##
# 
# Before taking a pointcloud of the **scene**, we will need a pointcloud of the **model** to compare against. Generally, this can be done by using existing tools that convert 3D representations (meshes, signed distance functions, etc.) into pointclouds.
# 
# Since our red foam brick is of rectangular shape, we'll cheat a bit and generate the points procedurally. When you click the cell below, you should be able to see the red brick and our pointcloud representation of the brick as blue dots.
# 
# We will save the model pointcloud in the variable `model_pcl_np`.

# %%
def visualize_red_foam_brick():
    """
    Visualize red foam brick in Meshcat.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.AddModelsFromUrl(
        "package://drake_models/manipulation_station/061_foam_brick.sdf"
    )
    AddMultibodyTriad(plant.GetFrameByName("base_link"), scene_graph)
    plant.Finalize()

    # Setup Meshcat
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)


def generate_model_pointcloud(xrange, yrange, zrange, res):
    """
    Procedurally generate pointcloud of a rectangle for each side.
    """
    # Decide on how many samples
    x_lst = np.linspace(xrange[0], xrange[1], int((xrange[1] - xrange[0]) / res))
    y_lst = np.linspace(yrange[0], yrange[1], int((yrange[1] - yrange[0]) / res))
    z_lst = np.linspace(zrange[0], zrange[1], int((zrange[1] - zrange[0]) / res))

    pcl_lst = []
    # Do XY Plane
    for x in x_lst:
        for y in y_lst:
            pcl_lst.append([x, y, zrange[0]])
            pcl_lst.append([x, y, zrange[1]])

    # Do YZ Plane
    for y in y_lst:
        for z in z_lst:
            pcl_lst.append([xrange[0], y, z])
            pcl_lst.append([xrange[1], y, z])

    # Do XZ Plane
    for x in x_lst:
        for z in z_lst:
            pcl_lst.append([x, yrange[0], z])
            pcl_lst.append([x, yrange[1], z])

    return np.array(pcl_lst).T


visualize_red_foam_brick()
model_pcl_np = generate_model_pointcloud(
    [-0.0375, 0.0375], [-0.025, 0.025], [0.0, 0.05], 0.002
)
# meshcat.SetObject("pcl_model", ToPointCloud(model_pcl_np), rgba=Rgba(0, 0, 1, 1))

# %% [markdown]
# ## Getting the Scene Pointcloud
# 
# Now let's set up the ClutteringStation from last lecture and actually take a pointcloud snapshot of the scene with the `red_foam_brick`. We'll place the camera where we have good coverage of the bin. We'll also take a pointcloud snapshot without the `red_foam_brick` so that we can use it for segmentation later.
# 
# NOTE: There are around `3e7` points that are trying to be published to the visualizer, so things might load slowly, and occasionally the Colab session might crash. Keep calm and run the cells from the beginning!

# %%
meshcat.Delete()


def setup_clutter_station(with_brick=True):
    builder = DiagramBuilder()

    scenario_data = """
directives:
- add_model:
    name: bin0
    file: package://manipulation/hydro/bin.sdf

- add_weld:
    parent: world
    child: bin0::bin_base
    X_PC:
      rotation: !Rpy { deg: [0.0, 0.0, 90.0 ]}
      translation: [-0.145, -0.63, 0.075]

- add_model:
    name: bin1
    file: package://manipulation/hydro/bin.sdf

- add_weld:
    parent: world
    child: bin1::bin_base
    X_PC:
      rotation: !Rpy { deg: [0.0, 0.0, 180.0 ]}
      translation: [0.5, -0.1, 0.075]
"""
    if with_brick:
        scenario_data += """
- add_model:
    name: brick
    file: package://manipulation/hydro/061_foam_brick.sdf
    default_free_body_pose:
        base_link:
            translation: [-0.1, -0.6, 0.09]
            rotation: !Rpy { deg: [0, 0, 18] }    
"""
    scenario_data += """
- add_model:
    name: camera
    file: package://manipulation/camera_box.sdf
- add_weld:
    parent: world
    child: camera::base
    X_PC:
        translation: [-0.1, -0.8, 0.5]
        rotation: !Rpy { deg: [-150, 0, 0] }
cameras:
    main_camera:
        name: camera0
        depth: True
        X_PB:
            base_frame: camera::base
"""

    scenario = LoadScenario(data=scenario_data)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")
    AddMultibodyTriad(
        plant.GetFrameByName("base", plant.GetModelInstanceByName("camera")),
        scene_graph,
    )

    # Send the point cloud to meshcat for visualization, too.
    to_point_cloud = AddPointClouds(
        scenario=scenario, station=station, builder=builder, meshcat=meshcat
    )
    if isinstance(to_point_cloud, list):
        # TODO(russt): Remove this after Fall 2023 pset 4 is safely wrapped up
        builder.ExportOutput(to_point_cloud[0].get_output_port(), "camera_point_cloud")
    else:
        builder.ExportOutput(
            to_point_cloud["camera0"].get_output_port(), "camera_point_cloud"
        )

    diagram = builder.Build()
    diagram.set_name("clutter_station")
    return diagram


# Take a pointcloud snapshot of the background to use for subtraction
diagram = setup_clutter_station(with_brick=False)
context = diagram.CreateDefaultContext()
diagram.ForcedPublish(context)
# Note: The use of Crop here removes non-finite returns, and also makes a copy of
# the data, which is important since the diagram that owns it will be garbage
# collected.
scene_pcl_drake_background = (
    diagram.GetOutputPort("camera_point_cloud")
    .Eval(context)
    .Crop(lower_xyz=[-5, -5, -5], upper_xyz=[5, 5, 5])
)

# Take a pointcloud snapshot of the scene with the brick.
diagram = setup_clutter_station(with_brick=True)
context = diagram.CreateDefaultContext()
diagram.ForcedPublish(context)
scene_pcl_drake = (
    diagram.GetOutputPort("camera_point_cloud")
    .Eval(context)
    .Crop(lower_xyz=[-5, -5, -5], upper_xyz=[5, 5, 5])
)

plant = diagram.GetSubsystemByName("station").GetSubsystemByName("plant")
plant_context = plant.GetMyContextFromRoot(context)
X_WO = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("base_link"))

# %% [markdown]
# ## Visualizing the Problem ##
# 
# That was a lot of work, but if you run the below cell, Meshcat will finally show you a clean formulation of the main problem. We have 3 pointcloud objects in Meshcat:
# 
# - `pcl_model`: Pointcloud of models
# - `pcl_scene`: Raw pointcloud of the foam-brick scene obtained from a RGBD camera.
# - `pcl_scene_background`: Raw pointcloud of the background obtained from a RGBD camera.
# 
# In case you forgot, In Meshcat's menu you can go into the `meshcat` tab and turn different objects on and off so that you can see what the background pointcloud looks like as well.
# 
# NOTE: You might have to wait a bit until the bin pointcloud shows up.
# 
# 

# %%
meshcat.Delete()

# meshcat.SetObject("pcl_model", ToPointCloud(model_pcl_np), rgba=Rgba(0, 0, 1, 1))
meshcat.SetObject("pcl_scene", scene_pcl_drake)
meshcat.SetObject("pcl_scene_background", scene_pcl_drake_background)

# %% [markdown]
# If we simply run ICP with `pcl_model` and `pcl_scene`, we might get a terrible result because there might be features in the background that the model is trying to run correspondence with. So we'd like to vet the problem a bit and perform **segmentation**: which parts of the scene pointcloud corresponds to an actual point on the `red_foam_brick`?
# 
# 
# **Now it's your turn to code!**
# 
# Below, you will implement a function `segment_scene_pcl` that takes in a pointcloud of the scene and return the relevant points that are actually on the `red_foam_brick`. But here are the rules of the game:
# - You **may** use color data, the background pointcloud, and any outlier detection algorithm that you can write to perform segmentation.
# - You may **not** explicitly impose conditions on the position to filter out the data. Remember that our goal is to estimate the pose in the first place, so using position will be considered cheating.
# - You may **not** use external libraries that are not in this notebook already.
# 
# In order to get full score for this assignment, you need to satisfy both criteria:
# - The number of false outliers (points which are not on the red brick but was caught by segmentation) must not exceed 80 points.
# - The number of missed inliers (points that are on the red brick but was not caught by segmentation) must not exceed 80 points.
# 
# You will be able to visualize your segmented pointclouds on Meshcat by running the cell.

# %%
def segment_scene_pcl(
    scene_pcl_np,
    scene_rgb_np,
    scene_pcl_np_background,
    scene_rgb_np_background,
):
    # 1. Background subtraction via voxel grid
    voxel_size = 0.003

    def voxel_indices(xyz, v):
        return np.floor(xyz / v).astype(np.int32)

    bg_idx = voxel_indices(scene_pcl_np_background, voxel_size)
    sc_idx = voxel_indices(scene_pcl_np, voxel_size)

    bg_voxels = set(map(tuple, bg_idx.T))
    mask_fg = np.array([tuple(i) not in bg_voxels for i in sc_idx.T])

    xyz_fg = scene_pcl_np[:, mask_fg]
    rgb_fg = scene_rgb_np[:, mask_fg]

    # 2. Color filtering for red brick
    rgb = rgb_fg.astype(np.float32)
    r, g, b = rgb[0], rgb[1], rgb[2]
    sum_rgb = r + g + b + 1e-6
    r_frac = r / sum_rgb
    g_frac = g / sum_rgb
    b_frac = b / sum_rgb

    mask_red = (r_frac > 0.55) & (g_frac < 0.30) & (b_frac < 0.30)
    xyz_red = xyz_fg[:, mask_red]

    if xyz_red.shape[1] == 0:
        return xyz_red  # nothing found

    # 3. Geometric outlier removal: keep main cluster
    center = np.mean(xyz_red, axis=1, keepdims=True)
    diff = xyz_red - center
    d2 = np.sum(diff**2, axis=0)
    thr = np.quantile(d2, 0.95)  # or tune this
    mask_cluster = d2 < thr

    scene_pcl_np_filtered = xyz_red[:, mask_cluster]
    return scene_pcl_np_filtered


scene_pcl_np_filtered = segment_scene_pcl(
    scene_pcl_drake.xyzs(),
    scene_pcl_drake.rgbs(),
    scene_pcl_drake_background.xyzs(),
    scene_pcl_drake_background.rgbs(),
)
meshcat.SetObject(
    "pcl_scene_filtered",
    ToPointCloud(scene_pcl_np_filtered),
    rgba=Rgba(0, 1, 0, 1),
)

# %% [markdown]
# ## ICP for Pose Estimation
# 
# Now that we have a subset of scene points that we want to use to estimate the pose, let's do ICP to figure out what ${}^W X^O$ is. Instead of implementing your own ICP this time, we will use the version we developed in the chapter notes.
# 
# We know that ICP can't work very well without even a rough initialization. Let's assume that we at least know that the `red_foam_brick` is inside the bin, so that we can initialize the ${}^W X^O$ to be at the center of the bin with an identity rotation.

# %%
initial_guess = RigidTransform()
initial_guess.set_translation([-0.145, -0.63, 0.09])
initial_guess.set_rotation(RotationMatrix.MakeZRotation(np.pi / 2))

# %% [markdown]
# Let's run the algorithm on your processed point cloud and see how we do!
# 

# %%
X_MS_hat, chat = IterativeClosestPoint(
    p_Om=model_pcl_np,
    p_Ws=scene_pcl_np_filtered,
    X_Ohat=initial_guess,
    meshcat=meshcat,
    meshcat_scene_path="icp",
    max_iterations=25 if running_as_notebook else 2,
)
meshcat.SetObject("pcl_estimated", ToPointCloud(model_pcl_np), rgba=Rgba(1, 0, 1, 1))
meshcat.SetTransform("pcl_estimated", X_MS_hat)

np.set_printoptions(precision=3, suppress=True)
X_OOhat = X_MS_hat.inverse().multiply(X_WO)

rpy = RollPitchYaw(X_OOhat.rotation()).vector()
xyz = X_OOhat.translation()

print("RPY Error: " + str(rpy))
print("XYZ Error: " + str(xyz))

# %% [markdown]
# ## How will this notebook be Graded?
# 
# If you are enrolled in the class, this notebook will be graded using [Gradescope](www.gradescope.com). You should have gotten the enrollement code on our announcement in Piazza.
# 
# For submission of this assignment, you must do as follows:.
# - Download and submit the notebook `pose_estimation_icp.ipynb` to Gradescope's notebook submission section, along with your notebook for the other problems.
# 
# We will evaluate the local functions in the notebook to see if the function behaves as we have expected. For this exercise, the rubric is as follows:
# - [4 pts] `segment_scene_pcl` correctly segments the scene by having less than 80 missed inliers and 80 false outliers.
# 
# Below is our autograder where you can check your score!

# %%
Grader.grade_output([TestPoseEstimation], [locals()], "results.json")
Grader.print_test_results("results.json")

# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=badeacdd-2130-4e60-a32d-8ae26d91817c' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>


