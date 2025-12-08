# %% [markdown]
# # Sampling Grasp Meshes
# 
# In previous chapters, we manually specified grasps for the objects our robot was manipulating. This is fine when you are only manipulating a fixed object in a fixed scene, but we want our robots to be able to manipulate all sorts of objects in scenes we have not seen before. In this notebook, we will build familiarity with methods for generating grasp poses from arbitary meshes. 
# 
# 
# **Learning Objectives:**
# 1. Antipodal grasp sampling on a mesh
# 2. Heuristic design for grasp filtering
# 
# **What you'll build:** A simulation of the IIWA grasping and reorienting meshes corresponding to your initials. 
# 
# **Reference:** Make sure you understand the full grasp sampling demo in [Chapter 5](https://manipulation.mit.edu/clutter.html#grasp_sampling), many of the same principles apply. It will also be helpful, but not necessary, to have solved [Exercise 4.11](https://manipulation.mit.edu/pose.html#exercises).
# 
# Your end result will look something like this, where the letters are spawned laying down on the table, and the iiwa will pick them up from above:
# 
# ![geometry_pick_and_place_point_clouds.png](https://raw.githubusercontent.com/RussTedrake/manipulation/master/book/figures/exercises/clutter_sampling_grasps_letters.png
# )
# 
# Let's start by getting our imports out of the way and launching Meshcat. 

# %%
import os
import random
from pathlib import Path
from typing import List, Tuple

import mpld3
import numpy as np
import trimesh
from pydrake.all import (
    AddFrameTriadIllustration,
    BasicVector,
    Context,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    LeafSystem,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    TrajectorySource,
)

from manipulation import running_as_notebook
from manipulation.exercises.clutter.test_grasp_letters import TestLetterGrasp
from manipulation.exercises.grader import Grader
from manipulation.letter_generation import create_sdf_asset_from_letter
from manipulation.station import LoadScenario, MakeHardwareStation

if running_as_notebook:
    mpld3.enable_notebook()

# Start the visualizer.
meshcat = StartMeshcat()

# %% [markdown]
# # Mesh Pre-Processing
# 
# The first step will be load in the geometry of the part we are manipulating. Because the focus of this excercise is on grasp sampling, we will assume access to the ground truth pose of the part on the table and its geometry.

# %%
# TODO fill in your initials here.
initials = None

# %%
create_sdf_asset_from_letter(
    text=initials[0],
    font_name="DejaVu Sans",
    letter_height_meters=0.25,
    extrusion_depth_meters=0.07,
    output_dir="assets",
    include_normals=True,
    mu_static=1.17,
    mass=0.1,
)
create_sdf_asset_from_letter(
    text=initials[1],
    font_name="DejaVu Sans",
    letter_height_meters=0.25,
    extrusion_depth_meters=0.07,
    output_dir="assets",
    include_normals=True,
    mu_static=1.17,
    mass=0.1,
)

# %%
# TODO: load your first initial with trimesh.load(...) as a mesh.
# To do this, you should make sure to use the kwargs force="mesh".
# See the docs for more info at https://trimesh.org/. (see exercise 4.1)


def load_first_initial() -> trimesh.Trimesh:
    return None

# %% [markdown]
# # Grasp Sampling
# 
# The next task will be to find candidate grasps. We are looking for collinear, antipodal points that can fit within the width of the gripper and do not put the gripper in collision. From these points, we can define gripper poses that we command the robot to achieve. We will break this into three steps. 
# 
# 1. Finding Pairs of Collinear Points via ray casting
# 2. Taking a pair of colinear points and using them to compute a gripper pose
# 2. Filtering grasps on antipodality, finger width, and collision
# 
# **Reference:** You will need to call the following functions from `trimesh` when sampling colinear points:
# 
# - [sample_surface](https://trimesh.org/trimesh.sample.html#trimesh.sample.sample_surface)
# 
# - [intersects_location](https://trimesh.org/trimesh.ray.ray_pyembree.html#trimesh.ray.ray_pyembree.RayMeshIntersector.intersects_location)
# 
# And constructing the grasp transform from a point and its normal is demonstrated in the demo from [Example 5.12 in the textbook](https://manipulation.mit.edu/clutter.html#grasp_sampling).

# %%
# Structure: (point_1, point_2, normal_1, normal_2)
AntipodeCandidateType = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def sample_colinear_points(
    mesh: trimesh.Trimesh, n_sample_points: int
) -> List[AntipodeCandidateType]:
    """
    Compute n_sample_points point pairs for the mesh that are colinear.
    This is done by sampling points from the surface and casting a ray along the normal vector of that point
    until another point on the mesh surface is hit. This function returns
    a list of length n_sample_points of tuples, with every tuple having the structure:
    (point_1, point_2, normal_1, normal_2)

    Note that the returned points will be expressed in the object frame.
    """
    candidates = []

    # TODO: sample `n_sample_points` from the surface of the mesh

    # TODO: index into the mesh `face_normals` to get the normals q for all the sampled points

    for i in range(n_sample_points):
        # TODO: check for the first point hit while traversing a ray starting at the sampled point
        # and moving along the negative of the normal vector

        # TODO: if no hits are found, skip this point

        # TODO: get the `face_normal` from the mesh corresponding to the colinear point.

        # TODO: add the tuple of the two points and their normals to the `candidates` list
        pass  # TODO: delete this when the for loop is implemented
    return candidates

# %%
def compute_grasp_from_points(
    antipodal_pt: AntipodeCandidateType,
) -> RigidTransform | None:
    """
    Given the tuple of antipodal points and their normals on the object O, compute the grasp X_OG.
    """
    z_axis_O = np.array([0.0, 0.0, 1.0])

    # TODO: the x-axis of frame G is given by the normal of the sampled point in frame O

    # TODO: if the x-axis of frame G is parallel to the frame O z-axis, return None
    # TODO: the y-axis of frame G should point along the -z axis of the O, such
    #       that we pick up the object from above.
    #       The x and y axis have to be orthogonal, so project. (Hint: Gram-Schmidt)

    # TODO: the z-axis of frame G is orthogonal to the x- and y-axis of frame G

    # TODO: construct the rotation matrix R_OG

    # TODO: define p_OG_O by computing the median of the two colinear points.

    # TODO: define the transform X_OG, then add an offset of -0.1m in the y-axis to account for finger length
    # return the resulting transform
    return RigidTransform()

# %%
def check_collision_free(X_WG: RigidTransform) -> bool:
    """
    Checks if the gripper collides with the table. The table can be represented as a flat plane centered at the
    origin with a normal vector pointing in the z-axis in world coordinates.
    params:
        X_WG (ndarray): (4x4) matrix describing gripper frame in the world coordinates
    returns:
        True if the gripper is in a collision free pose, False otherwise
    """
    gripper_vertices = np.array([[-0.073, -0.085383, -0.025], [0.073, 0.069, 0.025]])
    # vertices modeling the gripper collision body in homogenous coordinates
    verts_h = np.hstack((gripper_vertices, np.ones((gripper_vertices.shape[0], 1))))

    # TODO: map the gripper vertices to the world frame

    # TODO: the gripper is collision free if all the vertices are above z=0.
    # return true if collision free, false otherwise

# %%
def get_filtered_grasps(
    candidate_list: List[AntipodeCandidateType],
    antipodal_thresh: float,
    z_axis_thresh: float,
    max_pt_dist: float,
    min_pt_dist: float,
    X_WO: RigidTransform,
) -> List[RigidTransform]:
    """
    Return a list of grasps filtered on the following criteria
    (1) Antipodality: antipodality is a good heuristic for finding grasps with a large total wrench cone
    (2) Point Distance: pairs of points too far apart won't fit inside the gripper.
        Points too close together are "false positives", that appear due to the numerics of the ray casting.
    (3) Collision:
    """
    filtered_candidates = []
    for candidate in candidate_list:
        # TODO: compute the dot product of the normals.
        # If the points are roughly antipodal, their dot product will be less than the antipodal_thresh.

        # TODO: compute the distance between the point pairs. check that it is between
        # max_pt_dist and min_pt_dist.

        # TODO: compute the grasp corresponding to the candidate list
        # if the grasp computation fails (returns None), then `continue` to the next candidate

        # TODO: map the grasp to the world frame and check if it is collision free.

        # TODO: If a candidate passes all three checks, add the grasp in world-frame to `filtered_candidates`
        pass  # delete this when the for loop is implemented
    return filtered_candidates

# %%
def sample_grasp(
    mesh: trimesh.Trimesh, X_WO: RigidTransform, n_sample_pts: int = 500
) -> RigidTransform:
    colinear_pts = sample_colinear_points(mesh, n_sample_points=n_sample_pts)
    candidate_grasps = get_filtered_grasps(
        colinear_pts,
        antipodal_thresh=-0.95,
        z_axis_thresh=0.8,
        max_pt_dist=0.04,
        min_pt_dist=0.005,
        X_WO=X_WO,
    )
    return candidate_grasps[0]

# %%
def compute_prepick_pose(X_WG: RigidTransform) -> RigidTransform:
    X_GGprepick = RigidTransform([0, -0.17, 0.0])
    return X_WG @ X_GGprepick

# %% [markdown]
# # Building the Diagram
# 
# The next few steps should look familiar. We will define a jacobian pseudo-inverse based controller, and define a yaml with all the geometries in our scene. The last step will be to use the grasps to define a robot trajectory.
# 
# **Fill out the keyframes so that the robot starts at the initial pose. Then it:**
# 
# **(1) goes to a randomly sampled grasp.**
# 
# **(2) rotates the letter 30 degrees clockwise about the gripper y-axis.**
# 
# When the robot is done, the first and second initial should have the same orientation.
# If you find that the robot's fingers are bumping into the letter on its' way to manipulate it, try adjusting the `opened` constant, which controls the finger width when the robot is not grasping something. It may be helpful to go to a pre-pick pose before step 1 and step 2.

# %%
class PseudoInverseController(LeafSystem):
    def __init__(self, plant: MultibodyPlant) -> None:
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context: Context, output: BasicVector) -> None:
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kV,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )
        J_G = J_G[:, self.iiwa_start : self.iiwa_end + 1]  # Only iiwa terms.
        v = np.linalg.pinv(J_G).dot(V_G)
        output.SetFromVector(v)

# %%
scenario_yaml = f"""directives:
    - add_model:
        name: iiwa
        file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
        default_joint_positions:
            iiwa_joint_1: [-1.57]
            iiwa_joint_2: [0.1]
            iiwa_joint_3: [0]
            iiwa_joint_4: [-1.2]
            iiwa_joint_5: [0]
            iiwa_joint_6: [ 1.6]
            iiwa_joint_7: [0]
    - add_weld:
        parent: world
        child: iiwa::iiwa_link_0
        X_PC:
            translation: [0, -0.5, 0]
            rotation: !Rpy {{ deg: [0, 0, 180] }}
    - add_model:
        name: wsg
        file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
    - add_weld:
        parent: iiwa::iiwa_link_7
        child: wsg::body
        X_PC:
            translation: [0, 0, 0.09]
            rotation: !Rpy {{deg: [90, 0, 90]}}
    - add_model:
        name: table
        file: package://manipulation/table.sdf
    - add_weld:
        parent: world
        child: table::table_link
        X_PC:
            translation: [0.0, 0.0, -0.05]
            rotation: !Rpy {{ deg: [0, 0, -90] }}
    - add_model:
        name: {initials[0]}_letter
        file: file://{Path.cwd()}/assets/{initials[0]}.sdf
        default_free_body_pose:
            {initials[0]}_body_link:
                translation: [-0.2, 0, 0]
                rotation: !Rpy {{ deg: [0, 0, 30] }}
    - add_model:
        name: {initials[1]}_letter
        file: file://{Path.cwd()}/assets/{initials[1]}.sdf
        default_free_body_pose:
            {initials[1]}_body_link:
                translation: [0.25, 0, 0]
                rotation: !Rpy {{ deg: [0, 0, 0] }}
model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg
    wsg: !SchunkWsgDriver {{}}
"""
with open("scene.yaml", "w") as f:
    f.write(scenario_yaml)

# %%
station = MakeHardwareStation(LoadScenario(filename="scene.yaml"), meshcat=meshcat)
builder = DiagramBuilder()
builder.AddSystem(station)

plant = station.GetSubsystemByName("plant")
temp_context = station.CreateDefaultContext()
temp_plant_context = plant.GetMyContextFromRoot(temp_context)
X_WGinitial = plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName("body"))

model_instance0 = plant.GetModelInstanceByName(f"{initials[0]}_letter")
model_instance1 = plant.GetModelInstanceByName(f"{initials[1]}_letter")
X_WO1initial = plant.EvalBodyPoseInWorld(
    temp_plant_context, plant.GetBodyByName(f"{initials[0]}_body_link", model_instance0)
)

opened = 0.04
closed = 0.0

# TODO: redefine `keyframes` so the robot performs the behavior described above.
# `keyframes` is a list of 2-tuples. The first element in each tuple is a
# gripper pose in the world frame. the second element is a float corresponding to the wsg position.
# you can use `sample_grasp` to get a grasp pose.
# a helper function to go to prepick poses has been provided (`compute_prepick_pose`).
keyframes = [(X_WGinitial, opened), (X_WGinitial, closed)]

sample_times = [3 * i for i in range(len(keyframes))]
robot_position_trajectory = PiecewisePose.MakeLinear(
    sample_times, [kf[0] for kf in keyframes]
)
traj_V_G = robot_position_trajectory.MakeDerivative()
gripper_values = np.array([kf[1] for kf in keyframes])[None]
traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, gripper_values)
V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
controller = builder.AddSystem(PseudoInverseController(plant))
integrator = builder.AddSystem(Integrator(7))
wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))

builder.Connect(V_G_source.get_output_port(), controller.GetInputPort("V_WG"))
builder.Connect(controller.get_output_port(), integrator.get_input_port())
builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa.position"))
builder.Connect(
    station.GetOutputPort("iiwa.position_measured"),
    controller.GetInputPort("iiwa.position"),
)

# visualize axes (useful for debugging)
scenegraph = station.GetSubsystemByName("scene_graph")
AddFrameTriadIllustration(
    scene_graph=scenegraph,
    body=plant.GetBodyByName(f"{initials[0]}_body_link", model_instance0),
    length=0.1,
)
AddFrameTriadIllustration(
    scene_graph=scenegraph,
    body=plant.GetBodyByName(f"{initials[1]}_body_link", model_instance1),
    length=0.1,
)
AddFrameTriadIllustration(
    scene_graph=scenegraph, body=plant.GetBodyByName("body"), length=0.1
)

builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))
diagram = builder.Build()

# %%
# Define the simulator.
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
station_context = station.GetMyContextFromRoot(context)
integrator.set_integral_value(
    integrator.GetMyContextFromRoot(context),
    plant.GetPositions(
        plant.GetMyContextFromRoot(context),
        plant.GetModelInstanceByName("iiwa"),
    ),
)
diagram.ForcedPublish(context)
print(f"sanity check, simulation will run for {traj_V_G.end_time()} seconds")

# run simulation!
meshcat.StartRecording()
if running_as_notebook:
    simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(traj_V_G.end_time())
meshcat.StopRecording()
meshcat.PublishRecording()

# %% [markdown]
# # Gradescope Verification
# 
# Take a video of the trajectory and upload it to gradescope as an mp4, the file should be (much) smaller than 500MB. The robot should grasp the first initial using an antipodal grasp and rotate is so it has the same orientation as the second initial. Optionally, consider adding more advanced heuristics, like checking for collision between the gripper and the mesh in the pick pose as is done in chapter 5. 

# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=094905d9-dcb3-4fa9-9be0-1d0e62891bc4' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>


