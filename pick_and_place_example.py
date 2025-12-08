# %% [markdown]
# This notebook provides examples to go along with the [textbook](http://manipulation.csail.mit.edu/pose.html).  I recommend having both windows open, side-by-side!

# %%
import numpy as np
from pydrake.all import (
    AbstractValue,
    Concatenate,
    DiagramBuilder,
    LeafSystem,
    PiecewisePolynomial,
    PiecewisePose,
    PointCloud,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
)

from manipulation import FindResource, running_as_notebook
from manipulation.icp import IterativeClosestPoint
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.mustard_depth_camera_example import MustardPointCloud
from manipulation.pick import (
    MakeGripperCommandTrajectory,
    MakeGripperFrames,
    MakeGripperPoseTrajectory,
)
from manipulation.station import AddPointClouds, LoadScenario, MakeHardwareStation
from manipulation.systems import AddIiwaDifferentialIK

# %%
# Start the visualizer.
meshcat = StartMeshcat()

# %% [markdown]
# # Putting it all together
# 
# In the code above, we worked with a point cloud using functions.  To assemble this into a full-stack manipulation system, we need to specify the timing semantics of when those functions are called.  That's precisely what Drake's systems framework provides.  I've introduced two systems below:
# - `MustardIterativeClosestPoint` system that takes the camera inputs and outputs the pose estimate using ICP, and
# - `PickAndPlaceTrajectory` system that takes this pose estimate (and the state of the robot), computes the trajectory, and stores that trajectory in its Context so that it can output the instantaneous command.
# 
# We don't use a `TrajectorySource` here, because the trajectory is not known when we first build the Diagram... the information we need to plan the trajectory requires reading sensors at runtime.

# %%
# Takes 3 point clouds (in world coordinates) as input, and outputs and estimated pose for the mustard bottle.


class MustardIterativeClosestPoint(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        model_point_cloud = AbstractValue.Make(PointCloud(0))
        self.DeclareAbstractInputPort("cloud0", model_point_cloud)
        self.DeclareAbstractInputPort("cloud1", model_point_cloud)
        self.DeclareAbstractInputPort("cloud2", model_point_cloud)

        self.DeclareAbstractOutputPort(
            "X_WO",
            lambda: AbstractValue.Make(RigidTransform()),
            self.EstimatePose,
        )

        self.mustard = MustardPointCloud()
        meshcat.SetObject("icp_scene", self.mustard)

    def EstimatePose(self, context, output):
        pcd = []
        for i in range(3):
            cloud = self.get_input_port(i).Eval(context)
            pcd.append(
                cloud.Crop(lower_xyz=[0.4, -0.2, 0.001], upper_xyz=[0.6, 0.3, 0.3])
            )
        merged_pcd = Concatenate(pcd)
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
        meshcat.SetObject("icp_observations", down_sampled_pcd, point_size=0.001)

        X_WOhat, chat = IterativeClosestPoint(
            self.mustard.xyzs(),
            down_sampled_pcd.xyzs(),
            meshcat=meshcat,
            meshcat_scene_path="icp_scene",
        )

        output.set_value(X_WOhat)


class PickAndPlaceTrajectory(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareAbstractInputPort("X_WO", AbstractValue.Make(RigidTransform()))

        self.DeclareInitializationUnrestrictedUpdateEvent(self.Plan)
        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose())
        )
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )

        self.DeclareAbstractOutputPort(
            "X_WG",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose,
        )
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

    def Plan(self, context, state):
        X_G = {
            "initial": self.get_input_port(0).Eval(context)[
                int(self._gripper_body_index)
            ]
        }
        X_O = {
            "initial": self.get_input_port(1).Eval(context),
            "goal": RigidTransform([0, -0.6, 0]),
        }
        X_GgraspO = RigidTransform(RollPitchYaw(np.pi / 2, 0, 0), [0, 0.22, 0])
        X_OGgrasp = X_GgraspO.inverse()
        X_G["pick"] = X_O["initial"] @ X_OGgrasp
        X_G["place"] = X_O["goal"] @ X_OGgrasp
        X_G, times = MakeGripperFrames(X_G)
        print(f"Planned {times['postplace']} second trajectory.")

        if False:  # Useful for debugging
            AddMeshcatTriad(meshcat, "X_Oinitial", X_PT=X_O["initial"])
            AddMeshcatTriad(meshcat, "X_Gprepick", X_PT=X_G["prepick"])
            AddMeshcatTriad(meshcat, "X_Gpick", X_PT=X_G["pick"])
            AddMeshcatTriad(meshcat, "X_Gplace", X_PT=X_G["place"])

        traj_X_G = MakeGripperPoseTrajectory(X_G, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(
            traj_wsg_command
        )

    def start_time(self, context):
        return (
            context.get_abstract_state(int(self._traj_X_G_index))
            .get_value()
            .start_time()
        )

    def end_time(self, context):
        return (
            context.get_abstract_state(int(self._traj_X_G_index)).get_value().end_time()
        )

    def CalcGripperPose(self, context, output):
        # Evaluate the trajectory at the current time, and write it to the
        # output port.
        output.set_value(
            context.get_abstract_state(int(self._traj_X_G_index))
            .get_value()
            .GetPose(context.get_time())
        )

    def CalcWsgPosition(self, context, output):
        # Evaluate the trajectory at the current time, and write it to the
        # output port.
        output.SetFromVector(
            context.get_abstract_state(int(self._traj_wsg_index))
            .get_value()
            .value(context.get_time())
        )


def icp_pick_and_place_demo():
    builder = DiagramBuilder()

    scenario = LoadScenario(
        filename=FindResource("models/clutter.scenarios.yaml"),
        scenario_name="Mustard",
    )
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
    to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder)

    icp = builder.AddSystem(MustardIterativeClosestPoint())
    builder.Connect(to_point_cloud["camera3"].get_output_port(), icp.get_input_port(0))
    builder.Connect(to_point_cloud["camera4"].get_output_port(), icp.get_input_port(1))
    builder.Connect(to_point_cloud["camera5"].get_output_port(), icp.get_input_port(2))

    plant = station.GetSubsystemByName("plant")
    plan = builder.AddSystem(PickAndPlaceTrajectory(plant))
    builder.Connect(
        station.GetOutputPort("body_poses"), plan.GetInputPort("body_poses")
    )
    builder.Connect(icp.GetOutputPort("X_WO"), plan.GetInputPort("X_WO"))

    robot = station.GetSubsystemByName("iiwa_controller_plant_pointer_system").get()

    # Set up differential inverse kinematics.
    diff_ik = AddIiwaDifferentialIK(builder, robot, frame=plant.GetFrameByName("body"))
    builder.Connect(diff_ik.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(plan.GetOutputPort("X_WG"), diff_ik.get_input_port(0))
    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        diff_ik.GetInputPort("robot_state"),
    )

    builder.Connect(
        plan.GetOutputPort("wsg_position"),
        station.GetInputPort("wsg.position"),
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_context()

    simulator.Initialize()
    if False:  # draw the trajectory triads
        X_G_traj = plan.GetMyContextFromRoot(context).get_abstract_state(0).get_value()
        for t in np.linspace(X_G_traj.start_time(), X_G_traj.end_time(), 40):
            AddMeshcatTriad(
                meshcat,
                f"X_G/({t})",
                X_PT=X_G_traj.GetPose(t),
                length=0.1,
                radius=0.004,
            )

    if running_as_notebook:
        meshcat.StartRecording(set_visualizations_while_recording=True)
        simulator.AdvanceTo(plan.end_time(plan.GetMyContextFromRoot(context)))
        meshcat.PublishRecording()
    else:
        simulator.AdvanceTo(0.1)


icp_pick_and_place_demo()

# %%


# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=92fdbe38-dfb3-48cf-9b46-1660ec071b29' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>


