# import trimesh
from pydrake.all import (
    PointCloud,
    Rgba,
    RigidTransform,
    RotationMatrix,
    System,
    Diagram,
    Context,
    Meshcat,
)
from manipulation.icp import IterativeClosestPoint
from manipulation.mustard_depth_camera_example import MustardPointCloud
import numpy as np

N_SAMPLE_POINTS = 1500


def _remove_shelf_points(point_cloud: PointCloud) -> PointCloud:
    xyzs = point_cloud.mutable_xyzs()[:]
    mask = xyzs[2, :] > 0.01
    xyzs = xyzs[:, mask]

    new_point_cloud = PointCloud(xyzs.shape[1])
    new_point_cloud.mutable_xyzs()[:] = xyzs
    return new_point_cloud


def get_icp_estimated_pose(
    station: System,
    diagram: Diagram,
    context: Context,
    meshcat: Meshcat,
    visualize: bool = False,
) -> RigidTransform:
    point_cloud = MustardPointCloud()

    plant = station.plant()  # type: ignore
    plant_context = diagram.GetSubsystemContext(plant, context)

    world_frame = plant.world_frame()

    model_bottle = plant.GetModelInstanceByName("target_object")
    frame_bottle = plant.GetFrameByName(
        "target_object_body_link_mustard", model_instance=model_bottle
    )
    X_PC_bottle = plant.CalcRelativeTransform(plant_context, world_frame, frame_bottle)

    bottle_p = X_PC_bottle.translation()

    v = 0.3
    bottle_lower = bottle_p + np.array([-v, -v, -v])
    bottle_upper = bottle_p + np.array([v, v, v])

    camera_bottle_point_cloud = diagram.GetOutputPort(  # type: ignore
        "wrist_camera_sensor_point_cloud"
    ).Eval(context)

    camera_bottle_point_cloud: PointCloud = camera_bottle_point_cloud.Crop(
        lower_xyz=bottle_lower, upper_xyz=bottle_upper
    )

    if bottle_lower is not None and bottle_upper is not None and visualize:
        meshcat.SetLineSegments(
            "bounding_line",
            np.array(bottle_lower).T,  # type: ignore
            np.array(bottle_upper).T,  # type: ignore
            1.0,
            Rgba(0, 1, 0, 1),
        )

    down_sampled_pcd = camera_bottle_point_cloud.VoxelizedDownSample(voxel_size=0.005)

    down_sampled_pcd_no_shelf = _remove_shelf_points(down_sampled_pcd)

    if visualize:
        meshcat.SetObject(  # type: ignore
            "bottle_point_cloud",
            down_sampled_pcd_no_shelf,
            point_size=0.05,
            rgba=Rgba(1, 1, 0),
        )

    MAX_ITERATIONS = 100

    initial_guess = RigidTransform()
    initial_guess.set_translation([-0.35, 0.0, 0.0])  # type: ignore
    initial_guess.set_rotation(RotationMatrix.MakeXRotation(-np.pi / 4))

    model_xyzs = point_cloud.xyzs()
    data_xyzs = down_sampled_pcd_no_shelf.xyzs()

    X_bottle_Ohat, _ = IterativeClosestPoint(
        p_Om=model_xyzs,
        p_Ws=data_xyzs,
        X_Ohat=initial_guess,
        max_iterations=MAX_ITERATIONS,
    )

    return X_bottle_Ohat
