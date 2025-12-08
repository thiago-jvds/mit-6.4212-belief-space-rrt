import numpy as np
from pathlib import Path
import time
from pydrake.all import (
    DiagramBuilder,
    RigidTransform,
    RotationMatrix,
    MultibodyPlant,
    InverseKinematics,
    Solve,
    Meshcat,
    MeshcatParams,
)
from manipulation.station import LoadScenario, MakeHardwareStation
from manipulation.meshcat_utils import AddMeshcatTriad


def solve_ik_for_pose(
    plant: MultibodyPlant,
    X_WG_target: RigidTransform,
    q_nominal: tuple = tuple(
        np.array([0.0, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0])  # the initial joint positions
    ),
    theta_bound: float = 0.01,
    pos_tol: float = 0.015,
) -> tuple:
    """
    Solve IK for a single end-effector pose.

    Args:
        plant: A MultibodyPlant with the iiwa + gripper model.
        X_WG_target: Desired gripper pose in world frame.
        q_nominal: Nominal joint angles for joint-centering.
        theta_bound: Orientation tolerance (radians).
        pos_tol: Position tolerance (meters).

    Returns:
        q_solution: 7 element tuple representing the Optimal
        joint configuration. Each element of the tuple is a float.
    """
    world_frame = plant.world_frame()

    gripper_frame = plant.GetFrameByName("body")

    ik = InverseKinematics(plant)

    q_vars = ik.q()[:7]
    prog = ik.prog()

    ik.AddOrientationConstraint(
        frameAbar=gripper_frame,
        R_AbarA=RotationMatrix(),
        frameBbar=world_frame,
        R_BbarB=X_WG_target.rotation(),
        theta_bound=theta_bound,
    )

    ik.AddPositionConstraint(
        frameB=gripper_frame,
        p_BQ=[0, 0, 0],
        frameA=world_frame,
        p_AQ_lower=X_WG_target.translation() - pos_tol * np.array([1.0, 1.0, 1.0]),
        p_AQ_upper=X_WG_target.translation() + pos_tol * np.array([1.0, 1.0, 1.0]),
    )

    prog.AddQuadraticCost(np.identity(len(q_vars)), q_nominal, q_vars)

    prog.SetInitialGuess(q_vars, q_nominal)

    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError("IK did not succeed")

    return tuple(result.GetSolution(q_vars))

def main():
    print("=" * 50)
    print("Starting Meshcat on Port 7000...")
    try:
        params = MeshcatParams()
        params.port = 7000
        meshcat = Meshcat(params=params)
    except RuntimeError:
        meshcat = Meshcat()
    print(f"URL: {meshcat.web_url()}")
    print("=" * 50)

    project_root = Path(__file__).resolve().parents[2]
    scenario_path = project_root / "config" / "scenario.yaml"

    with open(scenario_path, "r") as f:
        scenario = LoadScenario(data=f.read())

    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario=scenario, meshcat=meshcat))

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    print("Visualizing Initial State...")
    diagram.ForcedPublish(context)

    mustard_body = plant.GetBodyByName("base_link_mustard")
    X_WMustard = plant.EvalBodyPoseInWorld(plant_context, mustard_body)

    p_Mustard = X_WMustard.translation()
    p_Target = p_Mustard + np.array([-0.17, 0, 0])
    R_Target = RotationMatrix.MakeXRotation(np.pi) @ RotationMatrix.MakeZRotation(
        -np.pi / 2
    )


    X_WG_target = RigidTransform(R_Target, p_Target)

    # --- VISUALIZE FRAMES ---
    print("Drawing Target Frame (RGB Axes)...")
    AddMeshcatTriad(meshcat, "Target_Pose", length=0.15, radius=0.005, X_PT=X_WG_target)

    gripper_body = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))
    X_WG_Actual = plant.EvalBodyPoseInWorld(plant_context, gripper_body)

    print("Drawing Solution Frame...")
    AddMeshcatTriad(
        meshcat,
        "Solution_Pose",
        length=0.2,
        radius=0.008,
        opacity=1.0,
        X_PT=X_WG_Actual,
    )

    print(f"Target Pos: {np.round(p_Target, 3)}")

    print("\nSolving IK...")
    try:
        q_sol = solve_ik_for_pose(plant, X_WG_target, theta_bound=0.05, pos_tol=0.01)
        print(f"✓ IK Success! q: {np.round(q_sol, 3)}")

        iiwa_model = plant.GetModelInstanceByName("iiwa")
        plant.SetPositions(plant_context, iiwa_model, q_sol)

        diagram.ForcedPublish(context)
        print("\nRobot updated in Meshcat.")

    except RuntimeError as e:
        print(f"✗ IK Failed: {e}")

    print("\nScript running... Press Ctrl+C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
