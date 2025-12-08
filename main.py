#!/usr/bin/env python3
"""
Main script - Visualizes the robot scenario in Meshcat.

Starts Meshcat on port 7000, loads the scenario, and keeps running
until you press Ctrl+C.

Usage:
    python main.py

Then open http://localhost:7000 in your browser.
"""

import numpy as np
import time
from pathlib import Path
from manipulation.station import MakeHardwareStation, LoadScenario
from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    Meshcat,
    MeshcatParams,
    ConstantVectorSource,
    Box,
    Rgba,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    PiecewisePolynomial,
    TrajectorySource,
)
import argparse
from src.perception.light_and_dark import LightDarkRegionSystem
from src.simulation.simulation_tools import IiwaProblemBelief, IiwaProblem
from src.planning.standard_rrt import rrt_planning
from src.planning.belief_space_rrt import rrbt_planning
from src.simulation.sim_setup import (
    visualize_noisy_execution,
    visualize_belief_path,
    visualize_belief_tree,
)
from src.visualization.belief_visualizer import BeliefVisualizerSystem
from src.estimation.belief_estimator import BeliefEstimatorSystem
from src.utils.config_loader import load_rrbt_config
from src.utils.camera_pose_manager import restore_camera_pose
from src.utils.ik_solver import solve_ik_for_pose


def path_to_trajectory(path: list, time_per_segment: float = 0.02) -> PiecewisePolynomial:
    """
    Convert a list of joint configurations to a time-parameterized trajectory.
    
    Args:
        path: List of joint configurations (each is a 7-element array/tuple)
        time_per_segment: Time allocated per path segment in seconds
        
    Returns:
        PiecewisePolynomial trajectory (7D output matching iiwa.position port)
    """
    # Convert path to numpy array (n_points x 7)
    path_array = np.array([np.array(q) for q in path])
    
    # Create time breakpoints
    n_points = len(path)
    times = np.linspace(0, (n_points - 1) * time_per_segment, n_points)
    
    # Create trajectory using first-order hold (linear interpolation)
    # PiecewisePolynomial.FirstOrderHold expects:
    # - breaks: 1D array of times
    # - samples: 2D array where each column is a sample (7 x n_points)
    trajectory = PiecewisePolynomial.FirstOrderHold(times, path_array.T)
    
    return trajectory


def debug_path_beliefs(problem, path):
    """Print uncertainty at each waypoint."""
    sigma = np.eye(7) * 1e-6

    print("\n" + "=" * 60)
    print("PATH BELIEF ANALYSIS")
    print("=" * 60)
    print(f"{'Step':>4} | {'Light?':>6} | {'Trace(Σ)':>12} | {'Status':>10}")
    print("-" * 60)

    for i, q in enumerate(path):
        A, Q, C, R = problem.get_dynamics_and_observation(q)
        sigma_pred = A @ sigma @ A.T + Q
        S = C @ sigma_pred @ C.T + R
        K = sigma_pred @ C.T @ np.linalg.inv(S)
        sigma = (np.eye(7) - K @ C) @ sigma_pred

        uncertainty = np.trace(sigma)
        in_light = problem.is_in_light(q)
        status = "✓ OK" if uncertainty < 0.01 else "⚠️ HIGH"

        print(
            f"{i:>4} | {'LIGHT' if in_light else 'DARK':>6} | {uncertainty:>12.6f} | {status:>10}"
        )

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MIT 6.4212 Robot Simulation")
    parser.add_argument(
        "--visualize",
        type=str,
        nargs="?",
        const="True",
        default="True",
        help="Enable/Disable Meshcat visualization (True/False)",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="rrbt",
        choices=["rrt", "rrbt"],
        help="Which planner to run",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MIT 6.4212 - Belief-Space RRT")
    print("Robot Manipulation Visualization")
    print("=" * 60)

    config = load_rrbt_config()
    print("Loaded Configuration:")
    print(f"    > Physics: Q_scale={config.physics.process_noise_scale}")
    print(
        f"    > Planner: MaxUncert={config.planner.max_uncertainty}, LightBias={config.planner.prob_sample_light}"
    )
    print()

    try:
        params = MeshcatParams()
        params.port = 7000
        meshcat = Meshcat(params=params)
    except RuntimeError as e:
        print("\n✗ ERROR: Could not start Meshcat on port 7000")
        print(f"  {e}")
        print("\n  Port 7000 is likely already in use.")
        print(
            "  Please stop any other Meshcat servers or Python processes using that port."
        )
        raise

    # Restore saved camera pose if available (only if visualization is enabled)
    if args.visualize == "True":
        restore_camera_pose(meshcat)

    scenario_path = Path(__file__).parent / "config" / "scenario.yaml"

    with open(scenario_path, "r") as f:
        scenario = LoadScenario(data=f.read())

    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario=scenario, meshcat=meshcat))

    plant = station.GetSubsystemByName("plant")

    # ====== Perception system ======
    perception_sys = builder.AddSystem(
        LightDarkRegionSystem(
            plant=plant,
            light_region_center=config.simulation.light_center,
            light_region_size=config.simulation.light_size,
            sigma_light=np.sqrt(float(config.physics.meas_noise_light)),
            sigma_dark=np.sqrt(float(config.physics.meas_noise_dark)),
        )
    )

    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        perception_sys.GetInputPort("iiwa.position"),
    )

    # Set robot joint positions
    q_home = config.simulation.q_home
    iiwa_position_source = builder.AddSystem(ConstantVectorSource(q_home))
    builder.Connect(
        iiwa_position_source.get_output_port(), station.GetInputPort("iiwa.position")
    )

    wsg_position_source = builder.AddSystem(ConstantVectorSource([0.1]))
    builder.Connect(
        wsg_position_source.get_output_port(), station.GetInputPort("wsg.position")
    )

    # Build diagram and create simulator
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    if args.visualize == "True":
        meshcat.SetObject(
            "light_region_indicator",
            Box(*config.simulation.light_size),
            Rgba(0, 1, 0, 0.3),  # Green, 0.3 Alpha
        )
        meshcat.SetTransform(
            "light_region_indicator",
            RigidTransform(RotationMatrix(), config.simulation.light_center),
        )

    simulator.AdvanceTo(0.1)

    print("\n" + "=" * 40)
    print(f"RUNNING PLANNER: {args.planner.upper()}")
    print("=" * 40)

    q_start = q_home
    
    plant_context = plant.CreateDefaultContext()
    iiwa = plant.GetModelInstanceByName("iiwa")

    # Compute q_goal from tf_goal (task-space goal) using IK
    tf_goal = config.simulation.tf_goal
    X_WG_goal = RigidTransform(
        RollPitchYaw(tf_goal.rpy).ToRotationMatrix(),
        tf_goal.translation
    )
    
    print(f"Computing q_goal from tf_goal:")
    print(f"  translation: {tf_goal.translation}")
    print(f"  rpy: {tf_goal.rpy}")
    
    try:
        q_goal = solve_ik_for_pose(
            plant=plant,
            X_WG_target=X_WG_goal,
            q_nominal=tuple(q_home),
            theta_bound=0.1,  # Relaxed orientation tolerance
            pos_tol=0.01,     # 1cm position tolerance
        )
        q_goal = np.array(q_goal)
        print(f"✓ q_goal computed: {q_goal}")
    except RuntimeError as e:
        print(f"✗ IK failed for tf_goal: {e}")
        raise RuntimeError("Cannot compute q_goal from tf_goal. Check that the goal pose is reachable.")

    # Visualize the goal pose
    AddMeshcatTriad(meshcat, "goal_pose", length=0.2, radius=0.005)
    meshcat.SetTransform("goal_pose", X_WG_goal)

    # Calculate q_light_hint from light_center using IK (shared by both planners)
    # This creates a waypoint in the light region for better observation
    light_center = config.simulation.light_center
    target_rotation = RotationMatrix.MakeXRotation(np.pi) @ RotationMatrix.MakeZRotation(np.pi)
    X_WG_light = RigidTransform(target_rotation, light_center)
    
    print(f"Computing q_light_hint from light_center {light_center}...")
    try:
        q_light_hint = solve_ik_for_pose(
            plant=plant,
            X_WG_target=X_WG_light,
            q_nominal=tuple(q_home),
            theta_bound=0.1,  # Relaxed orientation tolerance
            pos_tol=0.05,     # 5cm position tolerance
        )
        q_light_hint = np.array(q_light_hint)
        print(f"✓ q_light_hint computed: {q_light_hint}")
        
        # Visualize the light region sampling pose in Meshcat
        if args.visualize == "True":
            AddMeshcatTriad(meshcat, "light_region_sampling_pose", length=0.15, radius=0.004)
            meshcat.SetTransform("light_region_sampling_pose", X_WG_light)
            print(f"  Visualized light region sampling pose in Meshcat")
    except RuntimeError as e:
        print(f"⚠ IK failed for light_center, using q_home as fallback: {e}")
        # q_light_hint = np.array(q_home)
        raise RuntimeError("Cannot compute q_light_hint from light_center. Check that the light region is reachable.")

    problem = IiwaProblemBelief(
        q_start=q_start,
        q_goal=q_goal,
        gripper_setpoint=0.1,
        meshcat=meshcat,
        light_center=config.simulation.light_center,
        light_size=config.simulation.light_size,
        scale_R_light=float(config.physics.meas_noise_light),
        scale_R_dark=float(config.physics.meas_noise_dark),
    )

    final_path = None
    if args.planner == "rrt":
        print("Running Standard RRT (two-phase: start → light → goal)...")
        
        # Phase 1: RRT from q_start to q_light_hint (information gathering waypoint)
        print("\n  Phase 1: Planning path to light region...")
        problem_to_light = IiwaProblem(
            q_start=q_start,
            q_goal=tuple(q_light_hint),
            gripper_setpoint=0.1,
            meshcat=meshcat,
        )
        path_to_light, k1 = rrt_planning(
            problem_to_light,
            max_iterations=config.planner.max_iterations,
            prob_sample_q_goal=float(config.planner.prob_sample_goal),
        )
        
        if path_to_light:
            print(f"  ✓ Phase 1 complete: {len(path_to_light)} waypoints, {k1} iterations")
            
            # Phase 2: RRT from q_light_hint to q_goal
            print("\n  Phase 2: Planning path from light region to goal...")
            problem_to_goal = IiwaProblem(
                q_start=path_to_light[-1],  # Start from end of first path
                q_goal=q_goal,
                gripper_setpoint=0.1,
                meshcat=meshcat,
            )
            path_to_goal, k2 = rrt_planning(
                problem_to_goal,
                max_iterations=config.planner.max_iterations,
                prob_sample_q_goal=0.25,  # Higher bias for goal-directed motion
            )
            
            if path_to_goal:
                print(f"  ✓ Phase 2 complete: {len(path_to_goal)} waypoints, {k2} iterations")
                # Combine paths (avoid duplicating the junction point)
                final_path = path_to_light + path_to_goal[1:]
                print(f"✓ Two-phase RRT complete: {len(final_path)} total waypoints")
            else:
                print("  ✗ Phase 2 failed (could not reach goal from light region)")
                final_path = path_to_light  # At least visualize the path to light
        else:
            print("  ✗ Phase 1 failed (could not reach light region)")
            final_path = None
            
    elif args.planner == "rrbt":
        print("Running RRBT...")

        # Create visualization callback for debugging the belief tree
        def tree_viz_callback(rrbt_tree, iteration):
            visualize_belief_tree(rrbt_tree, problem, meshcat, iteration)

        rrbt_result, k = rrbt_planning(
            problem,
            max_iterations=int(config.planner.max_iterations),
            prob_sample_q_goal=float(config.planner.prob_sample_goal),
            prob_sample_q_light=float(config.planner.prob_sample_light),
            max_uncertainty=float(config.planner.max_uncertainty),
            lambda_weight=float(config.planner.lambda_weight),
            q_light_hint=q_light_hint,
            visualize_callback=None, # Set to tree_viz_callback to see tree grow
            visualize_interval=1000,
            verbose=False
        )

        if rrbt_result:
            print("✓ RRBT Converged. Planning Grasp...")
            path_to_info, pred_q_goal = rrbt_result
            
            # CALL 2: Standard RRT to Reach the Estimated Goal
            # We update the problem start/goal
            problemRRT = IiwaProblem(
                q_start=path_to_info[-1],
                q_goal=pred_q_goal,
                gripper_setpoint=0.1,
                meshcat=meshcat,
            )
            
            path_to_grasp, _ = rrt_planning(
                problemRRT, 
                max_iterations=1000,
                prob_sample_q_goal=0.25, # High bias for short grasp
            )
            
            if path_to_grasp:
                final_path = path_to_info + path_to_grasp
            else:
                print("✗ Grasp planning failed (RRT could not connect info state to goal).")
                final_path = path_to_info # At least visualize the info gathering
        else:
            print("✗ RRBT Failed to find information path.")
            final_path = None

    # --- 7. EXECUTION DIAGRAM ---
    if final_path:
        print("✓ Sequence Complete. Building Execution Diagram...")
        
        # Convert path to trajectory
        trajectory = path_to_trajectory(final_path, time_per_segment=0.02)
        print(f"   Trajectory duration: {trajectory.end_time():.2f}s ({len(final_path)} waypoints)")
        
        # Build the Execution Diagram
        exec_builder = DiagramBuilder()
        
        # Re-load scenario and create new station for execution
        with open(scenario_path, "r") as f:
            exec_scenario = LoadScenario(data=f.read())
        
        exec_station = exec_builder.AddSystem(
            MakeHardwareStation(scenario=exec_scenario, meshcat=meshcat)
        )
        exec_station.set_name("ExecutionStation")
        
        exec_plant = exec_station.GetSubsystemByName("plant")
        
        # Add TrajectorySource for the planned path (outputs 7D positions)
        traj_source = exec_builder.AddSystem(TrajectorySource(trajectory))
        traj_source.set_name("PlannedTrajectorySource")
        
        # Connect TrajectorySource to station's iiwa.position input port
        exec_builder.Connect(
            traj_source.get_output_port(),
            exec_station.GetInputPort("iiwa.position")
        )
        
        # Add WSG gripper source (constant open position)
        exec_wsg_source = exec_builder.AddSystem(ConstantVectorSource([0.1]))
        exec_wsg_source.set_name("GripperPositionSource")
        exec_builder.Connect(
            exec_wsg_source.get_output_port(),
            exec_station.GetInputPort("wsg.position")
        )
        
        # Add Perception System (LightDarkRegionSystem)
        exec_perception = exec_builder.AddSystem(
            LightDarkRegionSystem(
                plant=exec_plant,
                light_region_center=config.simulation.light_center,
                light_region_size=config.simulation.light_size,
                sigma_light=np.sqrt(float(config.physics.meas_noise_light)),
                sigma_dark=np.sqrt(float(config.physics.meas_noise_dark)),
            )
        )
        exec_perception.set_name("LightDarkPerception")
        
        # Connect perception to station output
        exec_builder.Connect(
            exec_station.GetOutputPort("iiwa.position_measured"),
            exec_perception.GetInputPort("iiwa.position")
        )
        
        # Add Belief Estimator System (Kalman Filter for covariance estimation)
        # Receives measurement variance from LightDarkRegionSystem (single source of truth)
        belief_estimator = exec_builder.AddSystem(
            BeliefEstimatorSystem(
                initial_sigma=np.eye(7) * 1.0,  # High initial uncertainty
            )
        )
        belief_estimator.set_name("BeliefEstimator")
        
        # Connect estimator to perception's measurement_variance output (single source of truth)
        exec_builder.Connect(
            exec_perception.GetOutputPort("measurement_variance"),
            belief_estimator.GetInputPort("measurement_variance")
        )
        
        # Add Belief Visualizer System (renders goal uncertainty ellipsoid)
        # Receives covariance from BeliefEstimatorSystem
        belief_viz = exec_builder.AddSystem(
            BeliefVisualizerSystem(
                meshcat=meshcat,
                plant=exec_plant,
                goal_config=q_goal,  # Use computed q_goal from tf_goal
            )
        )
        belief_viz.set_name("BeliefVisualizer")
        
        # Connect visualizer to estimator output
        exec_builder.Connect(
            belief_estimator.GetOutputPort("covariance"),
            belief_viz.GetInputPort("covariance")
        )
        
        # Build execution diagram
        exec_diagram = exec_builder.Build()
        exec_diagram.set_name("ExecutionDiagram")
        
        # Create simulator
        exec_simulator = Simulator(exec_diagram)
        exec_simulator.set_target_realtime_rate(1.0)
        
        # Setup Meshcat recording
        meshcat.StartRecording()
        
        print("   Running execution simulation...")
        exec_simulator.AdvanceTo(trajectory.end_time())
        
        meshcat.StopRecording()
        meshcat.PublishRecording()
        
        print("✓ Execution Complete! Replay available in Meshcat.")
    else:
        print("✗ No path found.")

    print("\nSimulation complete. Press Ctrl+C to exit.")


if __name__ == "__main__":
    main()
