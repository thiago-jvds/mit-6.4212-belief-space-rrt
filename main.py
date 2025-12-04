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
)
import argparse
from src.perception.light_and_dark import LightDarkRegionSystem
from src.simulation.sim_setup import IiwaProblemBelief
from src.planning.standard_rrt import rrt_planning
from src.planning.belief_space_rrt import rrbt_planning
from src.simulation.sim_setup import visualize_noisy_execution, visualize_belief_path, visualize_belief_tree
from src.utils.config_loader import load_rrbt_config


def debug_path_beliefs(problem, path):
    """Print uncertainty at each waypoint."""
    sigma = np.eye(7) * 1e-6
    
    print("\n" + "="*60)
    print("PATH BELIEF ANALYSIS")
    print("="*60)
    print(f"{'Step':>4} | {'Light?':>6} | {'Trace(Σ)':>12} | {'Status':>10}")
    print("-"*60)
    
    for i, q in enumerate(path):
        A, Q, C, R = problem.get_dynamics_and_observation(q)
        sigma_pred = A @ sigma @ A.T + Q
        S = C @ sigma_pred @ C.T + R
        K = sigma_pred @ C.T @ np.linalg.inv(S)
        sigma = (np.eye(7) - K @ C) @ sigma_pred
        
        uncertainty = np.trace(sigma)
        in_light = problem.is_in_light(q)
        status = "✓ OK" if uncertainty < 0.01 else "⚠️ HIGH"
        
        print(f"{i:>4} | {'LIGHT' if in_light else 'DARK':>6} | {uncertainty:>12.6f} | {status:>10}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="MIT 6.4212 Robot Simulation")
    parser.add_argument(
        "--visualize",
        type=str,
        nargs="?",
        const=True,
        default=True,
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
    print(f"    > Planner: MaxUncert={config.planner.max_uncertainty}, LightBias={config.planner.prob_sample_light}")
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
            "light_region_indicator", RigidTransform(RotationMatrix(), config.simulation.light_center)
        )

    simulator.AdvanceTo(0.1)

    print("\n" + "=" * 40)
    print(f"RUNNING PLANNER: {args.planner.upper()}")
    print("=" * 40)

    q_start = q_home
    q_goal = config.simulation.q_goal

    plant_context = plant.CreateDefaultContext()
    iiwa = plant.GetModelInstanceByName("iiwa")

    # 2. Set the plant to q_goal
    plant.SetPositions(plant_context, iiwa, q_goal)

    # 3. Calculate Pose of the Gripper (wsg body)
    # Note: Ensure "body" is the correct link name for your gripper in the SDF
    wsg_body = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))
    X_Goal = plant.EvalBodyPoseInWorld(plant_context, wsg_body)

    # 4. Draw the Triad
    AddMeshcatTriad(meshcat, "goal_pose", length=0.2, radius=0.005)
    meshcat.SetTransform("goal_pose", X_Goal)

    problem = IiwaProblemBelief(
        q_start=q_start,
        q_goal=q_goal,
        gripper_setpoint=0.1,
        meshcat=meshcat,
        light_center=config.simulation.light_center,
        light_size=config.simulation.light_size,
        Q_uncertainty=float(config.physics.process_noise_scale),
        R_light_uncertainty=float(config.physics.meas_noise_light),
        R_dark_uncertainty=float(config.physics.meas_noise_dark),
    )

    path = None
    if args.planner == "rrt":
        print("Running Standard RRT...")
        path, k = rrt_planning(
            problem,
            max_iterations=config.planner.max_iterations,
            prob_sample_q_goal=float(config.planner.prob_sample_goal),
        )
    elif args.planner == "rrbt":
        print("Running RRBT...")
        
        # Create visualization callback for debugging the belief tree
        def tree_viz_callback(rrbt_tree, iteration):
            visualize_belief_tree(rrbt_tree, problem, meshcat, iteration)
        
        path, k = rrbt_planning(
            problem,
            max_iterations=config.planner.max_iterations,
            max_uncertainty=float(config.planner.max_uncertainty),
            prob_sample_q_goal=float(config.planner.prob_sample_goal),
            prob_sample_q_light=float(config.planner.prob_sample_light),
            q_light_hint=config.planner.q_light_hint,
            visualize_callback=tree_viz_callback,
            visualize_interval=1,  # Visualize every iteration
        )

    # 4. Visualize Noisy Execution if path found
    if path:
        print(f"✓ Path found ({k+1} iters). Replaying with NOISE...")
        
        # Debug the belief path
        debug_path_beliefs(problem, path)
        
        # # Visualize the belief path before noisy execution
        # visualize_belief_path(problem, path, meshcat)

        # visualize_noisy_execution(problem, path, meshcat)
        print(f"\n✓ Path found in ({k+1} iters).")
    else:
        print("✗ No path found.")

    print("\nSimulation complete. Press Ctrl+C to exit.")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
