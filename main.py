#!/usr/bin/env python3
"""
Main script - Unified RRBT-RRT Planning and Execution.

Uses a PlannerSystem LeafSystem that integrates planning into the Drake
simulation loop. The planner runs RRBT for information gathering, then
RRT for reaching the goal, all within a single continuous simulation.

Starts Meshcat on port 7000, loads the scenario, and keeps running
until you press Ctrl+C.

Usage:
    python main.py

Then open http://localhost:7000 in your browser.
"""

import numpy as np
from pathlib import Path
from manipulation.station import MakeHardwareStation, LoadScenario, AddPointClouds
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
)
import argparse
from src.perception.light_and_dark import BinLightDarkRegionSensorSystem, MustardPositionLightDarkRegionSensorSystem
from src.perception.mustard_pose_estimator import MustardPoseEstimatorSystem
from src.planning.planner_system import PlannerSystem, PlannerState
from src.visualization.belief_bar_chart import BeliefBarChartSystem
from src.estimation.belief_estimator import BeliefEstimatorSystem
from src.utils.config_loader import load_rrbt_config
from src.utils.camera_pose_manager import restore_camera_pose


# Initialize numpy random generator for uniform random numbers
np_rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility


def place_mustard_bottle_randomly_in_bin(meshcat, plant, plant_context, true_bin, np_rng: np.random.Generator):        
    """
    Place the mustard bottle randomly within the specified bin.
    
    Args:
        meshcat: Meshcat visualizer instance
        plant: MultibodyPlant reference
        plant_context: Plant context from simulator
        true_bin: Bin index (0 or 1) to place the bottle in
        np_rng: Numpy random generator
        
    Returns:
        X_WM: RigidTransform of mustard bottle in world frame
    """
    # Get true bin's pose in world frame
    true_bin_instance = plant.GetModelInstanceByName(f"bin{true_bin}")
    true_bin_body = plant.GetBodyByName("bin_base", true_bin_instance)
    X_WB = plant.EvalBodyPoseInWorld(plant_context, true_bin_body)
    
    # Generate random position and orientation for mustard bottle
    random_rotation = RollPitchYaw(-np.pi/2, 0, np_rng.uniform(0, 2*np.pi)).ToRotationMatrix()

    # Random XY position within bin bounds, Z height above bin
    x_offset = -0.01
    x_range = (-0.01+x_offset, 0.01+x_offset)  # min, max for x offset from bin center
    y_range = (-0.15, 0.15)  # min, max for y offset from bin center
    random_z = 0.2  # Height above bin

    random_x = np_rng.uniform(x_range[0], x_range[1])
    random_y = np_rng.uniform(y_range[0], y_range[1])

    # Visualize the random initialization space as a translucent green box
    box_width = x_range[1] - x_range[0]
    box_depth = y_range[1] - y_range[0]
    box_height = 0.02  # Thin box to show the XY region at the drop height

    init_space_box = Box(box_width, box_depth, box_height)
    meshcat.SetObject("init_space", init_space_box, Rgba(0, 1, 0, 0.3))  # Translucent green

    # Position the box at the center of the initialization region (relative to bin)
    box_center_in_bin = [
        (x_range[0] + x_range[1]) / 2,
        (y_range[0] + y_range[1]) / 2,
        random_z
    ]
    X_WBox = X_WB.multiply(RigidTransform(box_center_in_bin))
    meshcat.SetTransform("init_space", X_WBox)

    # Create transform relative to bin, then convert to world frame
    X_BM = RigidTransform(random_rotation, [random_x, random_y, random_z])
    X_WM = X_WB.multiply(X_BM)

    # Set mustard bottle pose
    mustard_body = plant.GetBodyByName("base_link_mustard")
    plant.SetFreeBodyPose(plant_context, mustard_body, X_WM)
    
    print(f"  Placed mustard bottle in bin{true_bin}:")
    print(f"    Bin position: {X_WB.translation()}")
    print(f"    Random offset: [{random_x:.3f}, {random_y:.3f}, {random_z:.3f}]")
    print(f"    World position: {X_WM.translation()}")
    
    return X_WM


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
    args = parser.parse_args()

    print("=" * 60)
    print("MIT 6.4212 - Belief-Space RRT")
    print("Unified RRBT-RRT Planning and Execution")
    print("=" * 60)

    # Load configuration
    config = load_rrbt_config()
    print("Loaded Configuration:")
    print(f"    > Physics: Q_scale={config.physics.process_noise_scale}")
    print(
        f"    > Planner: max_bin_uncertainty={config.planner.max_bin_uncertainty}, "
        f"LightBias={config.planner.bias_prob_sample_q_bin_light}"
    )
    print()

    # Start Meshcat
    try:
        params = MeshcatParams()
        params.port = 7000
        meshcat = Meshcat(params=params)
    except RuntimeError as e:
        print("\n[ERROR] Could not start Meshcat on port 7000")
        print(f"  {e}")
        print("\n  Port 7000 is likely already in use.")
        print("  Please stop any other Meshcat servers or Python processes using that port.")
        raise

    # Restore saved camera pose if available
    if args.visualize == "True":
        restore_camera_pose(meshcat)

    # Load scenario
    scenario_path = Path(__file__).parent / "config" / "scenario.yaml"
    with open(scenario_path, "r") as f:
        scenario = LoadScenario(data=f.read())

    # ============================================================
    # BUILD UNIFIED DIAGRAM
    # ============================================================
    print("\nBuilding unified diagram...")
    builder = DiagramBuilder()
    
    # Add hardware station
    station = builder.AddSystem(MakeHardwareStation(scenario=scenario, meshcat=meshcat))
    station.set_name("HardwareStation")
    plant = station.GetSubsystemByName("plant")

    # ============================================================
    # GET BIN TRANSFORMS FOR BELIEF BAR CHART POSITIONING
    # ============================================================
    # Create a temporary context to get bin poses (determined by model file)
    temp_plant_context = plant.CreateDefaultContext()
    
    # Get bin0 transform
    bin0_instance = plant.GetModelInstanceByName("bin0")
    bin0_body = plant.GetBodyByName("bin_base", bin0_instance)
    X_W_bin0 = plant.EvalBodyPoseInWorld(temp_plant_context, bin0_body)
    
    # Get bin1 transform
    bin1_instance = plant.GetModelInstanceByName("bin1")
    bin1_body = plant.GetBodyByName("bin_base", bin1_instance)
    X_W_bin1 = plant.EvalBodyPoseInWorld(temp_plant_context, bin1_body)
    
    # Define relative transform from bin frame to chart position
    # "Top right edge" of bin: positive X (forward), negative Y (right), raised Z
    # Bin dimensions are roughly 0.6m x 0.4m, so offset to corner and above
    X_bin_chart = RigidTransform([-0.23, -0.28, 0.21])
    
    print(f"  Bin0 position: {X_W_bin0.translation()}")
    print(f"  Bin1 position: {X_W_bin1.translation()}")
    print(f"  Chart offset from bin: {X_bin_chart.translation()}")

    # Add PlannerSystem (replaces ConstantVectorSource)
    planner = builder.AddSystem(PlannerSystem(plant, config, meshcat, scenario_path))
    planner.set_name("PlannerSystem")
    
    # Connect planner to robot arm
    builder.Connect(
        planner.GetOutputPort("iiwa_position_command"),
        station.GetInputPort("iiwa.position")
    )

    # Add WSG gripper source (constant open position)
    wsg_position_source = builder.AddSystem(ConstantVectorSource([0.1]))
    wsg_position_source.set_name("GripperPositionSource")
    builder.Connect(
        wsg_position_source.get_output_port(),
        station.GetInputPort("wsg.position")
    )

    # Add Perception System (LightDarkRegionSystem)
    bin_perception_sys = builder.AddSystem(
        BinLightDarkRegionSensorSystem(
            plant=plant,
            light_region_center=config.simulation.bin_light_center,
            light_region_size=config.simulation.bin_light_size,
            tpr_light=float(config.physics.tpr_light),
            fpr_light=float(config.physics.fpr_light),
        )
    )
    bin_perception_sys.set_name("LightDarkPerception")
    # Connect perception to station output
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        bin_perception_sys.GetInputPort("iiwa.position"),
    )
    
    # Add Mustard Position Perception System
    mustard_position_perception_sys = builder.AddSystem(
        MustardPositionLightDarkRegionSensorSystem(
            plant=plant,
            light_region_center=config.simulation.mustard_position_light_center,
            light_region_size=config.simulation.mustard_position_light_size,
            meas_noise_light=float(config.physics.meas_noise_light),
            meas_noise_dark=float(config.physics.meas_noise_dark),
        )
    )
    mustard_position_perception_sys.set_name("MustardPositionPerception")
    # Connect mustard position perception to station output
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        mustard_position_perception_sys.GetInputPort("iiwa.position"),
    )

    # Add Belief Estimator System (Discrete Bayes Filter)
    # Note: true_bin will be set after we randomly choose it
    # For now, use a placeholder value - it will be updated via configure
    belief_estimator = builder.AddSystem(
        BeliefEstimatorSystem(
            n_bins=2,
            true_bin=0,  # Placeholder, will be updated
        )
    )
    belief_estimator.set_name("BeliefEstimator")
    
    # Connect estimator to perception's sensor_model output
    builder.Connect(
        bin_perception_sys.GetOutputPort("sensor_model"),
        belief_estimator.GetInputPort("sensor_model")
    )

    # Add Belief Bar Chart Visualizer (positioned near each bin)
    belief_viz = builder.AddSystem(
        BeliefBarChartSystem(
            meshcat=meshcat,
            n_bins=2,
            X_W_bin0=X_W_bin0,
            X_W_bin1=X_W_bin1,
            X_bin_chart=X_bin_chart,
        )
    )
    belief_viz.set_name("BeliefBarChart")
    
    # Connect visualizer to estimator output
    builder.Connect(
        belief_estimator.GetOutputPort("belief"),
        belief_viz.GetInputPort("belief")
    )

    # ============================================================
    # ADD POINT CLOUD GENERATION FROM CAMERAS
    # ============================================================
    print("  Adding point cloud generation from cameras...")
    to_point_cloud = AddPointClouds(
        scenario=scenario,
        station=station,
        builder=builder,
        meshcat=meshcat,
    )
    print(f"    Added point cloud converters for: {list(to_point_cloud.keys())}")

    # ============================================================
    # ADD MUSTARD POSE ESTIMATOR SYSTEM
    # ============================================================
    print("  Adding MustardPoseEstimatorSystem...")
    pose_estimator = builder.AddSystem(
        MustardPoseEstimatorSystem(meshcat=meshcat, n_bins=2)
    )
    pose_estimator.set_name("MustardPoseEstimator")

    # Connect camera point clouds to pose estimator
    for i in range(6):
        camera_name = f"camera{i}"
        if camera_name in to_point_cloud:
            builder.Connect(
                to_point_cloud[camera_name].get_output_port(),
                pose_estimator.GetInputPort(f"camera{i}_point_cloud")
            )
            print(f"    Connected {camera_name} point cloud")
        else:
            print(f"    WARNING: {camera_name} not found in point cloud converters!")

    # Connect belief to pose estimator
    builder.Connect(
        belief_estimator.GetOutputPort("belief"),
        pose_estimator.GetInputPort("belief")
    )
    print("    Connected belief to pose estimator")

    # Connect pose estimator to planner
    builder.Connect(
        pose_estimator.GetOutputPort("estimated_pose"),
        planner.GetInputPort("estimated_mustard_pose")
    )
    print("    Connected pose estimator to planner")

    # Build the diagram
    diagram = builder.Build()
    diagram.set_name("UnifiedPlanningDiagram")

    # ============================================================
    # INITIALIZE SIMULATOR AND ENVIRONMENT
    # ============================================================
    print("\nInitializing simulator...")
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    # Visualize light region
    if args.visualize == "True":
        # Visualize bin light region indicator
        meshcat.SetObject(
            "bin_light_region_indicator",
            Box(*config.simulation.bin_light_size),
            Rgba(0, 1, 0, 0.3),  # Green, 0.3 Alpha
        )
        meshcat.SetTransform(
            "bin_light_region_indicator",
            RigidTransform(RotationMatrix(), config.simulation.bin_light_center),
        )
        # Visualize mustard position light region indicator
        meshcat.SetObject(
            "mustard_position_light_region_indicator",
            Box(*config.simulation.mustard_position_light_size),
            Rgba(0, 1, 0, 0.3),  # Green, 0.3 Alpha
        )
        meshcat.SetTransform(
            "mustard_position_light_region_indicator",
            RigidTransform(RotationMatrix(), config.simulation.mustard_position_light_center),
        )

    # Step simulation briefly to initialize
    simulator.AdvanceTo(0.1)

    # ============================================================
    # SETUP ENVIRONMENT (PLACE MUSTARD BOTTLE)
    # ============================================================
    print("\n" + "=" * 60)
    print("ENVIRONMENT SETUP")
    print("=" * 60)

    # Get simulator context for environment setup
    sim_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(sim_context)

    # Randomly choose a true bin
    true_bin = np.random.randint(0, 2)
    print(f"Randomly chosen true bin: {true_bin}")

    # Place mustard bottle randomly in the true bin
    X_WM_mustard = place_mustard_bottle_randomly_in_bin(
        meshcat, plant, plant_context, true_bin, np_rng
    )

    # Force publish to update Meshcat visualization with new mustard position
    diagram.ForcedPublish(sim_context)

    # Update belief estimator with correct true_bin
    # Note: BeliefEstimatorSystem stores true_bin as instance variable
    belief_estimator._true_bin = true_bin

    # ============================================================
    # CONFIGURE PLANNER AND RUN SIMULATION
    # ============================================================
    print("\n" + "=" * 60)
    print("STARTING UNIFIED PLANNING AND EXECUTION")
    print("=" * 60)

    # Configure planner to start planning
    planner.configure_for_execution(true_bin, X_WM_mustard)

    # Start Meshcat recording
    meshcat.StartRecording()

    print("\nMeshcat running at http://localhost:7000")
    print("Running simulation... (Press Ctrl+C to exit)")
    print()

    # Run simulation loop
    try:
        while True:
            current_time = simulator.get_context().get_time()
            simulator.AdvanceTo(current_time + 0.1)
            
            # Check if planning and execution are complete
            if planner.is_complete():
                print("\n" + "=" * 60)
                print("SIMULATION COMPLETE")
                print("=" * 60)
                break
                
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")

    # Stop recording and publish
    meshcat.StopRecording()
    meshcat.PublishRecording()

    print("\nExecution complete! Replay available in Meshcat.")
    print("Press Ctrl+C to exit.")
    
    # Keep Meshcat alive for replay viewing
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
