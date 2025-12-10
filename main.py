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

import random
import subprocess
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
import matplotlib.pyplot as plt
from src.perception.light_and_dark import BinLightDarkRegionSensorSystem, MustardPositionLightDarkRegionSensorSystem
from src.perception.mustard_pose_estimator import MustardPoseEstimatorSystem
from src.planning.planner_system import PlannerSystem, PlannerState
from src.visualization.belief_bar_chart import BeliefBarChartSystem
from src.visualization.covariance_ellipsoid import CovarianceEllipsoidSystem
from src.estimation.belief_estimator import BinBeliefEstimatorSystem
from src.estimation.mustard_position_estimator import MustardPositionBeliefEstimatorSystem
from src.utils.config_loader import load_config
from src.utils.camera_pose_manager import restore_camera_pose


# ============================================================
# RANDOM SEED CONFIGURATION - Set this for deterministic runs
# ============================================================
RANDOM_SEED = 7

# Seed all random number generators for reproducibility
np.random.seed(RANDOM_SEED)  # Global numpy random state (for external libraries)
random.seed(RANDOM_SEED)     # Global Python random (for manipulation library)
np_rng = np.random.default_rng(seed=RANDOM_SEED)  # NumPy Generator for internal use


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
    # meshcat.SetObject("init_space", init_space_box, Rgba(0, 1, 0, 0.3))  # Translucent green

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


def probability_to_color(prob: float) -> tuple:
    """
    Convert probability to RGB color: red (0) - yellow (0.5) - green (1).
    
    Matches the color scheme used in BeliefBarChartSystem for Meshcat visualization.
    
    Args:
        prob: Probability value in [0, 1]
        
    Returns:
        Tuple of (r, g, b) values in [0, 1]
    """
    prob = np.clip(prob, 0.0, 1.0)
    if prob <= 0.5:
        # Red to Yellow: R=1, G increases from 0 to 1
        r, g, b = 1.0, prob * 2.0, 0.0
    else:
        # Yellow to Green: R decreases from 1 to 0, G=1
        r, g, b = 1.0 - (prob - 0.5) * 2.0, 1.0, 0.0
    return (r, g, b)


def plot_belief_bar_chart(belief: np.ndarray, bins: list, title: str, filename: str, 
                          true_bin: int = None, confidence_threshold: float = None):
    """
    Generate and save a matplotlib bar chart for belief visualization.
    
    Uses dynamic coloring: red (0) - yellow (0.5) - green (1) based on probability.
    
    Args:
        belief: Probability vector [P(bin0), P(bin1), ...]
        bins: List of bin labels for x-axis
        title: Plot title
        filename: Output filename for the saved plot
        true_bin: Optional index of the true bin (for reference in title)
        confidence_threshold: Optional misclassification risk threshold for RRBT goal condition.
                              If provided, draws a horizontal line at y = 1 - threshold.
    """
    plt.figure(figsize=(8, 6))
    
    # Dynamic colors based on probability: red (0) - yellow (0.5) - green (1)
    colors = [probability_to_color(p) for p in belief]
    bars = plt.bar(bins, belief, color=colors, alpha=0.8, edgecolor='black')
    
    plt.title(title, fontsize=14)
    plt.ylabel('Probability', fontsize=12)
    plt.xlabel('Bin Location', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add confidence threshold line if provided
    if confidence_threshold is not None:
        threshold_y = 1.0 - confidence_threshold
        plt.axhline(y=threshold_y, color='black', linestyle='--', linewidth=2, 
                    label=f'Confidence Threshold ({threshold_y:.2f})')
        plt.legend(loc='upper right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  Saved belief plot: {filename}")


def save_block_diagram(diagram, output_stem: Path):
    """
    Save the Drake diagram structure to Graphviz .dot and .png files.
    
    Args:
        diagram: Drake Diagram instance
        output_stem: Path stem (without extension) for output files
    """
    output_stem = Path(output_stem)
    if not output_stem.is_absolute():
        output_stem = Path.cwd() / output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    dot_path = output_stem.with_suffix(".dot")
    png_path = output_stem.with_suffix(".png")

    graphviz_str = diagram.GetGraphvizString()
    dot_path.write_text(graphviz_str)
    print(f"  Saved diagram .dot to: {dot_path}")

    try:
        subprocess.run(
            ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
            check=True,
        )
        print(f"  Saved diagram .png to: {png_path}")
    except FileNotFoundError:
        print("  Warning: 'dot' command not found. Install graphviz to render .png.")
    except subprocess.CalledProcessError as e:
        print(f"  Warning: graphviz 'dot' failed to render .png: {e}")


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
        "--generate-block-diagram",
        action="store_true",
        help="If set, export the built Drake diagram to .dot and .png then exit",
    )
    parser.add_argument(
        "--diagram-output-stem",
        type=str,
        default="system_diagram_full",
        help="Output path stem for diagram files (without extension)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MIT 6.4212 - Belief-Space RRT")
    print("Unified RRBT-RRT Planning and Execution")
    print("=" * 60)

    # Load configuration
    config = load_config()
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
    X_bin0_chart = RigidTransform([-0.22, 0.29, 0.21])
    X_bin1_chart = RigidTransform([-0.22, -0.29, 0.21])
    
    print(f"  Bin0 position: {X_W_bin0.translation()}")
    print(f"  Bin1 position: {X_W_bin1.translation()}")
    print(f"  Chart offset from bin0: {X_bin0_chart.translation()}")
    print(f"  Chart offset from bin1: {X_bin1_chart.translation()}")

    # Add PlannerSystem (replaces ConstantVectorSource)
    planner = builder.AddSystem(PlannerSystem(plant, config, meshcat, scenario_path, rng=np_rng))
    planner.set_name("PlannerSystem")
    
    # Connect planner to robot arm
    builder.Connect(
        planner.GetOutputPort("iiwa_position_command"),
        station.GetInputPort("iiwa.position")
    )

    # Connect planner's gripper command to robot gripper
    builder.Connect(
        planner.GetOutputPort("wsg_position_command"),
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
            rng=np_rng,
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
            rng=np_rng,
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
        BinBeliefEstimatorSystem(
            n_bins=2,
            true_bin=0,  # Placeholder, will be updated
            max_bin_uncertainty=float(config.planner.max_bin_uncertainty),
            rng=np_rng,
        )
    )
    belief_estimator.set_name("BinBeliefEstimator")
    
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
            X_bin0_chart=X_bin0_chart,
            X_bin1_chart=X_bin1_chart,
            max_height=0.15,
            bar_width=0.05,
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
        meshcat=None,  # Don't visualize continuously - we'll do it once
    )
    print(f"    Added point cloud converters for: {list(to_point_cloud.keys())}")
    
    # Export point cloud ports for one-time visualization
    point_cloud_output_ports = {}
    for camera_name, converter in to_point_cloud.items():
        port_name = f"{camera_name}_point_cloud"
        builder.ExportOutput(converter.get_output_port(), port_name)
        point_cloud_output_ports[camera_name] = port_name

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

    # Connect belief_confident trigger to pose estimator
    # This triggers ICP estimation only when bin belief is confident
    builder.Connect(
        belief_estimator.GetOutputPort("belief_confident"),
        pose_estimator.GetInputPort("estimation_trigger")
    )
    print("    Connected belief_confident trigger to pose estimator")

    # Connect pose estimator to planner
    builder.Connect(
        pose_estimator.GetOutputPort("estimated_pose"),
        planner.GetInputPort("estimated_mustard_pose")
    )
    print("    Connected pose estimator to planner")

    # ============================================================
    # ADD MUSTARD POSITION BELIEF ESTIMATOR (3D Kalman Filter)
    # ============================================================
    print("  Adding MustardPositionBeliefEstimatorSystem...")
    mustard_belief_estimator = builder.AddSystem(
        MustardPositionBeliefEstimatorSystem(
            initial_uncertainty=float(config.planner.mustard_position_initial_uncertainty),
        )
    )
    mustard_belief_estimator.set_name("MustardPositionBeliefEstimator")

    # Connect measurement variance from perception
    builder.Connect(
        mustard_position_perception_sys.GetOutputPort("measurement_variance"),
        mustard_belief_estimator.GetInputPort("measurement_variance")
    )
    print("    Connected measurement_variance from MustardPositionPerception")

    # Connect estimated pose from ICP (for initial position)
    builder.Connect(
        pose_estimator.GetOutputPort("estimated_pose"),
        mustard_belief_estimator.GetInputPort("estimated_pose")
    )
    print("    Connected estimated_pose from MustardPoseEstimator")

    # ============================================================
    # ADD COVARIANCE ELLIPSOID VISUALIZER
    # ============================================================
    print("  Adding CovarianceEllipsoidSystem...")
    covariance_viz = builder.AddSystem(
        CovarianceEllipsoidSystem(
            meshcat=meshcat,
            scale_factor=3.0,  # 3-sigma ellipsoid
            color=Rgba(1.0, 0.0, 0.0, 0.5),  # Red with 50% transparency
        )
    )
    covariance_viz.set_name("CovarianceEllipsoid")

    # Connect position and covariance from belief estimator
    builder.Connect(
        mustard_belief_estimator.GetOutputPort("position_mean"),
        covariance_viz.GetInputPort("position")
    )
    builder.Connect(
        mustard_belief_estimator.GetOutputPort("covariance"),
        covariance_viz.GetInputPort("covariance")
    )
    print("    Connected position and covariance to ellipsoid visualizer")

    # Connect covariance to planner for grasp planning
    builder.Connect(
        mustard_belief_estimator.GetOutputPort("covariance"),
        planner.GetInputPort("position_covariance")
    )
    print("    Connected covariance to planner for grasp planning")

    # Build the diagram
    diagram = builder.Build()
    diagram.set_name("UnifiedPlanningDiagram")

    # Optionally generate a block diagram and exit early
    if args.generate_block_diagram:
        print("\nGenerating block diagram...")
        save_block_diagram(diagram, Path(args.diagram_output_stem))
        print("Block diagram generated. Exiting before simulation.")
        return

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
        # Visualize mustard position light region indicator (orange to distinguish)
        meshcat.SetObject(
            "mustard_position_light_region_indicator",
            Box(*config.simulation.mustard_position_light_size),
            Rgba(1.0, 0.5, 0.0, 0.3),  # Orange, 0.3 Alpha
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
    true_bin = np_rng.integers(0, 2)
    print(f"Randomly chosen true bin: {true_bin}")

    # Place mustard bottle randomly in the true bin
    X_WM_mustard = place_mustard_bottle_randomly_in_bin(
        meshcat, plant, plant_context, true_bin, np_rng
    )

    # Force publish to update Meshcat visualization with new mustard position
    diagram.ForcedPublish(sim_context)

    # ============================================================
    # LET THE MUSTARD BOTTLE SETTLE
    # ============================================================
    # The bottle is placed 0.2m above the bin and falls under gravity.
    # We need to let physics simulation run so the bottle settles before
    # collecting point clouds for pose estimation.
    SETTLE_TIME = 2.0  # seconds for bottle to fall and settle
    print(f"\nLetting mustard bottle settle for {SETTLE_TIME}s...")
    current_time = simulator.get_context().get_time()
    simulator.AdvanceTo(current_time + SETTLE_TIME)
    print(f"  Bottle settled. Simulation time: {simulator.get_context().get_time():.2f}s")

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

    # Plot prior belief (uniform distribution before any sensing)
    prior_belief = np.array([0.5, 0.5])  # Initial uniform belief
    plot_belief_bar_chart(
        belief=prior_belief,
        bins=['Bin 0', 'Bin 1'],
        title=f'Prior Belief (Maximum Entropy)\nTrue Location: Bin {true_bin}',
        filename='prior_belief.png',
        true_bin=true_bin
    )

    # Start Meshcat recording
    meshcat.StartRecording()

    print("\nMeshcat running at http://localhost:7000")
    print("Running simulation... (Press Ctrl+C to exit)")
    print()

    # Run simulation loop
    point_clouds_visualized = False
    posterior_belief_plotted = False
    try:
        while True:
            current_time = simulator.get_context().get_time()
            
            # Get state BEFORE advancing simulation (to detect transitions)
            state_before = planner.get_state()
            
            simulator.AdvanceTo(current_time + 0.1)
            
            # Get state AFTER advancing simulation
            state_after = planner.get_state()
            
            # Detect when we've moved past RRBT_EXECUTING (entered POSE_ESTIMATION or later)
            # This is when the arm has reached the light region and belief has been updated
            if (state_before == PlannerState.RRBT_EXECUTING and 
                state_after != PlannerState.RRBT_EXECUTING and
                not posterior_belief_plotted):
                
                # Get current belief from belief estimator
                sim_context = simulator.get_context()
                belief_est_context = diagram.GetSubsystemContext(belief_estimator, sim_context)
                posterior_belief = belief_estimator.GetOutputPort("belief").Eval(belief_est_context)
                
                # Plot posterior belief after first RRBT
                plot_belief_bar_chart(
                    belief=posterior_belief,
                    bins=['Bin 0', 'Bin 1'],
                    title=f'Posterior Belief after RRBT (Light Region)\nTrue Location: Bin {true_bin}',
                    filename='posterior_belief_rrbt.png',
                    true_bin=true_bin,
                    confidence_threshold=float(config.planner.max_bin_uncertainty)
                )
                posterior_belief_plotted = True
            
            # Visualize point clouds once after first step
            if not point_clouds_visualized:
                sim_context = simulator.get_context()
                for camera_name, port_name in point_cloud_output_ports.items():
                    pcl = diagram.GetOutputPort(port_name).Eval(sim_context)
                    meshcat.SetObject(
                        f"{camera_name}.cloud",
                        pcl,
                        point_size=0.003,
                    )
                print("  Point clouds visualized (one-time)")
                point_clouds_visualized = True
            
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
