#!/usr/bin/env python3
"""Unified RRBT-RRT planning and execution with Meshcat visualization."""

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


RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np_rng = np.random.default_rng(seed=RANDOM_SEED)


def place_mustard_bottle_randomly_in_bin(meshcat, plant, plant_context, true_bin, np_rng: np.random.Generator):        
    """Place mustard bottle randomly within the specified bin. Returns X_WM."""
    true_bin_instance = plant.GetModelInstanceByName(f"bin{true_bin}")
    true_bin_body = plant.GetBodyByName("bin_base", true_bin_instance)
    X_WB = plant.EvalBodyPoseInWorld(plant_context, true_bin_body)
    
    random_rotation = RollPitchYaw(-np.pi/2, 0, np_rng.uniform(0, 2*np.pi)).ToRotationMatrix()
    x_offset = -0.01
    x_range = (-0.01+x_offset, 0.01+x_offset)
    y_range = (-0.15, 0.15)
    random_z = 0.2

    random_x = np_rng.uniform(x_range[0], x_range[1])
    random_y = np_rng.uniform(y_range[0], y_range[1])

    box_width = x_range[1] - x_range[0]
    box_depth = y_range[1] - y_range[0]
    box_height = 0.02
    init_space_box = Box(box_width, box_depth, box_height)

    box_center_in_bin = [
        (x_range[0] + x_range[1]) / 2,
        (y_range[0] + y_range[1]) / 2,
        random_z
    ]
    X_WBox = X_WB.multiply(RigidTransform(box_center_in_bin))
    meshcat.SetTransform("init_space", X_WBox)

    X_BM = RigidTransform(random_rotation, [random_x, random_y, random_z])
    X_WM = X_WB.multiply(X_BM)

    mustard_body = plant.GetBodyByName("base_link_mustard")
    plant.SetFreeBodyPose(plant_context, mustard_body, X_WM)
    
    print(f"  Placed mustard bottle in bin{true_bin}:")
    print(f"    Bin position: {X_WB.translation()}")
    print(f"    Random offset: [{random_x:.3f}, {random_y:.3f}, {random_z:.3f}]")
    print(f"    World position: {X_WM.translation()}")
    
    return X_WM


def probability_to_color(prob: float) -> tuple:
    """Convert probability to RGB color: red (0) - yellow (0.5) - green (1)."""
    prob = np.clip(prob, 0.0, 1.0)
    if prob <= 0.5:
        r, g, b = 1.0, prob * 2.0, 0.0
    else:
        r, g, b = 1.0 - (prob - 0.5) * 2.0, 1.0, 0.0
    return (r, g, b)


def plot_belief_bar_chart(belief: np.ndarray, bins: list, title: str, filename: str, 
                          true_bin: int = None, confidence_threshold: float = None):
    """Generate and save a matplotlib bar chart for belief visualization."""
    plt.figure(figsize=(8, 6))
    colors = [probability_to_color(p) for p in belief]
    bars = plt.bar(bins, belief, color=colors, alpha=0.8, edgecolor='black')
    
    plt.title(title, fontsize=14)
    plt.ylabel('Probability', fontsize=12)
    plt.xlabel('Bin Location', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    if confidence_threshold is not None:
        threshold_y = 1.0 - confidence_threshold
        plt.axhline(y=threshold_y, color='black', linestyle='--', linewidth=2, 
                    label=f'Confidence Threshold ({threshold_y:.2f})')
        plt.legend(loc='upper right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  Saved belief plot: {filename}")


def save_block_diagram(diagram, output_stem: Path):
    """Save Drake diagram to Graphviz .dot and .png files."""
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

    config = load_config()
    print("Loaded Configuration:")
    print(f"    > Physics: Q_scale={config.physics.process_noise_scale}")
    print(
        f"    > Planner: max_bin_uncertainty={config.planner.max_bin_uncertainty}, "
        f"LightBias={config.planner.bias_prob_sample_q_bin_light}"
    )
    print()

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

    if args.visualize == "True":
        restore_camera_pose(meshcat)

    scenario_path = Path(__file__).parent / "config" / "scenario.yaml"
    with open(scenario_path, "r") as f:
        scenario = LoadScenario(data=f.read())

    print("\nBuilding unified diagram...")
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario=scenario, meshcat=meshcat))
    station.set_name("HardwareStation")
    plant = station.GetSubsystemByName("plant")

    temp_plant_context = plant.CreateDefaultContext()
    bin0_instance = plant.GetModelInstanceByName("bin0")
    bin0_body = plant.GetBodyByName("bin_base", bin0_instance)
    X_W_bin0 = plant.EvalBodyPoseInWorld(temp_plant_context, bin0_body)
    bin1_instance = plant.GetModelInstanceByName("bin1")
    bin1_body = plant.GetBodyByName("bin_base", bin1_instance)
    X_W_bin1 = plant.EvalBodyPoseInWorld(temp_plant_context, bin1_body)
    X_bin0_chart = RigidTransform([-0.22, 0.29, 0.21])
    X_bin1_chart = RigidTransform([-0.22, -0.29, 0.21])
    
    print(f"  Bin0 position: {X_W_bin0.translation()}")
    print(f"  Bin1 position: {X_W_bin1.translation()}")
    print(f"  Chart offset from bin0: {X_bin0_chart.translation()}")
    print(f"  Chart offset from bin1: {X_bin1_chart.translation()}")

    planner = builder.AddSystem(PlannerSystem(plant, config, meshcat, scenario_path, rng=np_rng))
    planner.set_name("PlannerSystem")
    builder.Connect(
        planner.GetOutputPort("iiwa_position_command"),
        station.GetInputPort("iiwa.position")
    )
    builder.Connect(
        planner.GetOutputPort("wsg_position_command"),
        station.GetInputPort("wsg.position")
    )

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
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        bin_perception_sys.GetInputPort("iiwa.position"),
    )
    
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
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        mustard_position_perception_sys.GetInputPort("iiwa.position"),
    )

    belief_estimator = builder.AddSystem(
        BinBeliefEstimatorSystem(
            n_bins=2,
            true_bin=0,  # Placeholder, will be updated
            max_bin_uncertainty=float(config.planner.max_bin_uncertainty),
            rng=np_rng,
        )
    )
    belief_estimator.set_name("BinBeliefEstimator")
    builder.Connect(
        bin_perception_sys.GetOutputPort("sensor_model"),
        belief_estimator.GetInputPort("sensor_model")
    )

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
    builder.Connect(
        belief_estimator.GetOutputPort("belief"),
        belief_viz.GetInputPort("belief")
    )

    print("  Adding point cloud generation from cameras...")
    to_point_cloud = AddPointClouds(
        scenario=scenario,
        station=station,
        builder=builder,
        meshcat=None,  # Don't visualize continuously - we'll do it once
    )
    print(f"    Added point cloud converters for: {list(to_point_cloud.keys())}")
    point_cloud_output_ports = {}
    for camera_name, converter in to_point_cloud.items():
        port_name = f"{camera_name}_point_cloud"
        builder.ExportOutput(converter.get_output_port(), port_name)
        point_cloud_output_ports[camera_name] = port_name

    print("  Adding MustardPoseEstimatorSystem...")
    pose_estimator = builder.AddSystem(
        MustardPoseEstimatorSystem(meshcat=meshcat, n_bins=2)
    )
    pose_estimator.set_name("MustardPoseEstimator")

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

    builder.Connect(
        belief_estimator.GetOutputPort("belief"),
        pose_estimator.GetInputPort("belief")
    )
    print("    Connected belief to pose estimator")

    builder.Connect(
        belief_estimator.GetOutputPort("belief_confident"),
        pose_estimator.GetInputPort("estimation_trigger")
    )
    print("    Connected belief_confident trigger to pose estimator")

    builder.Connect(
        pose_estimator.GetOutputPort("estimated_pose"),
        planner.GetInputPort("estimated_mustard_pose")
    )
    print("    Connected pose estimator to planner")

    print("  Adding MustardPositionBeliefEstimatorSystem...")
    mustard_belief_estimator = builder.AddSystem(
        MustardPositionBeliefEstimatorSystem(
            initial_uncertainty=float(config.planner.mustard_position_initial_uncertainty),
        )
    )
    mustard_belief_estimator.set_name("MustardPositionBeliefEstimator")

    builder.Connect(
        mustard_position_perception_sys.GetOutputPort("measurement_variance"),
        mustard_belief_estimator.GetInputPort("measurement_variance")
    )
    print("    Connected measurement_variance from MustardPositionPerception")

    builder.Connect(
        pose_estimator.GetOutputPort("estimated_pose"),
        mustard_belief_estimator.GetInputPort("estimated_pose")
    )
    print("    Connected estimated_pose from MustardPoseEstimator")

    print("  Adding CovarianceEllipsoidSystem...")
    covariance_viz = builder.AddSystem(
        CovarianceEllipsoidSystem(
            meshcat=meshcat,
            scale_factor=3.0,
            color=Rgba(1.0, 0.0, 0.0, 0.5),
        )
    )
    covariance_viz.set_name("CovarianceEllipsoid")

    builder.Connect(
        mustard_belief_estimator.GetOutputPort("position_mean"),
        covariance_viz.GetInputPort("position")
    )
    builder.Connect(
        mustard_belief_estimator.GetOutputPort("covariance"),
        covariance_viz.GetInputPort("covariance")
    )
    print("    Connected position and covariance to ellipsoid visualizer")

    builder.Connect(
        mustard_belief_estimator.GetOutputPort("covariance"),
        planner.GetInputPort("position_covariance")
    )
    print("    Connected covariance to planner for grasp planning")

    diagram = builder.Build()
    diagram.set_name("UnifiedPlanningDiagram")

    if args.generate_block_diagram:
        print("\nGenerating block diagram...")
        save_block_diagram(diagram, Path(args.diagram_output_stem))
        print("Block diagram generated. Exiting before simulation.")
        return

    print("\nInitializing simulator...")
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    if args.visualize == "True":
        meshcat.SetObject(
            "bin_light_region_indicator",
            Box(*config.simulation.bin_light_size),
            Rgba(0, 1, 0, 0.3),
        )
        meshcat.SetTransform(
            "bin_light_region_indicator",
            RigidTransform(RotationMatrix(), config.simulation.bin_light_center),
        )
        meshcat.SetObject(
            "mustard_position_light_region_indicator",
            Box(*config.simulation.mustard_position_light_size),
            Rgba(1.0, 0.5, 0.0, 0.3),
        )
        meshcat.SetTransform(
            "mustard_position_light_region_indicator",
            RigidTransform(RotationMatrix(), config.simulation.mustard_position_light_center),
        )

    simulator.AdvanceTo(0.1)

    print("\n" + "=" * 60)
    print("ENVIRONMENT SETUP")
    print("=" * 60)

    sim_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(sim_context)

    true_bin = np_rng.integers(0, 2)
    print(f"Randomly chosen true bin: {true_bin}")

    X_WM_mustard = place_mustard_bottle_randomly_in_bin(
        meshcat, plant, plant_context, true_bin, np_rng
    )

    diagram.ForcedPublish(sim_context)

    SETTLE_TIME = 2.0
    print(f"\nLetting mustard bottle settle for {SETTLE_TIME}s...")
    current_time = simulator.get_context().get_time()
    simulator.AdvanceTo(current_time + SETTLE_TIME)
    print(f"  Bottle settled. Simulation time: {simulator.get_context().get_time():.2f}s")

    belief_estimator._true_bin = true_bin

    print("\n" + "=" * 60)
    print("STARTING UNIFIED PLANNING AND EXECUTION")
    print("=" * 60)

    planner.configure_for_execution(true_bin, X_WM_mustard)

    prior_belief = np.array([0.5, 0.5])
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

    meshcat.StopRecording()
    meshcat.PublishRecording()

    print("\nExecution complete! Replay available in Meshcat.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
