#!/usr/bin/env python3
"""
Script to generate a block diagram of the Drake Execution Diagram.
This mirrors the execution architecture from main.py:
  TrajectorySource -> Station -> LightDarkRegionSystem -> BeliefEstimator -> BeliefVisualizer

Uses Graphviz (dot) to render the diagram to a PNG file.
"""

import numpy as np
import subprocess
import os
from pathlib import Path
from manipulation.station import MakeHardwareStation, LoadScenario
from pydrake.all import (
    DiagramBuilder,
    ConstantVectorSource,
    PiecewisePolynomial,
    TrajectorySource,
)

# Import project specific modules
# Ensure we can import from src
import sys
sys.path.append(os.getcwd())

from src.perception.light_and_dark import BinLightDarkRegionSensorSystem
from src.estimation.belief_estimator import BinBeliefEstimatorSystem
from src.visualization.belief_visualizer import BeliefVisualizerSystem
from src.utils.config_loader import load_config


def generate_diagram(output_file="system_diagram.png"):
    print("Building Execution Diagram structure...")
    
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Load Scenario
    scenario_path = Path(__file__).parent / "config" / "scenario.yaml"
    if not scenario_path.exists():
        print(f"Error: Scenario file not found at {scenario_path}")
        return

    with open(scenario_path, "r") as f:
        scenario = LoadScenario(data=f.read())

    builder = DiagramBuilder()
    
    # Create Hardware Station (RobotDiagram with SimIiwaDriver)
    # We pass meshcat=None since we don't need visualization for the diagram structure
    station = builder.AddSystem(MakeHardwareStation(scenario=scenario, meshcat=None))
    station.set_name("ExecutionStation")
    plant = station.GetSubsystemByName("plant")

    # ====== TrajectorySource (replaces ConstantVectorSource for iiwa) ======
    # Create a dummy trajectory (2 points) to instantiate TrajectorySource
    # This represents the planned path from RRBT/RRT
    q_home = np.array(config.simulation.q_home)
    q_goal = np.array(config.simulation.q_goal)
    
    # Create a simple 2-point trajectory for diagram generation
    times = np.array([0.0, 1.0])
    positions = np.column_stack([q_home, q_goal])  # 7x2 matrix
    dummy_trajectory = PiecewisePolynomial.FirstOrderHold(times, positions)
    
    traj_source = builder.AddSystem(TrajectorySource(dummy_trajectory))
    traj_source.set_name("PlannedTrajectorySource")
    
    # Connect TrajectorySource to station's iiwa.position input port
    builder.Connect(
        traj_source.get_output_port(),
        station.GetInputPort("iiwa.position")
    )

    # ====== WSG Gripper Source ======
    wsg_position_source = builder.AddSystem(ConstantVectorSource([0.1]))
    wsg_position_source.set_name("GripperPositionSource")
    
    builder.Connect(
        wsg_position_source.get_output_port(),
        station.GetInputPort("wsg.position")
    )

    # ====== Perception System (LightDarkRegionSystem) ======
    perception_sys = builder.AddSystem(
        BinLightDarkRegionSensorSystem(
            plant=plant,
            light_region_center=config.simulation.light_center,
            light_region_size=config.simulation.light_size,
            sigma_light=np.sqrt(float(config.physics.meas_noise_light)),
            sigma_dark=np.sqrt(float(config.physics.meas_noise_dark)),
        )
    )
    perception_sys.set_name("LightDarkPerception")

    # Connect perception to station output
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        perception_sys.GetInputPort("iiwa.position"),
    )

    # ====== Estimation System (BeliefEstimatorSystem) ======
    # Receives measurement variance from LightDarkRegionSystem (single source of truth)
    belief_estimator = builder.AddSystem(
        BinBeliefEstimatorSystem()
    )
    belief_estimator.set_name("BeliefEstimator")
    
    # Connect estimator to perception's measurement_variance output (single source of truth)
    builder.Connect(
        perception_sys.GetOutputPort("measurement_variance"),
        belief_estimator.GetInputPort("measurement_variance"),
    )

    # ====== Visualization System (BeliefVisualizerSystem) ======
    # Note: We pass meshcat=None for diagram generation (no actual visualization)
    # Receives covariance from BeliefEstimatorSystem
    belief_viz = builder.AddSystem(
        BeliefVisualizerSystem(
            meshcat=None,  # No Meshcat for diagram generation
            plant=plant,
            goal_config=config.simulation.q_goal,
        )
    )
    belief_viz.set_name("BeliefVisualizer")
    
    # Connect visualizer to estimator output
    builder.Connect(
        belief_estimator.GetOutputPort("covariance"),
        belief_viz.GetInputPort("covariance"),
    )

    # Build diagram
    diagram = builder.Build()
    diagram.set_name("ExecutionDiagram")
    
    # Generate Graphviz string
    print("Generating Graphviz string...")
    graphviz_str = diagram.GetGraphvizString()
    
    # Save to .dot file
    dot_file = "system_diagram.dot"
    with open(dot_file, "w") as f:
        f.write(graphviz_str)
    print(f"Saved dot file to {dot_file}")
    
    # Convert to PNG
    print(f"Converting to {output_file}...")
    try:
        subprocess.run(["dot", "-Tpng", dot_file, "-o", output_file], check=True)
        print(f"✓ Successfully generated block diagram: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running dot: {e}")
    except FileNotFoundError:
        print("✗ Error: 'dot' command not found. Please install graphviz (sudo apt-get install graphviz).")


if __name__ == "__main__":
    generate_diagram()
