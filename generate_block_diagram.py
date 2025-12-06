#!/usr/bin/env python3
"""
Script to generate a block diagram of the Drake system defined in main.py.
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
)

# Import project specific modules
# Ensure we can import from src
import sys
sys.path.append(os.getcwd())

from src.perception.light_and_dark import LightDarkRegionSystem
from src.utils.config_loader import load_rrbt_config

def generate_diagram(output_file="system_diagram.png"):
    print("Building system diagram...")
    
    # Load configuration
    try:
        config = load_rrbt_config()
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
    
    # Create Hardware Station
    # We pass meshcat=None since we don't need visualization for the diagram structure
    station = builder.AddSystem(MakeHardwareStation(scenario=scenario, meshcat=None))
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
    perception_sys.set_name("LightDarkRegionSystem")

    # Connections
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        perception_sys.GetInputPort("iiwa.position"),
    )

    # Set robot joint positions sources
    q_home = config.simulation.q_home
    iiwa_position_source = builder.AddSystem(ConstantVectorSource(q_home))
    iiwa_position_source.set_name("IiwaPositionSource")
    
    builder.Connect(
        iiwa_position_source.get_output_port(), station.GetInputPort("iiwa.position")
    )

    wsg_position_source = builder.AddSystem(ConstantVectorSource([0.1]))
    wsg_position_source.set_name("WsgPositionSource")
    
    builder.Connect(
        wsg_position_source.get_output_port(), station.GetInputPort("wsg.position")
    )

    # Build diagram
    diagram = builder.Build()
    diagram.set_name("MainSystemDiagram")
    
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

