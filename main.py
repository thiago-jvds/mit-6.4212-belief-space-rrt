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
from pathlib import Path
from manipulation.station import MakeHardwareStation, load_scenario
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    Meshcat,
    MeshcatParams,
    ConstantVectorSource,
)

def main():
    print("=" * 60)
    print("MIT 6.4212 - Belief-Space RRT")
    print("Robot Manipulation Visualization")
    print("=" * 60)
    
    # Create Meshcat on port 7000 specifically
    print("\nStarting Meshcat on port 7000...")
    try:
        params = MeshcatParams()
        params.port = 7000
        meshcat = Meshcat(params=params)
        print(f"✓ Meshcat is running at: {meshcat.web_url()}")
        print(f"  Open this URL in your browser: {meshcat.web_url()}")
    except RuntimeError as e:
        print(f"\n✗ ERROR: Could not start Meshcat on port 7000")
        print(f"  {e}")
        print(f"\n  Port 7000 is likely already in use.")
        print(f"  Please stop any other Meshcat servers or Python processes using that port.")
        raise
    
    # Load the scenario
    print("\nLoading scenario from config/scenario.yaml...")
    scenario_path = Path(__file__).parent / "config" / "scenario.yaml"
    
    with open(scenario_path, "r") as f:
        scenario = load_scenario(data=f.read())
    print("✓ Scenario loaded")
    
    # Build the system
    print("\nBuilding robot system...")
    builder = DiagramBuilder()
    station = builder.AddSystem(
        MakeHardwareStation(scenario=scenario, meshcat=meshcat)
    )
    
    # Set robot joint positions (poses the robot)
    iiwa_position_source = builder.AddSystem(
        ConstantVectorSource(np.array([0, 0.1, 0, -1.2, 0, 0.8, 0]))
    )
    builder.Connect(
        iiwa_position_source.get_output_port(),
        station.GetInputPort("iiwa.position")
    )
    
    # Set gripper position (0.1m = fully open)
    wsg_position_source = builder.AddSystem(ConstantVectorSource([0.1]))
    builder.Connect(
        wsg_position_source.get_output_port(),
        station.GetInputPort("wsg.position")
    )
    
    # Build diagram and create simulator
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    
    print("✓ System built successfully")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION READY")
    print("=" * 60)
    print(f"Meshcat URL: {meshcat.web_url()}")
    print("\nThe robot scene is now visible in your browser.")
    print("Press Ctrl+C to exit.")
    print("=" * 60)
    
    # Run simulation and keep it alive
    try:
        print("\nSimulation running...")
        # Advance to initialize the visualization
        simulator.AdvanceTo(0.1)
        
        # Keep running indefinitely
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        print("Goodbye!")


if __name__ == "__main__":
    main()

