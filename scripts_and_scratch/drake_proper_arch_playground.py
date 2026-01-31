import time
import numpy as np
import subprocess
from pathlib import Path
from pydrake.all import (
    DiagramBuilder, 
    RigidTransform, RollPitchYaw, LeafSystem, AbstractValue,
    ImageDepth32F, RotationMatrix,
    InverseKinematics, Solve, BasicVector, Simulator,
    ConstantVectorSource, Meshcat, MeshcatParams
)
from manipulation.station import MakeHardwareStation, LoadScenario


def save_diagram_as_png(diagram, output_file: str = "diagram.png") -> bool:
    """
    Generate and save a block diagram of a Drake Diagram as a PNG file.
    
    Args:
        diagram: A Drake Diagram object (must be built)
        output_file: Path to save the PNG file (default: "diagram.png")
        
    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_file)
    dot_file = output_path.with_suffix(".dot")
    
    # Generate Graphviz string from diagram
    graphviz_str = diagram.GetGraphvizString()
    
    # Save intermediate .dot file
    with open(dot_file, "w") as f:
        f.write(graphviz_str)
    
    # Convert to PNG using Graphviz
    try:
        subprocess.run(
            ["dot", "-Tpng", str(dot_file), "-o", str(output_path)],
            check=True,
            capture_output=True
        )
        print(f"✓ Block diagram saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating PNG: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("✗ Error: 'dot' command not found. Install graphviz: sudo apt-get install graphviz")
        return False


# Start Meshcat visualizer
params = MeshcatParams()
params.port = 7000
meshcat = Meshcat(params=params)

# ---------------------------------------------------------
# COMPONENT 1: THE MOCK PERCEPTION SYSTEM
# ---------------------------------------------------------
class MockPerceptionSystem(LeafSystem):
    """
    Simulates a vision system. 
    In reality, this would take an Image, run a neural net, and find a pose.
    Here, we mock it by outputting a fixed target pose relative to the camera.
    """
    def __init__(self):
        LeafSystem.__init__(self)
        
        # Input Port: Expects a Depth Image
        # We declare it Abstract because an Image is not a simple Vector
        self.DeclareAbstractInputPort(
            "camera_depth_image", 
            AbstractValue.Make(ImageDepth32F(640, 480))
        )
        
        # Output Port: The detected object pose (RigidTransform)
        self.DeclareAbstractOutputPort(
            "target_pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcObjectPose
        )
        self.perception_complete = False
        self.perception_X_WO = None

    def CalcObjectPose(self, context, output):
        
        if not self.perception_complete:
            print("Mock Perception System: Calculating object pose...")
            for i in range(5):
                print(f"Mock Perception System: Calculating object pose... {i+1}/5")
                time.sleep(1)
            self.perception_X_WO = RigidTransform(RollPitchYaw(0, np.pi, 0), [0.5, 0, 0.3])
            self.perception_complete = True
            print("Mock Perception System: Object pose calculated.")
        
        output.set_value(self.perception_X_WO)

# ---------------------------------------------------------
# COMPONENT 2: THE KINEMATIC PLANNER
# ---------------------------------------------------------
class IKPlannerSystem(LeafSystem):
    """
    Takes a desired Cartesian Pose and computes the Joint Angles
    to reach that pose using Inverse Kinematics.
    Generated a trajectory from q_home to q_goal.
    """
    def __init__(self, plant, q_home):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa_model = plant.GetModelInstanceByName("iiwa")
        self._wsg_model = plant.GetModelInstanceByName("wsg")
        self._G = plant.GetBodyByName("body", self._wsg_model).body_frame() # The gripper frame
        self._W = plant.world_frame()
        self._q_home = np.array(q_home)

        # Input: The Target Pose (from Perception)
        self.DeclareAbstractInputPort(
            "target_pose", AbstractValue.Make(RigidTransform()))
        
        # Output: Joint Positions (7 DoF for IIWA)
        self.DeclareVectorOutputPort(
            "iiwa_position_command", BasicVector(7), self.CalcJointCommand)

    def CalcJointCommand(self, context, output):
        print("IKPlannerSystem: Calculating joint command...")
        # 1. Get the target pose from the input port
        X_WT = self.get_input_port(0).Eval(context)

        # 2. Setup Inverse Kinematics (IK)
        ik = InverseKinematics(self._plant, self._plant_context)
        
        # Constraint: Gripper frame (G) matches Target frame (T)
        ik.AddPositionConstraint(
            frameB=self._G, p_BQ=[0,0,0],
            frameA=self._W, p_AQ_lower=X_WT.translation(), p_AQ_upper=X_WT.translation())
        
        ik.AddOrientationConstraint(
            frameAbar=self._W, R_AbarA=X_WT.rotation(),
            frameBbar=self._G, R_BbarB=RotationMatrix(),
            theta_bound=0.001)

        # 3. Solve IK
        prog = ik.get_mutable_prog()
        # Create initial guess with correct size (full plant q, not just iiwa)
        # Get default positions from plant context
        q_guess = self._plant.GetPositions(self._plant_context)
        # Set a comfortable iiwa pose for the first 7 joints
        iiwa_q_guess = [0, 0, 0, -1.5, 0, 1.0, 0]
        q_guess[:7] = iiwa_q_guess
        prog.SetInitialGuess(ik.q(), q_guess)
        result = Solve(ik.prog())

        if result.is_success():
            q_sol = result.GetSolution(ik.q())[:7]
        else:
            print("IK Failed to converge")
            q_sol = self._q_home

        # 4. Interpolate trajectory (Move from Home to Goal over 5 seconds)
        t = context.get_time()
        duration = 5.0
        
        # Simple cubic interpolation (smoothstep)
        if t >= duration:
            alpha = 1.0
        else:
            # Normalized time
            tau = t / duration
            # 3*t^2 - 2*t^3 (SmoothStep)
            alpha = 3 * tau**2 - 2 * tau**3
            
        q_cmd = (1 - alpha) * self._q_home + alpha * q_sol
        output.SetFromVector(q_cmd)

# ---------------------------------------------------------
# MAIN: WIRING THE DIAGRAM
# ---------------------------------------------------------
builder = DiagramBuilder()

# 1. Load the scenario and create the hardware station (like main.py)
scenario_path = Path(__file__).parent / "config" / "scenario.yaml"
with open(scenario_path, "r") as f:
    scenario = LoadScenario(data=f.read())

station = builder.AddSystem(MakeHardwareStation(scenario=scenario, meshcat=meshcat))

# Get the plant from the hardware station
plant = station.GetSubsystemByName("plant")

# Define Home Position (matches scenario.yaml default)
q_home = [0, 0.1, 0, -1.2, 0, 1.6, 0]

# 2. Add Perception and Planning Components
perception = builder.AddSystem(MockPerceptionSystem())
planner = builder.AddSystem(IKPlannerSystem(plant, q_home))

# 3. Wire the Connections
# A. Perception -> Planner
# (Note: MockPerceptionSystem doesn't actually use the depth image input,
#  it just outputs a fixed mock pose. We leave the input unconnected for now.)
builder.Connect(
    perception.GetOutputPort("target_pose"),
    planner.GetInputPort("target_pose")
)

# B. Planner -> Robot Position Command
# Connect planner output to the hardware station's iiwa position input
builder.Connect(
    planner.GetOutputPort("iiwa_position_command"),
    station.GetInputPort("iiwa.position")
)

# C. WSG Gripper Position (constant open position)
wsg_position_source = builder.AddSystem(ConstantVectorSource([0.1]))
builder.Connect(
    wsg_position_source.get_output_port(),
    station.GetInputPort("wsg.position")
)

print("Diagram Built successfully.")
diagram = builder.Build()

# Save block diagram as PNG
save_diagram_as_png(diagram, "playground_diagram.png")

# 4. Run Simulation
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
context = simulator.get_mutable_context()

print("\nMeshcat running at http://localhost:7000")
print("Press Ctrl+C to exit.")

# Run simulation loop
try:
    while True:
        # Step forward by small increments to visualize movement
        current_time = simulator.get_context().get_time()
        simulator.AdvanceTo(current_time + 0.1)
except KeyboardInterrupt:
    print("\nExiting...")