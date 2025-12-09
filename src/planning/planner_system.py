"""
Planner System - A Drake LeafSystem for sequential RRBT planning stages.

This system implements a state machine that:
1. Initially outputs q_home while waiting for configuration
2. Runs RRBT planning (blocking) for bin belief information gathering
3. Executes the RRBT trajectory
4. Runs pose estimation from point clouds (ICP)
5. Runs RRBT2 planning (blocking) for mustard position uncertainty reduction
6. Executes the RRBT2 trajectory (ellipsoid shrinks as robot enters light region)
7. Runs grasp planning (samples from covariance, selects best grasp, computes pre-grasp)
8. Holds at the final position when complete
"""

from enum import Enum, auto
import numpy as np
from pydrake.all import (
    LeafSystem,
    BasicVector,
    AbstractValue,
    PiecewisePolynomial,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    PointCloud,
    Fields,
    BaseField,
    Rgba,
)

from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.mustard_depth_camera_example import MustardPointCloud

from src.simulation.simulation_tools import (
    IiwaProblem,
    IiwaProblemBinBelief,
    IiwaProblemMustardPositionBelief,
)
from src.planning.belief_space_rrt import rrbt_planning
from src.planning.standard_rrt import rrt_planning
from src.utils.ik_solver import solve_ik_for_pose
from src.grasping.grasp_selection import (
    select_best_grasp,
    compute_pregrasp_pose,
    sample_position_from_covariance,
    draw_grasp_candidate,
)


class PlannerState(Enum):
    """State machine states for the planner."""
    IDLE = auto()              # Waiting for setup
    RRBT_PLANNING = auto()     # Running RRBT for bin belief (blocking)
    RRBT_EXECUTING = auto()    # Playing RRBT trajectory
    POSE_ESTIMATION = auto()   # Running pose estimation from point clouds
    RRBT2_PLANNING = auto()    # Running RRBT for position belief (blocking)
    RRBT2_EXECUTING = auto()   # Playing RRBT2 trajectory
    GRASP_PLANNING = auto()    # Compute grasp and pre-grasp poses from covariance
    GRASP_EXECUTING = auto()   # Execute grasp trajectory (pregrasp → grasp → lift)
    COMPLETE = auto()          # Done, holding at final position


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


class PlannerSystem(LeafSystem):
    """
    A Drake LeafSystem that performs sequential RRBT planning.
    
    State Machine:
        IDLE -> RRBT_PLANNING -> RRBT_EXECUTING -> POSE_ESTIMATION -> 
        RRBT2_PLANNING -> RRBT2_EXECUTING -> COMPLETE
    
    Outputs:
        - iiwa_position_command: 7D joint position command
    """
    
    def __init__(self, plant, config, meshcat, scenario_path):
        """
        Initialize the PlannerSystem.
        
        Args:
            plant: MultibodyPlant reference (for IK solving)
            config: RRBT configuration namespace (from load_rrbt_config)
            meshcat: Meshcat visualizer instance
            scenario_path: Path to scenario.yaml file
        """
        LeafSystem.__init__(self)
        
        # Configuration
        self._plant = plant
        self._config = config
        self._meshcat = meshcat
        self._scenario_path = scenario_path
        
        # Positions (computed in constructor via IK)
        self._q_home = np.array(config.simulation.q_home)
        self._q_goal = None      # Computed via IK from tf_goal
        self._q_bin_light_hint = None  # Computed via IK from bin_light_center
        self._q_mustard_position_light_hint = None  # Computed via IK from mustard_position_light_center
        
        # State machine
        self._state = PlannerState.IDLE
        
        # RRBT stage 1 (bin belief)
        self._rrbt_trajectory = None
        self._rrbt_start_time = None
        self._rrbt_end_position = None
        self._pred_q_goal = None  # Goal predicted by RRBT belief
        
        # RRBT stage 2 (mustard position belief)
        self._rrbt2_trajectory = None
        self._rrbt2_start_time = None
        self._rrbt2_end_position = None
        self._estimated_mustard_position = None  # 3D position from ICP
        
        # Runtime parameters (set by configure_for_execution)
        self._true_bin = None
        self._X_WM_mustard = None
        
        # Compute IK targets
        self._compute_ik_targets()
        
        # Instance variable to store the estimated mustard pose
        self._estimated_mustard_pose = None
        
        # Input port: estimated mustard pose from MustardPoseEstimatorSystem
        self._pose_input_port = self.DeclareAbstractInputPort(
            "estimated_mustard_pose",
            AbstractValue.Make(RigidTransform())
        )
        
        # Input port: position covariance from MustardPositionBeliefEstimatorSystem (2x2 flattened to 4, X-Y only)
        self._covariance_input_port = self.DeclareVectorInputPort(
            "position_covariance",
            4
        )
        
        # Grasp planning results
        self._best_grasp_pose = None
        self._pregrasp_pose = None
        self._grasp_candidates = []  # List of (cost, X_G) tuples for IK validation
        self._rng = np.random.default_rng()  # Random generator for grasp sampling
        
        # Grasp execution variables
        self._grasp_trajectory = None
        self._grasp_start_time = None
        self._grasp_close_time = None  # Time offset when gripper should close
        self._grasp_open_time = None   # Time offset when gripper should open (release)
        self._grasp_end_position = None
        self._gripper_command = 0.1  # Start open (0.1m = fully open)
        
        # Drop pose variables
        self._drop_pose = None
        
        # Create plant context for forward kinematics (gripper visualization)
        self._fk_plant_context = self._plant.CreateDefaultContext()
        self._gripper_body = self._plant.GetBodyByName("body", self._plant.GetModelInstanceByName("wsg"))
        
        # Add Meshcat triad for gripper visualization
        # AddMeshcatTriad(self._meshcat, "gripper_frame", length=0.1, radius=0.004)
        
        # Output port: joint position command
        self.DeclareVectorOutputPort(
            "iiwa_position_command",
            BasicVector(7),
            self.CalcJointCommand
        )
        
        # Output port: gripper position command
        self.DeclareVectorOutputPort(
            "wsg_position_command",
            BasicVector(1),
            self.CalcGripperCommand
        )
        
        print(f"PlannerSystem initialized:")
        print(f"  q_home: {self._q_home}")
        print(f"  q_goal: {self._q_goal}")
        print(f"  q_bin_light_hint: {self._q_bin_light_hint}")
        print(f"  q_mustard_position_light_hint: {self._q_mustard_position_light_hint}")
    
    def configure_for_execution(self, true_bin, X_WM_mustard=None):
        """
        Configure the planner for execution after mustard bottle is placed.
        
        This method must be called after the diagram is built and the mustard
        bottle has been positioned. It transitions the state machine from IDLE
        to RRBT_PLANNING.
        
        Args:
            true_bin: The ground truth bin index (0 or 1)
            X_WM_mustard: RigidTransform of mustard bottle in world frame (optional)
        """
        self._true_bin = true_bin
        self._X_WM_mustard = X_WM_mustard
        self._state = PlannerState.RRBT_PLANNING
        print(f"PlannerSystem configured for execution:")
        print(f"  true_bin: {true_bin}")
        print(f"  State: {self._state.name}")
    
    def get_state(self) -> PlannerState:
        """Return the current planner state."""
        return self._state
    
    def is_complete(self) -> bool:
        """Check if planning and execution are complete."""
        return self._state == PlannerState.COMPLETE
    
    def CalcJointCommand(self, context, output):
        """
        Calculate the joint position command based on current state.
        
        This is called by Drake's simulation loop to get the output value.
        The state machine transitions happen here based on time and completion.
        """
        t = context.get_time()
        q_command = self._q_home  # Default to home position
        
        if self._state == PlannerState.IDLE:
            # Waiting for configuration - hold at home
            q_command = self._q_home
            
        elif self._state == PlannerState.RRBT_PLANNING:
            # Run RRBT planning for bin belief (blocking)
            print(f"\n{'='*60}")
            print("RRBT PLANNING PHASE (Bin Belief)")
            print(f"{'='*60}")
            self._run_rrbt_planning()
            self._rrbt_start_time = t
            self._state = PlannerState.RRBT_EXECUTING
            print(f"RRBT planning complete. Starting trajectory execution at t={t:.2f}s")
            print(f"  RRBT trajectory duration: {self._rrbt_trajectory.end_time():.2f}s")
            q_command = self._q_home
            
        elif self._state == PlannerState.RRBT_EXECUTING:
            t_traj = t - self._rrbt_start_time
            if t_traj >= self._rrbt_trajectory.end_time():
                # RRBT trajectory complete, transition to POSE_ESTIMATION
                self._rrbt_end_position = self._rrbt_trajectory.value(
                    self._rrbt_trajectory.end_time()
                ).flatten()
                self._state = PlannerState.POSE_ESTIMATION
                print(f"\nRRBT execution complete at t={t:.2f}s")
                print(f"  End position: {np.round(self._rrbt_end_position, 3)}")
                q_command = self._rrbt_end_position
            else:
                q_command = self._rrbt_trajectory.value(t_traj).flatten()
        
        elif self._state == PlannerState.POSE_ESTIMATION:
            # Read estimated pose from input port (triggers MustardPoseEstimatorSystem)
            print(f"\n{'='*60}")
            print("POSE ESTIMATION PHASE")
            print(f"{'='*60}")
            
            self._estimated_mustard_pose = self._pose_input_port.Eval(context)
            self._estimated_mustard_position = self._estimated_mustard_pose.translation().copy()
            
            print(f"  Estimated mustard pose received:")
            print(f"    Translation: {self._estimated_mustard_position}")
            
            # Transition to RRBT2 planning (mustard position belief)
            self._state = PlannerState.RRBT2_PLANNING
            q_command = self._rrbt_end_position
                
        elif self._state == PlannerState.RRBT2_PLANNING:
            # Run RRBT2 for mustard position belief (blocking)
            print(f"\n{'='*60}")
            print("RRBT2 PLANNING PHASE (Position Belief)")
            print(f"{'='*60}")
            self._run_rrbt2_planning()
            self._rrbt2_start_time = t
            self._state = PlannerState.RRBT2_EXECUTING
            print(f"RRBT2 planning complete. Starting trajectory execution at t={t:.2f}s")
            print(f"  RRBT2 trajectory duration: {self._rrbt2_trajectory.end_time():.2f}s")
            q_command = self._rrbt_end_position
            
        elif self._state == PlannerState.RRBT2_EXECUTING:
            t_traj = t - self._rrbt2_start_time
            if t_traj >= self._rrbt2_trajectory.end_time():
                # RRBT2 trajectory complete, transition to GRASP_PLANNING
                self._rrbt2_end_position = self._rrbt2_trajectory.value(
                    self._rrbt2_trajectory.end_time()
                ).flatten()
                self._state = PlannerState.GRASP_PLANNING
                print(f"\nRRBT2 execution complete at t={t:.2f}s")
                print(f"  End position: {np.round(self._rrbt2_end_position, 3)}")
                q_command = self._rrbt2_end_position
            else:
                q_command = self._rrbt2_trajectory.value(t_traj).flatten()
        
        elif self._state == PlannerState.GRASP_PLANNING:
            # Run grasp planning (blocking)
            print(f"\n{'='*60}")
            print("GRASP PLANNING PHASE")
            print(f"{'='*60}")
            self._run_grasp_planning(context)
            
            if self._grasp_candidates:
                # Compute grasp execution trajectory (validates IK for each candidate)
                if self._compute_grasp_trajectory():
                    # Transition to grasp execution
                    self._grasp_start_time = t
                    self._gripper_command = 0.1  # Start with gripper open
                    self._state = PlannerState.GRASP_EXECUTING
                    print(f"\n{'='*60}")
                    print("GRASP EXECUTION PHASE")
                    print(f"{'='*60}")
                    print(f"  Starting grasp execution at t={t:.2f}s")
                    print(f"  Trajectory duration: {self._grasp_trajectory.end_time():.1f}s")
                else:
                    # All candidates failed IK, skip to complete
                    print(f"  All grasp candidates failed IK, skipping execution")
                    self._state = PlannerState.COMPLETE
            else:
                print(f"  No valid grasp candidates found, skipping execution")
                self._state = PlannerState.COMPLETE
            
            q_command = self._rrbt2_end_position
        
        elif self._state == PlannerState.GRASP_EXECUTING:
            # Execute grasp trajectory
            t_traj = t - self._grasp_start_time
            
            # Update gripper command based on trajectory phase
            # Order matters: check open time first (it comes after close time)
            if self._grasp_open_time is not None and t_traj >= self._grasp_open_time:
                self._gripper_command = 0.1  # Open gripper (release object)
            elif t_traj >= self._grasp_close_time:
                self._gripper_command = 0.0  # Close gripper (holding object)
            else:
                self._gripper_command = 0.1  # Keep gripper open (before grasp)
            
            if t_traj >= self._grasp_trajectory.end_time():
                # Grasp execution complete
                self._state = PlannerState.COMPLETE
                print(f"\n{'='*60}")
                print("EXECUTION COMPLETE")
                print(f"{'='*60}")
                print(f"  Final position: {np.round(self._grasp_end_position, 3)}")
                print(f"  Object released into square bin!")
                print(f"  Total time: {t:.2f}s")
                q_command = self._grasp_end_position
            else:
                q_command = self._grasp_trajectory.value(t_traj).flatten()
                # Debug: print progress at key milestones (only once each)
                if not hasattr(self, '_grasp_milestone_printed'):
                    self._grasp_milestone_printed = set()
                
                # Use dynamic milestone times (stored in _compute_grasp_trajectory)
                t_pre = getattr(self, '_grasp_t_pregrasp', 3.0)
                t_grp = getattr(self, '_grasp_t_grasp', 5.0)
                t_lft = getattr(self, '_grasp_t_lift_start', 6.0)
                t_drp = getattr(self, '_grasp_t_drop', 8.0)
                t_rel = getattr(self, '_grasp_t_release', 9.0)
                
                milestone = None
                if t_traj < 0.1:
                    milestone = "start"
                elif abs(t_traj - t_pre) < 0.1:
                    milestone = "pregrasp"
                elif abs(t_traj - t_grp) < 0.1:
                    milestone = "grasp"
                elif abs(t_traj - t_lft) < 0.1:
                    milestone = "lift"
                elif abs(t_traj - t_drp) < 0.1:
                    milestone = "drop"
                elif abs(t_traj - t_rel) < 0.1:
                    milestone = "release"
                
                if milestone and milestone not in self._grasp_milestone_printed:
                    self._grasp_milestone_printed.add(milestone)
                    if milestone == "start":
                        print(f"  [t={t_traj:.2f}s] Starting grasp trajectory execution")
                    elif milestone == "pregrasp":
                        print(f"  [t={t_traj:.2f}s] Reached PREGRASP pose")
                    elif milestone == "grasp":
                        print(f"  [t={t_traj:.2f}s] Reached GRASP pose - CLOSING GRIPPER")
                    elif milestone == "lift":
                        print(f"  [t={t_traj:.2f}s] Grasp hold complete - LIFTING")
                    elif milestone == "drop":
                        print(f"  [t={t_traj:.2f}s] Reached DROP pose above square bin")
                    elif milestone == "release":
                        print(f"  [t={t_traj:.2f}s] OPENING GRIPPER - releasing object")
                
        elif self._state == PlannerState.COMPLETE:
            # Hold at final position
            if self._grasp_end_position is not None:
                q_command = self._grasp_end_position
            elif self._rrbt2_end_position is not None:
                q_command = self._rrbt2_end_position
            else:
                q_command = self._q_home
        
        # Set output and update gripper triad visualization
        output.SetFromVector(q_command)
        self._update_gripper_triad(q_command)
    
    def CalcGripperCommand(self, context, output):
        """
        Calculate the gripper position command.
        
        The gripper stays open (0.1m) during motion and closes (0.0m) 
        during the grasp hold phase.
        """
        output.SetFromVector([self._gripper_command])
    
    def _update_gripper_triad(self, q_iiwa):
        """
        Update the Meshcat gripper triad visualization using forward kinematics.
        
        Args:
            q_iiwa: 7-element array of iiwa joint positions
        """
        # Set iiwa positions in the FK context
        iiwa_model = self._plant.GetModelInstanceByName("iiwa")
        self._plant.SetPositions(self._fk_plant_context, iiwa_model, q_iiwa)
        
        # Get gripper pose via forward kinematics
        X_WG = self._plant.EvalBodyPoseInWorld(self._fk_plant_context, self._gripper_body)
        
        # Update Meshcat triad
        self._meshcat.SetTransform("gripper_frame", X_WG)
    
    def _compute_ik_targets(self):
        """
        Compute q_goal, q_bin_light_hint, and q_mustard_position_light_hint via IK.
        
        Uses the tf_goal, bin_light_center, and mustard_position_light_center from
        config to solve IK for the corresponding joint configurations.
        """
        # Compute q_goal from tf_goal
        tf_goal = self._config.simulation.tf_goal
        X_WG_goal = RigidTransform(
            RollPitchYaw(tf_goal.rpy).ToRotationMatrix(),
            tf_goal.translation
        )
        
        print(f"Computing q_goal from tf_goal:")
        print(f"  translation: {tf_goal.translation}")
        print(f"  rpy: {tf_goal.rpy}")
        
        try:
            self._q_goal = np.array(solve_ik_for_pose(
                plant=self._plant,
                X_WG_target=X_WG_goal,
                q_nominal=tuple(self._q_home),
                theta_bound=0.1,  # Relaxed orientation tolerance
                pos_tol=0.01,     # 1cm position tolerance
            ))
            print(f"  q_goal computed: {self._q_goal}")
        except RuntimeError as e:
            print(f"  IK failed for tf_goal: {e}")
            raise RuntimeError("Cannot compute q_goal from tf_goal. Check that the goal pose is reachable.")

        # # Visualize goal triad
        # AddMeshcatTriad(self._meshcat, "ik_targets/goal", length=0.15, radius=0.005)
        # self._meshcat.SetTransform("ik_targets/goal", X_WG_goal)
        # print(f"  Added triad at goal: {tf_goal.translation}")

        # Compute q_bin_light_hint from bin_light_center
        bin_light_center = self._config.simulation.bin_light_center
        bin_target_rotation = RotationMatrix.MakeXRotation(np.pi) @ RotationMatrix.MakeZRotation(np.pi)
        X_WG_bin_light = RigidTransform(bin_target_rotation, bin_light_center)

        print(f"Computing q_bin_light_hint from bin_light_center {bin_light_center}...")
        try:
            self._q_bin_light_hint = np.array(solve_ik_for_pose(
                plant=self._plant,
                X_WG_target=X_WG_bin_light,
                q_nominal=tuple(self._q_home),
                theta_bound=0.1,  # Relaxed orientation tolerance
                pos_tol=0.05,     # 5cm position tolerance
            ))
            print(f"  q_bin_light_hint computed: {self._q_bin_light_hint}")
        except RuntimeError as e:
            print(f"  IK failed for bin_light_center, using q_home as fallback: {e}")
            raise RuntimeError("Cannot compute q_bin_light_hint from bin_light_center. Check that the bin light region is reachable.")

        # # Visualize bin light hint triad
        # AddMeshcatTriad(self._meshcat, "ik_targets/bin_light", length=0.15, radius=0.005)
        # self._meshcat.SetTransform("ik_targets/bin_light", X_WG_bin_light)
        # print(f"  Added triad at bin_light_center: {bin_light_center}")

        # Compute q_mustard_position_light_hint from mustard_position_light_center
        mustard_position_light_center = self._config.simulation.mustard_position_light_center
        mustard_target_rotation = RotationMatrix.MakeXRotation(np.pi) @ RotationMatrix.MakeZRotation(np.pi/2)
        X_WG_mustard_light = RigidTransform(mustard_target_rotation, mustard_position_light_center)
        
        print(f"Computing q_mustard_position_light_hint from mustard_position_light_center {mustard_position_light_center}...")
        try:
            self._q_mustard_position_light_hint = np.array(solve_ik_for_pose(
                plant=self._plant,
                X_WG_target=X_WG_mustard_light,
                q_nominal=tuple(self._q_home),
                theta_bound=0.1,  # Relaxed orientation tolerance
                pos_tol=0.05,     # 5cm position tolerance
            ))
            print(f"  q_mustard_position_light_hint computed: {self._q_mustard_position_light_hint}")
        except RuntimeError as e:
            print(f"  IK failed for mustard_position_light_center, using q_home as fallback: {e}")
            # Use q_home as fallback instead of raising
            self._q_mustard_position_light_hint = self._q_home.copy()

        # # Visualize mustard position light hint triad
        # AddMeshcatTriad(self._meshcat, "ik_targets/mustard_position_light", length=0.15, radius=0.005)
        # self._meshcat.SetTransform("ik_targets/mustard_position_light", X_WG_mustard_light)
        # print(f"  Added triad at mustard_position_light_center: {mustard_position_light_center}")
    
    def _run_rrbt_planning(self):
        """
        Execute RRBT planning for bin belief information gathering.
        
        Creates an IiwaProblemBinBelief and runs RRBT planning.
        Populates _rrbt_trajectory and _pred_q_goal.
        """
        print(f"Creating RRBT problem (bin belief)...")
        print(f"  q_start: {self._q_home}")
        print(f"  q_goal: {self._q_goal}")
        print(f"  true_bin: {self._true_bin}")
        
        problem = IiwaProblemBinBelief(
            q_start=tuple(self._q_home),
            q_goal=tuple(self._q_goal),
            gripper_setpoint=0.1,
            meshcat=self._meshcat,
            light_center=self._config.simulation.bin_light_center,
            light_size=self._config.simulation.bin_light_size,
            tpr_light=float(self._config.physics.tpr_light),
            fpr_light=float(self._config.physics.fpr_light),
            n_bins=2,
            true_bin=self._true_bin,
            max_bin_uncertainty=float(self._config.planner.max_bin_uncertainty),
            lambda_weight=float(self._config.planner.bin_lambda_weight),
        )
        
        print(f"Running RRBT planning (max_iterations={self._config.planner.max_iterations})...")
        rrbt_result, iterations = rrbt_planning(
            problem,
            max_iterations=int(self._config.planner.max_iterations),
            bias_prob_sample_q_goal=float(self._config.planner.bias_prob_sample_q_goal),
            bias_prob_sample_q_bin_light=float(self._config.planner.bias_prob_sample_q_bin_light),
            q_light_hint=self._q_bin_light_hint,
            visualize_callback=None,
            visualize_interval=1000,
            verbose=False,
        )
        
        if rrbt_result:
            path_to_info, self._pred_q_goal = rrbt_result
            self._pred_q_goal = np.array(self._pred_q_goal)
            self._rrbt_trajectory = path_to_trajectory(path_to_info)
            print(f"RRBT success:")
            print(f"  Path length: {len(path_to_info)} waypoints")
            print(f"  Trajectory duration: {self._rrbt_trajectory.end_time():.2f}s")
            print(f"  Predicted goal: {np.round(self._pred_q_goal, 3)}")
        else:
            # RRBT failed - create minimal trajectory to stay at home
            print("RRBT FAILED - falling back to home position")
            self._rrbt_trajectory = path_to_trajectory([tuple(self._q_home)])
            self._pred_q_goal = self._q_goal  # Fallback to original goal
    
    def _run_rrbt2_planning(self):
        """
        Execute RRBT2 planning for mustard position uncertainty reduction.
        
        Creates an IiwaProblemMustardPositionBelief and runs RRBT planning.
        The goal is to reduce 3D position uncertainty by moving to the
        mustard position light region.
        
        Populates _rrbt2_trajectory.
        """
        print(f"Creating RRBT2 problem (position belief)...")
        print(f"  q_start: {np.round(self._rrbt_end_position, 3)}")
        print(f"  estimated_mustard_position: {np.round(self._estimated_mustard_position, 3)}")
        
        # Get mustard position RRBT parameters from config
        initial_uncertainty = float(self._config.planner.mustard_position_initial_uncertainty)
        max_uncertainty = float(self._config.planner.mustard_position_max_uncertainty)
        lambda_weight = float(self._config.planner.mustard_position_lambda_weight)
        max_iterations = int(self._config.planner.mustard_position_max_iterations)
        bias_prob = float(self._config.planner.bias_prob_sample_q_mustard_position_light)
        
        problem = IiwaProblemMustardPositionBelief(
            q_start=tuple(self._rrbt_end_position),
            q_goal=tuple(self._q_goal),  # Not used for termination, but required
            gripper_setpoint=0.1,
            meshcat=self._meshcat,
            light_center=self._config.simulation.mustard_position_light_center,
            light_size=self._config.simulation.mustard_position_light_size,
            scale_R_light=float(self._config.physics.meas_noise_light) ** 2,
            scale_R_dark=float(self._config.physics.meas_noise_dark) ** 2,
            initial_uncertainty=initial_uncertainty,
            max_uncertainty=max_uncertainty,
            lambda_weight=lambda_weight,
            estimated_position=self._estimated_mustard_position,
        )
        
        print(f"Running RRBT2 planning (max_iterations={max_iterations})...")
        print(f"  initial_uncertainty: {initial_uncertainty}")
        print(f"  max_uncertainty: {max_uncertainty}")
        print(f"  lambda_weight: {lambda_weight}")
        
        rrbt2_result, iterations = rrbt_planning(
            problem,
            max_iterations=max_iterations,
            bias_prob_sample_q_goal=0.05,  # Low bias to goal (we don't care about it)
            bias_prob_sample_q_bin_light=bias_prob,  # Bias to mustard position light region
            q_light_hint=self._q_mustard_position_light_hint,
            visualize_callback=None,
            visualize_interval=1000,
            verbose=False,
        )
        
        if rrbt2_result:
            path_to_position, _ = rrbt2_result
            self._rrbt2_trajectory = path_to_trajectory(path_to_position)
            print(f"RRBT2 success:")
            print(f"  Path length: {len(path_to_position)} waypoints")
            print(f"  Trajectory duration: {self._rrbt2_trajectory.end_time():.2f}s")
        else:
            # RRBT2 failed - create minimal trajectory to hold position
            print("RRBT2 FAILED - holding at RRBT end position")
            self._rrbt2_trajectory = path_to_trajectory([tuple(self._rrbt_end_position)])

    def _run_grasp_planning(self, context):
        """
        Execute grasp planning after RRBT2 trajectory completion.
        
        This method:
        1. Reads the final covariance from the Kalman filter
        2. Samples a position from the covariance ellipsoid
        3. Combines sampled position with ICP rotation to create candidate pose
        4. Transforms mustard model to world frame at the sampled pose
        5. Runs antipodal grasp selection
        6. Computes pre-grasp pose
        
        Populates _best_grasp_pose and _pregrasp_pose.
        """
        print(f"Starting grasp planning...")
        
        # 1. Read covariance from input port (2x2 flattened to 4, X-Y only)
        covariance_flat = self._covariance_input_port.Eval(context)
        covariance_2x2 = covariance_flat.reshape(2, 2)
        
        print(f"  Final covariance trace (X-Y): {np.trace(covariance_2x2):.6f}")
        print(f"  Covariance diagonal (X-Y): {np.diag(covariance_2x2)}")
        
        # 2. Sample position from truncated 2D Gaussian (X-Y only, Z fixed)
        # Using max_sigma=0.5 for conservative sampling (very close to mean)
        # Note: This is conservative because the Kalman filter covariance reduction
        # doesn't seem to be working correctly during RRBT2 execution
        sampled_position = sample_position_from_covariance(
            mean=self._estimated_mustard_position,
            covariance=covariance_2x2,
            rng=self._rng,
            max_sigma=0.2,  # Conservative: stay close to ICP estimate
        )
        
        # Compute offset from mean for logging (X-Y only since Z is fixed)
        offset_xy = sampled_position[:2] - self._estimated_mustard_position[:2]
        offset_norm = np.linalg.norm(offset_xy)
        
        print(f"  ICP estimated position: {self._estimated_mustard_position}")
        print(f"  Sampled position: {sampled_position} (Z fixed from ICP)")
        print(f"  X-Y offset from mean: {offset_xy} (norm: {offset_norm:.4f}m)")
        
        # 3. Create candidate pose: sampled position + ICP rotation
        X_WM_sampled = RigidTransform(
            self._estimated_mustard_pose.rotation(),
            sampled_position
        )
        
        # # Visualize the sampled pose
        # AddMeshcatTriad(self._meshcat, "grasp_planning/sampled_pose", length=0.1, radius=0.003)
        # self._meshcat.SetTransform("grasp_planning/sampled_pose", X_WM_sampled)
        
        # 4. Load mustard model and transform to world frame
        print(f"  Loading mustard model...")
        mustard_model = MustardPointCloud()
        model_pcl = mustard_model.xyzs()  # (3, N) in model frame
        
        # Transform model to world frame at sampled pose
        model_world_xyz = X_WM_sampled @ model_pcl
        
        print(f"  Model point cloud: {model_pcl.shape[1]} points")
        
        # 5. Create Drake PointCloud with normals for grasp selection
        grasp_cloud = PointCloud(
            model_world_xyz.shape[1],
            Fields(BaseField.kXYZs | BaseField.kNormals)
        )
        grasp_cloud.mutable_xyzs()[:] = model_world_xyz
        
        # Estimate normals on the model point cloud
        grasp_cloud.EstimateNormals(radius=0.05, num_closest=30)
        
        # Flip normals outward (away from centroid)
        centroid = np.mean(model_world_xyz, axis=1)
        grasp_cloud.FlipNormalsTowardPoint(centroid + np.array([0, 0, 1]))
        
        print(f"  Model cloud in world frame: {grasp_cloud.size()} points with normals")
        
        # Visualize the grasp cloud
        self._meshcat.SetObject(
            "grasp_planning/grasp_cloud",
            grasp_cloud,
            point_size=0.003,
            rgba=Rgba(0, 1, 1, 1)  # Cyan
        )
        
        # 6. Get grasp candidates (will be validated by IK in _compute_grasp_trajectory)
        print(f"  Running grasp selection...")
        self._grasp_candidates = select_best_grasp(
            meshcat=self._meshcat,
            cloud=grasp_cloud,
            rng=self._rng,
            num_candidates=1000,
            num_to_draw=0,
            num_to_return=20,  # Return top 20 for IK validation
            debug=False,
        )
        
        if self._grasp_candidates:
            print(f"\n  Found {len(self._grasp_candidates)} grasp candidates for IK validation")
            # Best grasp will be selected during IK validation in _compute_grasp_trajectory
            self._best_grasp_pose = None  # Will be set after IK validation
            self._pregrasp_pose = None
        else:
            print(f"\n  No valid grasp candidates found!")
            self._grasp_candidates = []
            self._best_grasp_pose = None
            self._pregrasp_pose = None

    def _compute_grasp_trajectory(self):
        """
        Compute the grasp execution trajectory.
        
        Iterates through grasp candidates and validates each with IK.
        Uses the first candidate that passes all IK checks.
        
        Creates a trajectory through waypoints:
        1. current position (end of RRBT2)
        2. home position (safe intermediate waypoint if needed)
        3. pregrasp pose (30cm above grasp)
        4. grasp pose (at object)
        5. grasp hold (same pose, gripper closes)
        6. lift pose (grasp + 30cm in Z)
        7. drop pose (above square bin)
        
        Timing is adjusted based on joint space distances.
        
        Sets:
        - self._grasp_trajectory: PiecewisePolynomial trajectory
        - self._grasp_close_time: time offset when gripper should close
        """
        print(f"\nComputing grasp execution trajectory...")
        
        # Current position (end of RRBT2)
        q_current = self._rrbt2_end_position
        print(f"  Current position: {np.round(q_current, 3)}")
        
        if not self._grasp_candidates:
            print(f"  No grasp candidates to validate!")
            return False
        
        # Iterate through grasp candidates and find first one that passes all IK checks
        print(f"\n  Validating {len(self._grasp_candidates)} grasp candidates with IK...")
        
        q_pregrasp = None
        q_grasp = None
        q_lift = None
        q_drop = None
        lift_pos = None
        drop_pos = None
        
        for candidate_idx, (cost, grasp_pose) in enumerate(self._grasp_candidates):
            print(f"\n  Candidate {candidate_idx + 1}/{len(self._grasp_candidates)} (cost={cost:.3f}):")
            
            # Compute pregrasp pose for this candidate
            pregrasp_pose = compute_pregrasp_pose(grasp_pose, offset_z=0.3)
            
            # Try IK for pregrasp
            try:
                q_pregrasp = np.array(solve_ik_for_pose(
                    plant=self._plant,
                    X_WG_target=pregrasp_pose,
                    q_nominal=tuple(self._q_home),
                    theta_bound=0.05,
                    pos_tol=0.01,
                    q_initial=tuple(q_current),
                ))
                print(f"    Pregrasp IK: OK")
            except RuntimeError:
                print(f"    Pregrasp IK: FAILED")
                continue
            
            # Try IK for grasp
            try:
                q_grasp = np.array(solve_ik_for_pose(
                    plant=self._plant,
                    X_WG_target=grasp_pose,
                    q_nominal=tuple(q_pregrasp),
                    theta_bound=0.01,
                    pos_tol=0.00005,
                ))
                print(f"    Grasp IK: OK")
            except RuntimeError:
                print(f"    Grasp IK: FAILED")
                continue
            
            # Compute lift pose (maintain grasp orientation)
            lift_pos = grasp_pose.translation() + np.array([0, 0, 0.3])
            X_lift = RigidTransform(grasp_pose.rotation(), lift_pos)
            
            # Try IK for lift
            try:
                q_lift = np.array(solve_ik_for_pose(
                    plant=self._plant,
                    X_WG_target=X_lift,
                    q_nominal=tuple(q_grasp),
                    theta_bound=0.05,
                    pos_tol=0.005,
                ))
                print(f"    Lift IK: OK")
            except RuntimeError:
                print(f"    Lift IK: FAILED")
                continue
            
            # Compute drop pose (flat/flush - gripper x-z plane parallel to ground)
            drop_pos = np.array([0.5, -0.5, lift_pos[2]])
            
            # Make gripper flat: y-axis points down, x-z plane is horizontal
            # Get grasp x-axis and project to horizontal plane
            grasp_x_world = grasp_pose.rotation().matrix()[:, 0]  # Gripper x-axis in world frame
            grasp_x_horizontal = np.array([grasp_x_world[0], grasp_x_world[1], 0.0])
            grasp_x_horizontal_norm = np.linalg.norm(grasp_x_horizontal)
            if grasp_x_horizontal_norm > 1e-6:
                gripper_x = grasp_x_horizontal / grasp_x_horizontal_norm
            else:
                # Fallback: use world x-axis if grasp x is vertical
                gripper_x = np.array([1.0, 0.0, 0.0])
            
            # Gripper y-axis points down (world -z)
            gripper_y = np.array([0.0, 0.0, -1.0])
            
            # Gripper z-axis is cross product (ensures right-handed frame)
            gripper_z = np.cross(gripper_x, gripper_y)
            gripper_z = gripper_z / np.linalg.norm(gripper_z)
            
            # Recompute x to ensure orthonormality
            gripper_x = np.cross(gripper_y, gripper_z)
            gripper_x = gripper_x / np.linalg.norm(gripper_x)
            
            # Construct rotation matrix with flat orientation
            R_drop_flat = RotationMatrix(np.column_stack([gripper_x, gripper_y, gripper_z]))
            X_drop = RigidTransform(R_drop_flat, drop_pos)
            
            # Try IK for drop
            try:
                q_drop = np.array(solve_ik_for_pose(
                    plant=self._plant,
                    X_WG_target=X_drop,
                    q_nominal=tuple(q_lift),
                    theta_bound=0.05,
                    pos_tol=0.01,
                    q_initial=tuple(q_lift),
                ))
                print(f"    Drop IK: OK")
            except RuntimeError:
                print(f"    Drop IK: FAILED")
                continue
            
            # All IK checks passed! Use this grasp
            print(f"\n  SUCCESS! Using grasp candidate {candidate_idx + 1} (cost={cost:.3f})")
            
            # Store the validated grasp and pregrasp poses
            self._best_grasp_pose = grasp_pose
            self._pregrasp_pose = pregrasp_pose
            self._drop_pose = X_drop
            
            # Visualize the selected grasp
            rpy_grasp = RollPitchYaw(grasp_pose.rotation())
            print(f"    Position: {grasp_pose.translation()}")
            print(f"    RPY: [{rpy_grasp.roll_angle():.3f}, {rpy_grasp.pitch_angle():.3f}, {rpy_grasp.yaw_angle():.3f}]")
            
            draw_grasp_candidate(self._meshcat, grasp_pose, prefix="grasp_planning/best_grasp")
            AddMeshcatTriad(self._meshcat, "grasp_planning/grasp_pose", length=0.15, radius=0.005)
            self._meshcat.SetTransform("grasp_planning/grasp_pose", grasp_pose)
            
            AddMeshcatTriad(self._meshcat, "grasp_planning/pregrasp_pose", length=0.15, radius=0.005)
            self._meshcat.SetTransform("grasp_planning/pregrasp_pose", pregrasp_pose)
            draw_grasp_candidate(self._meshcat, pregrasp_pose, prefix="grasp_planning/pregrasp_gripper")
            
            break
        else:
            # No candidate passed all IK checks
            print(f"\n  All {len(self._grasp_candidates)} grasp candidates failed IK validation!")
            return False
        
        # Add intermediate waypoints for straight-up lift motion
        # This ensures the arm moves straight up in Cartesian space, not a curved path
        print(f"\n  Computing intermediate waypoints for straight-up lift...")
        grasp_pos = self._best_grasp_pose.translation()
        lift_intermediate_waypoints = []
        lift_intermediate_qs = []
        
        # Create waypoints at 0.1m, 0.2m, and 0.3m above grasp (same X, Y, orientation)
        for z_offset in [0.1, 0.2, 0.3]:
            intermediate_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + z_offset])
            X_intermediate = RigidTransform(self._best_grasp_pose.rotation(), intermediate_pos)
            
            try:
                # Use previous waypoint as seed (or grasp if first)
                q_seed = lift_intermediate_qs[-1] if lift_intermediate_qs else q_grasp
                q_intermediate = np.array(solve_ik_for_pose(
                    plant=self._plant,
                    X_WG_target=X_intermediate,
                    q_nominal=tuple(q_seed),
                    theta_bound=0.05,
                    pos_tol=0.01,
                    q_initial=tuple(q_seed),
                ))
                lift_intermediate_waypoints.append(X_intermediate)
                lift_intermediate_qs.append(q_intermediate)
                print(f"    Intermediate waypoint at +{z_offset:.1f}m: OK")
            except RuntimeError as e:
                print(f"    Intermediate waypoint at +{z_offset:.1f}m: FAILED ({e})")
                # If intermediate fails, fall back to direct grasp->lift
                lift_intermediate_waypoints = []
                lift_intermediate_qs = []
                break
        
        # If we successfully computed intermediate waypoints, use them
        # Otherwise, fall back to direct grasp->lift
        use_intermediate_lift = len(lift_intermediate_qs) > 0
        if use_intermediate_lift:
            print(f"  Using {len(lift_intermediate_qs)} intermediate waypoints for straight-up lift")
        else:
            print(f"  Falling back to direct grasp->lift (no intermediate waypoints)")
        
        # Add intermediate waypoints for horizontal transfer (lift → drop)
        # This ensures the arm moves smoothly horizontally, not in a curved/waving path
        print(f"\n  Computing intermediate waypoints for horizontal transfer...")
        transfer_intermediate_qs = []
        
        # Compute intermediate positions (interpolate XY, keep Z constant)
        lift_xy = lift_pos[:2]  # [x, y] at lift
        drop_xy = drop_pos[:2]  # [x, y] at drop
        transfer_z = lift_pos[2]  # Constant Z height
        
        # Get orientations for interpolation
        R_lift = self._best_grasp_pose.rotation()  # Grasp orientation at lift
        R_drop = self._drop_pose.rotation()  # Flat orientation at drop
        
        # Use 3 intermediate waypoints at 25%, 50%, 75% of the way
        for i, alpha in enumerate([0.25, 0.5, 0.75]):
            # Interpolate position linearly
            intermediate_xy = lift_xy + alpha * (drop_xy - lift_xy)
            intermediate_pos = np.array([intermediate_xy[0], intermediate_xy[1], transfer_z])
            
            # Keep grasp orientation for first two waypoints (25%, 50%), switch to flat at 75%
            # This ensures smooth motion and only rotates near the end
            if alpha < 0.75:
                R_intermediate = R_lift
            else:
                R_intermediate = R_drop
            
            X_intermediate = RigidTransform(R_intermediate, intermediate_pos)
            
            try:
                # Use previous waypoint as seed (or q_lift if first)
                q_seed = transfer_intermediate_qs[-1] if transfer_intermediate_qs else q_lift
                q_intermediate = np.array(solve_ik_for_pose(
                    plant=self._plant,
                    X_WG_target=X_intermediate,
                    q_nominal=tuple(q_seed),
                    theta_bound=0.05,
                    pos_tol=0.01,
                    q_initial=tuple(q_seed),
                ))
                transfer_intermediate_qs.append(q_intermediate)
                print(f"    Transfer waypoint at {int(alpha*100)}%: OK")
            except RuntimeError as e:
                print(f"    Transfer waypoint at {int(alpha*100)}%: FAILED ({e})")
                # Fall back to direct lift→drop
                transfer_intermediate_qs = []
                break
        
        # Check if transfer waypoints were successful
        use_intermediate_transfer = len(transfer_intermediate_qs) > 0
        if use_intermediate_transfer:
            print(f"  Using {len(transfer_intermediate_qs)} intermediate waypoints for horizontal transfer")
        else:
            print(f"  Falling back to direct lift->drop (no intermediate waypoints)")
        
        # Calculate joint space distances
        dist_current_to_home = np.linalg.norm(self._q_home - q_current)
        dist_home_to_pregrasp = np.linalg.norm(q_pregrasp - self._q_home)
        dist_current_to_pregrasp = np.linalg.norm(q_pregrasp - q_current)
        dist_pregrasp_to_grasp = np.linalg.norm(q_grasp - q_pregrasp)
        
        # Compute distances for lift segments
        if use_intermediate_lift:
            dist_grasp_to_lift_0_1 = np.linalg.norm(lift_intermediate_qs[0] - q_grasp)
            dist_lift_0_1_to_0_2 = np.linalg.norm(lift_intermediate_qs[1] - lift_intermediate_qs[0])
            dist_lift_0_2_to_lift = np.linalg.norm(q_lift - lift_intermediate_qs[1])
            dist_grasp_to_lift = dist_grasp_to_lift_0_1 + dist_lift_0_1_to_0_2 + dist_lift_0_2_to_lift
        else:
            dist_grasp_to_lift = np.linalg.norm(q_lift - q_grasp)
        
        # Compute distances for transfer segments (lift → drop)
        if use_intermediate_transfer:
            dist_lift_to_transfer_25 = np.linalg.norm(transfer_intermediate_qs[0] - q_lift)
            dist_transfer_25_to_50 = np.linalg.norm(transfer_intermediate_qs[1] - transfer_intermediate_qs[0])
            dist_transfer_50_to_75 = np.linalg.norm(transfer_intermediate_qs[2] - transfer_intermediate_qs[1])
            dist_transfer_75_to_drop = np.linalg.norm(q_drop - transfer_intermediate_qs[2])
            dist_lift_to_drop = dist_lift_to_transfer_25 + dist_transfer_25_to_50 + dist_transfer_50_to_75 + dist_transfer_75_to_drop
        else:
            dist_lift_to_drop = np.linalg.norm(q_drop - q_lift)
        
        print(f"\n  Joint space distances:")
        print(f"    current → pregrasp (direct): {dist_current_to_pregrasp:.4f} rad")
        print(f"    current → home:              {dist_current_to_home:.4f} rad")
        print(f"    home → pregrasp:             {dist_home_to_pregrasp:.4f} rad")
        print(f"    pregrasp → grasp:            {dist_pregrasp_to_grasp:.4f} rad")
        print(f"    grasp → lift:                {dist_grasp_to_lift:.4f} rad")
        print(f"    lift → drop:                 {dist_lift_to_drop:.4f} rad")
        
        # Decide whether to use home as intermediate waypoint
        # If direct path is much longer than going through home, use home
        use_home_waypoint = dist_current_to_pregrasp > (dist_current_to_home + dist_home_to_pregrasp) * 0.8
        
        # Time scaling: ~1 second per radian of joint motion (smooth motion)
        time_per_rad = 1.0
        
        if use_home_waypoint:
            print(f"\n  Using HOME as intermediate waypoint for safer motion")
            
            t_home = dist_current_to_home * time_per_rad
            t_pregrasp = t_home + dist_home_to_pregrasp * time_per_rad
            t_grasp_start = t_pregrasp + dist_pregrasp_to_grasp * time_per_rad
            t_grasp_end = t_grasp_start + 1.0  # 1 second hold for gripper close
            
            # Build trajectory with optional intermediate lift and transfer waypoints
            if use_intermediate_lift:
                t_lift_0_1 = t_grasp_end + dist_grasp_to_lift_0_1 * time_per_rad
                t_lift_0_2 = t_lift_0_1 + dist_lift_0_1_to_0_2 * time_per_rad
                t_lift = t_lift_0_2 + dist_lift_0_2_to_lift * time_per_rad
            else:
                t_lift = t_grasp_end + dist_grasp_to_lift * time_per_rad
            
            if use_intermediate_transfer:
                t_transfer_25 = t_lift + dist_lift_to_transfer_25 * time_per_rad
                t_transfer_50 = t_transfer_25 + dist_transfer_25_to_50 * time_per_rad
                t_transfer_75 = t_transfer_50 + dist_transfer_50_to_75 * time_per_rad
                t_drop = t_transfer_75 + dist_transfer_75_to_drop * time_per_rad
            else:
                t_drop = t_lift + dist_lift_to_drop * time_per_rad
            
            t_release = t_drop + 0.5  # 0.5 second hold at drop before release
            t_end = t_release + 0.5   # 0.5 second after release
            
            # Build times and waypoints arrays based on which intermediate waypoints are used
            times_list = [0.0, t_home, t_pregrasp, t_grasp_start, t_grasp_end]
            waypoints_list = [q_current, self._q_home, q_pregrasp, q_grasp, q_grasp]
            
            if use_intermediate_lift:
                times_list.extend([t_lift_0_1, t_lift_0_2, t_lift])
                waypoints_list.extend([lift_intermediate_qs[0], lift_intermediate_qs[1], q_lift])
            else:
                times_list.append(t_lift)
                waypoints_list.append(q_lift)
            
            if use_intermediate_transfer:
                times_list.extend([t_transfer_25, t_transfer_50, t_transfer_75, t_drop])
                waypoints_list.extend([transfer_intermediate_qs[0], transfer_intermediate_qs[1], transfer_intermediate_qs[2], q_drop])
            else:
                times_list.append(t_drop)
                waypoints_list.append(q_drop)
            
            times_list.extend([t_release, t_end])
            waypoints_list.extend([q_drop, q_drop])  # Hold at drop, stay after release
            
            times = np.array(times_list)
            waypoints = np.column_stack(waypoints_list)
        else:
            print(f"\n  Using DIRECT path to pregrasp")
            
            t_pregrasp = dist_current_to_pregrasp * time_per_rad
            t_grasp_start = t_pregrasp + dist_pregrasp_to_grasp * time_per_rad
            t_grasp_end = t_grasp_start + 1.0  # 1 second hold for gripper close
            
            # Build trajectory with optional intermediate lift and transfer waypoints
            if use_intermediate_lift:
                t_lift_0_1 = t_grasp_end + dist_grasp_to_lift_0_1 * time_per_rad
                t_lift_0_2 = t_lift_0_1 + dist_lift_0_1_to_0_2 * time_per_rad
                t_lift = t_lift_0_2 + dist_lift_0_2_to_lift * time_per_rad
            else:
                t_lift = t_grasp_end + dist_grasp_to_lift * time_per_rad
            
            if use_intermediate_transfer:
                t_transfer_25 = t_lift + dist_lift_to_transfer_25 * time_per_rad
                t_transfer_50 = t_transfer_25 + dist_transfer_25_to_50 * time_per_rad
                t_transfer_75 = t_transfer_50 + dist_transfer_50_to_75 * time_per_rad
                t_drop = t_transfer_75 + dist_transfer_75_to_drop * time_per_rad
            else:
                t_drop = t_lift + dist_lift_to_drop * time_per_rad
            
            t_release = t_drop + 0.5  # 0.5 second hold at drop before release
            t_end = t_release + 0.5   # 0.5 second after release
            
            # Build times and waypoints arrays based on which intermediate waypoints are used
            times_list = [0.0, t_pregrasp, t_grasp_start, t_grasp_end]
            waypoints_list = [q_current, q_pregrasp, q_grasp, q_grasp]
            
            if use_intermediate_lift:
                times_list.extend([t_lift_0_1, t_lift_0_2, t_lift])
                waypoints_list.extend([lift_intermediate_qs[0], lift_intermediate_qs[1], q_lift])
            else:
                times_list.append(t_lift)
                waypoints_list.append(q_lift)
            
            if use_intermediate_transfer:
                times_list.extend([t_transfer_25, t_transfer_50, t_transfer_75, t_drop])
                waypoints_list.extend([transfer_intermediate_qs[0], transfer_intermediate_qs[1], transfer_intermediate_qs[2], q_drop])
            else:
                times_list.append(t_drop)
                waypoints_list.append(q_drop)
            
            times_list.extend([t_release, t_end])
            waypoints_list.extend([q_drop, q_drop])  # Hold at drop, stay after release
            
            times = np.array(times_list)
            waypoints = np.column_stack(waypoints_list)
        
        # Create trajectory using first-order hold (linear interpolation)
        self._grasp_trajectory = PiecewisePolynomial.FirstOrderHold(times, waypoints)
        
        # Set gripper close time (gripper closes at grasp_start)
        self._grasp_close_time = t_grasp_start
        
        # Set gripper open time (gripper opens at release)
        self._grasp_open_time = t_release
        
        # Store final position
        self._grasp_end_position = q_drop
        
        print(f"\n  Grasp trajectory created:")
        print(f"    Total duration: {t_end:.1f}s")
        print(f"    Gripper closes at: {t_grasp_start:.1f}s")
        print(f"    Gripper opens at: {t_release:.1f}s")
        
        # Build waypoint description string
        waypoint_str = "current"
        if use_home_waypoint:
            waypoint_str += " → HOME"
        waypoint_str += " → pregrasp → grasp → hold"
        if use_intermediate_lift:
            waypoint_str += " → lift(+0.1m) → lift(+0.2m) → lift(+0.3m)"
        else:
            waypoint_str += " → lift"
        if use_intermediate_transfer:
            waypoint_str += " → transfer(25%) → transfer(50%) → transfer(75%)"
        waypoint_str += " → drop → hold → release"
        print(f"    Waypoints: {waypoint_str}")
        
        # Verify cartesian positions
        print(f"\n  Expected cartesian positions:")
        print(f"    pregrasp: {np.round(self._pregrasp_pose.translation(), 4)}")
        print(f"    grasp:    {np.round(self._best_grasp_pose.translation(), 4)}")
        print(f"    lift:     {np.round(lift_pos, 4)}")
        print(f"    drop:     {np.round(drop_pos, 4)}")
        print(f"    Z difference (pregrasp - grasp): {self._pregrasp_pose.translation()[2] - self._best_grasp_pose.translation()[2]:.3f}m")
        
        # Store milestone times for logging during execution
        self._grasp_t_pregrasp = t_pregrasp
        self._grasp_t_grasp = t_grasp_start
        self._grasp_t_lift_start = t_grasp_end
        self._grasp_t_drop = t_drop
        self._grasp_t_release = t_release
        
        return True
