"""
Planner System - A Drake LeafSystem for sequential RRBT-RRT planning.

This system implements a state machine that:
1. Initially outputs q_home while waiting for configuration
2. Runs RRBT planning (blocking) for information gathering
3. Executes the RRBT trajectory
4. Runs RRT planning (blocking) from RRBT endpoint to goal
5. Executes the RRT trajectory
6. Holds at the goal position when complete
"""

from enum import Enum, auto
import numpy as np
from pydrake.all import (
    LeafSystem,
    BasicVector,
    PiecewisePolynomial,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
)

from src.simulation.simulation_tools import IiwaProblem, IiwaProblemBinBelief
from src.planning.belief_space_rrt import rrbt_planning
from src.planning.standard_rrt import rrt_planning
from src.utils.ik_solver import solve_ik_for_pose


class PlannerState(Enum):
    """State machine states for the planner."""
    IDLE = auto()              # Waiting for setup
    RRBT_PLANNING = auto()     # Running RRBT (blocking)
    RRBT_EXECUTING = auto()    # Playing RRBT trajectory
    RRT_PLANNING = auto()      # Running RRT (blocking)
    RRT_EXECUTING = auto()     # Playing RRT trajectory
    COMPLETE = auto()          # Done, holding at goal


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
    A Drake LeafSystem that performs sequential RRBT-RRT planning.
    
    State Machine:
        IDLE -> RRBT_PLANNING -> RRBT_EXECUTING -> RRT_PLANNING -> RRT_EXECUTING -> COMPLETE
    
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
        self._q_light_hint = None  # Computed via IK from light_center
        
        # State machine
        self._state = PlannerState.IDLE
        
        # RRBT stage
        self._rrbt_trajectory = None
        self._rrbt_start_time = None
        self._rrbt_end_position = None
        self._pred_q_goal = None  # Goal predicted by RRBT belief
        
        # RRT stage
        self._rrt_trajectory = None
        self._rrt_start_time = None
        
        # Runtime parameters (set by configure_for_execution)
        self._true_bin = None
        self._X_WM_mustard = None
        
        # Compute q_goal and q_light_hint via IK
        self._compute_ik_targets()
        
        # Output port
        self.DeclareVectorOutputPort(
            "iiwa_position_command",
            BasicVector(7),
            self.CalcJointCommand
        )
        
        print(f"PlannerSystem initialized:")
        print(f"  q_home: {self._q_home}")
        print(f"  q_goal: {self._q_goal}")
        print(f"  q_light_hint: {self._q_light_hint}")
    
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
        
        if self._state == PlannerState.IDLE:
            # Waiting for configuration - hold at home
            output.SetFromVector(self._q_home)
            
        elif self._state == PlannerState.RRBT_PLANNING:
            # Run RRBT planning (blocking)
            print(f"\n{'='*60}")
            print("RRBT PLANNING PHASE")
            print(f"{'='*60}")
            self._run_rrbt_planning()
            self._rrbt_start_time = t
            self._state = PlannerState.RRBT_EXECUTING
            print(f"RRBT planning complete. Starting trajectory execution at t={t:.2f}s")
            print(f"  RRBT trajectory duration: {self._rrbt_trajectory.end_time():.2f}s")
            output.SetFromVector(self._q_home)
            
        elif self._state == PlannerState.RRBT_EXECUTING:
            t_traj = t - self._rrbt_start_time
            if t_traj >= self._rrbt_trajectory.end_time():
                # RRBT trajectory complete, transition to RRT planning
                self._rrbt_end_position = self._rrbt_trajectory.value(
                    self._rrbt_trajectory.end_time()
                ).flatten()
                self._state = PlannerState.RRT_PLANNING
                print(f"\nRRBT execution complete at t={t:.2f}s")
                print(f"  End position: {np.round(self._rrbt_end_position, 3)}")
                output.SetFromVector(self._rrbt_end_position)
            else:
                q = self._rrbt_trajectory.value(t_traj).flatten()
                output.SetFromVector(q)
                
        elif self._state == PlannerState.RRT_PLANNING:
            # Run RRT from RRBT endpoint to predicted goal (blocking)
            print(f"\n{'='*60}")
            print("RRT PLANNING PHASE")
            print(f"{'='*60}")
            self._run_rrt_planning()
            self._rrt_start_time = t
            self._state = PlannerState.RRT_EXECUTING
            print(f"RRT planning complete. Starting trajectory execution at t={t:.2f}s")
            print(f"  RRT trajectory duration: {self._rrt_trajectory.end_time():.2f}s")
            output.SetFromVector(self._rrbt_end_position)
            
        elif self._state == PlannerState.RRT_EXECUTING:
            t_traj = t - self._rrt_start_time
            if t_traj >= self._rrt_trajectory.end_time():
                # RRT trajectory complete
                self._state = PlannerState.COMPLETE
                final_pos = self._rrt_trajectory.value(
                    self._rrt_trajectory.end_time()
                ).flatten()
                print(f"\n{'='*60}")
                print("EXECUTION COMPLETE")
                print(f"{'='*60}")
                print(f"  Final position: {np.round(final_pos, 3)}")
                print(f"  Total time: {t:.2f}s")
                output.SetFromVector(final_pos)
            else:
                q = self._rrt_trajectory.value(t_traj).flatten()
                output.SetFromVector(q)
                
        elif self._state == PlannerState.COMPLETE:
            # Hold at final position
            final_pos = self._rrt_trajectory.value(
                self._rrt_trajectory.end_time()
            ).flatten()
            output.SetFromVector(final_pos)
    
    def _compute_ik_targets(self):
        """
        Compute q_goal and q_light_hint from task-space targets via IK.
        
        Uses the tf_goal and light_center from config to solve IK for
        the corresponding joint configurations.
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
        
        # Compute q_light_hint from light_center
        light_center = self._config.simulation.light_center
        target_rotation = RotationMatrix.MakeXRotation(np.pi) @ RotationMatrix.MakeZRotation(np.pi)
        X_WG_light = RigidTransform(target_rotation, light_center)
        
        print(f"Computing q_light_hint from light_center {light_center}...")
        try:
            self._q_light_hint = np.array(solve_ik_for_pose(
                plant=self._plant,
                X_WG_target=X_WG_light,
                q_nominal=tuple(self._q_home),
                theta_bound=0.1,  # Relaxed orientation tolerance
                pos_tol=0.05,     # 5cm position tolerance
            ))
            print(f"  q_light_hint computed: {self._q_light_hint}")
        except RuntimeError as e:
            print(f"  IK failed for light_center, using q_home as fallback: {e}")
            raise RuntimeError("Cannot compute q_light_hint from light_center. Check that the light region is reachable.")
    
    def _run_rrbt_planning(self):
        """
        Execute RRBT planning for information gathering.
        
        Creates an IiwaProblemBinBelief and runs RRBT planning.
        Populates _rrbt_trajectory and _pred_q_goal.
        """
        print(f"Creating RRBT problem...")
        print(f"  q_start: {self._q_home}")
        print(f"  q_goal: {self._q_goal}")
        print(f"  true_bin: {self._true_bin}")
        
        problem = IiwaProblemBinBelief(
            q_start=tuple(self._q_home),
            q_goal=tuple(self._q_goal),
            gripper_setpoint=0.1,
            meshcat=self._meshcat,
            light_center=self._config.simulation.light_center,
            light_size=self._config.simulation.light_size,
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
            q_light_hint=self._q_light_hint,
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
    
    def _run_rrt_planning(self):
        """
        Execute RRT planning from RRBT endpoint to predicted goal.
        
        Creates an IiwaProblem and runs standard RRT planning.
        Populates _rrt_trajectory.
        """
        print(f"Creating RRT problem...")
        print(f"  q_start: {np.round(self._rrbt_end_position, 3)}")
        print(f"  q_goal: {np.round(self._pred_q_goal, 3)}")
        
        problem = IiwaProblem(
            q_start=tuple(self._rrbt_end_position),
            q_goal=tuple(self._pred_q_goal),
            gripper_setpoint=0.1,
            meshcat=self._meshcat,
        )
        
        print(f"Running RRT planning (max_iterations=1000)...")
        path_to_grasp, iterations = rrt_planning(
            problem,
            max_iterations=1000,
            prob_sample_q_goal=0.25,  # Higher bias for goal-directed motion
        )
        
        if path_to_grasp:
            self._rrt_trajectory = path_to_trajectory(path_to_grasp)
            print(f"RRT success:")
            print(f"  Path length: {len(path_to_grasp)} waypoints")
            print(f"  Trajectory duration: {self._rrt_trajectory.end_time():.2f}s")
        else:
            # RRT failed - create trajectory to hold position
            print("RRT FAILED - holding at RRBT end position")
            self._rrt_trajectory = path_to_trajectory([tuple(self._rrbt_end_position)])
