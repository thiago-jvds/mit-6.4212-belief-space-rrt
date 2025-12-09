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
        self._rng = np.random.default_rng()  # Random generator for grasp sampling
        
        # Output port
        self.DeclareVectorOutputPort(
            "iiwa_position_command",
            BasicVector(7),
            self.CalcJointCommand
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
        
        if self._state == PlannerState.IDLE:
            # Waiting for configuration - hold at home
            output.SetFromVector(self._q_home)
            
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
            output.SetFromVector(self._q_home)
            
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
                output.SetFromVector(self._rrbt_end_position)
            else:
                q = self._rrbt_trajectory.value(t_traj).flatten()
                output.SetFromVector(q)
        
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
            output.SetFromVector(self._rrbt_end_position)
                
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
            output.SetFromVector(self._rrbt_end_position)
            
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
                output.SetFromVector(self._rrbt2_end_position)
            else:
                q = self._rrbt2_trajectory.value(t_traj).flatten()
                output.SetFromVector(q)
        
        elif self._state == PlannerState.GRASP_PLANNING:
            # Run grasp planning (blocking)
            print(f"\n{'='*60}")
            print("GRASP PLANNING PHASE")
            print(f"{'='*60}")
            self._run_grasp_planning(context)
            self._state = PlannerState.COMPLETE
            print(f"\n{'='*60}")
            print("EXECUTION COMPLETE")
            print(f"{'='*60}")
            print(f"  Final position: {np.round(self._rrbt2_end_position, 3)}")
            if self._best_grasp_pose is not None:
                print(f"  Best grasp position: {np.round(self._best_grasp_pose.translation(), 3)}")
                print(f"  Pre-grasp position: {np.round(self._pregrasp_pose.translation(), 3)}")
            else:
                print(f"  No valid grasp found")
            print(f"  Total time: {t:.2f}s")
            output.SetFromVector(self._rrbt2_end_position)
                
        elif self._state == PlannerState.COMPLETE:
            # Hold at final position
            output.SetFromVector(self._rrbt2_end_position)
    
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
        # max_sigma=2.0 restricts samples to ~95% probability region (within 2-sigma ellipse)
        sampled_position = sample_position_from_covariance(
            mean=self._estimated_mustard_position,
            covariance=covariance_2x2,
            rng=self._rng,
            max_sigma=2.0,  # Restrict to 2-sigma ellipse (high probability region)
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
        
        # Visualize the sampled pose
        AddMeshcatTriad(self._meshcat, "grasp_planning/sampled_pose", length=0.1, radius=0.003)
        self._meshcat.SetTransform("grasp_planning/sampled_pose", X_WM_sampled)
        
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
        
        # 6. Select best grasp
        print(f"  Running grasp selection...")
        best_X_G, best_cost = select_best_grasp(
            meshcat=self._meshcat,
            cloud=grasp_cloud,
            rng=self._rng,
            num_candidates=1000,
            num_to_draw=0,
            debug=False,
        )
        
        if best_X_G is not None:
            self._best_grasp_pose = best_X_G
            print(f"\n  Best grasp found!")
            print(f"    Cost: {best_cost:.3f}")
            print(f"    Position: {best_X_G.translation()}")
            rpy_grasp = RollPitchYaw(best_X_G.rotation())
            print(f"    RPY: [{rpy_grasp.roll_angle():.3f}, {rpy_grasp.pitch_angle():.3f}, {rpy_grasp.yaw_angle():.3f}]")
            
            # Visualize best grasp
            draw_grasp_candidate(self._meshcat, best_X_G, prefix="grasp_planning/best_grasp")
            
            # 7. Compute pre-grasp pose
            self._pregrasp_pose = compute_pregrasp_pose(best_X_G, offset_z=0.3)
            
            pre_grasp_pos = self._pregrasp_pose.translation()
            print(f"\n  Pre-grasp pose computed!")
            print(f"    Position: {pre_grasp_pos}")
            rpy_pregrasp = RollPitchYaw(self._pregrasp_pose.rotation())
            print(f"    RPY: [{rpy_pregrasp.roll_angle():.3f}, {rpy_pregrasp.pitch_angle():.3f}, {rpy_pregrasp.yaw_angle():.3f}]")
            
            # # Visualize pre-grasp pose
            # AddMeshcatTriad(self._meshcat, "grasp_planning/pregrasp_pose", length=0.15, radius=0.005)
            # self._meshcat.SetTransform("grasp_planning/pregrasp_pose", self._pregrasp_pose)
            # draw_grasp_candidate(self._meshcat, self._pregrasp_pose, prefix="grasp_planning/pregrasp_gripper")
            
        else:
            print(f"\n  No valid grasp found!")
            self._best_grasp_pose = None
            self._pregrasp_pose = None
