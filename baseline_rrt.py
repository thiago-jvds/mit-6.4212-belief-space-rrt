#!/usr/bin/env python3
"""
Baseline RRT Script - Vanilla RRT without Belief-Space Planning.

This script implements a "dumb" baseline that:
1. Makes immediate MAP predictions from 50/50 prior (no information gathering)
2. Skips RRBT trajectories entirely (no uncertainty reduction)
3. Samples grasp position from LARGE initial covariance
4. Expected to fail most of the time (demonstrates why RRBT is needed)

Usage:
    python baseline_rrt.py

Then open http://localhost:7001 in your browser.
"""

import random
import numpy as np
from enum import Enum, auto
from pathlib import Path
from manipulation.station import MakeHardwareStation, LoadScenario, AddPointClouds
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.mustard_depth_camera_example import MustardPointCloud
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    Meshcat,
    MeshcatParams,
    LeafSystem,
    BasicVector,
    AbstractValue,
    PiecewisePolynomial,
    Box,
    Rgba,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    PointCloud,
    Fields,
    BaseField,
    ConstantVectorSource,
)
import argparse
from src.perception.mustard_pose_estimator import MustardPoseEstimatorSystem, segment_by_yellow, ToPointCloud
from src.perception.light_and_dark import BinLightDarkRegionSensorSystem, MustardPositionLightDarkRegionSensorSystem
from manipulation.icp import IterativeClosestPoint
from pydrake.all import Concatenate
from src.visualization.covariance_ellipsoid import CovarianceEllipsoidSystem
from src.visualization.belief_bar_chart import BeliefBarChartSystem
from src.estimation.belief_estimator import BinBeliefEstimatorSystem
from src.estimation.mustard_position_estimator import MustardPositionBeliefEstimatorSystem
from src.grasping.grasp_selection import (
    select_best_grasp,
    compute_pregrasp_pose,
    sample_position_from_covariance,
    draw_grasp_candidate,
)
from src.utils.config_loader import load_config
from src.utils.ik_solver import solve_ik_for_pose
from src.utils.camera_pose_manager import restore_camera_pose
from src.planning.standard_rrt import rrt_planning
from src.simulation.simulation_tools import IiwaProblem


# ============================================================
# RANDOM SEED CONFIGURATION - Set this for deterministic runs
# ============================================================
RANDOM_SEED = 36  # Same as main.py for fair comparison

# Seed all random number generators for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np_rng = np.random.default_rng(seed=RANDOM_SEED)


class BaselinePlannerState(Enum):
    """State machine states for the baseline planner."""
    IDLE = auto()              # Waiting for setup
    PREDICT_BIN = auto()       # Immediately predict bin from 50/50 prior
    POSE_ESTIMATION = auto()   # Run pose estimation (ICP) on predicted bin
    GRASP_PLANNING = auto()    # Plan grasp with LARGE uncertainty (no reduction)
    RRT_PLANNING = auto()      # RRT motion planning from home to pregrasp
    GRASP_EXECUTING = auto()   # Execute grasp trajectory
    COMPLETE = auto()          # Done, report results


def path_to_trajectory(path: list, time_per_segment: float = 0.02) -> PiecewisePolynomial:
    """Convert a list of joint configurations to a time-parameterized trajectory."""
    path_array = np.array([np.array(q) for q in path])
    n_points = len(path)
    times = np.linspace(0, (n_points - 1) * time_per_segment, n_points)
    trajectory = PiecewisePolynomial.FirstOrderHold(times, path_array.T)
    return trajectory


class BaselinePlannerSystem(LeafSystem):
    """
    Baseline RRT Planner - No belief-space planning, immediate predictions.
    
    This planner demonstrates "naive" behavior:
    - Immediate bin prediction from 50/50 prior (coin flip)
    - No information-gathering trajectory (no RRBT1)
    - No uncertainty-reduction trajectory (no RRBT2)
    - Samples from LARGE initial covariance
    
    Expected to fail most of the time.
    """
    
    def __init__(self, plant, config, meshcat, scenario_path, rng=None):
        LeafSystem.__init__(self)
        
        self._plant = plant
        self._config = config
        self._meshcat = meshcat
        self._scenario_path = scenario_path
        self._rng = rng if rng is not None else np.random.default_rng()
        
        # Home position
        self._q_home = np.array(config.simulation.q_home)
        self._q_goal = None  # Will compute via IK
        
        # State machine
        self._state = BaselinePlannerState.IDLE
        
        # IMMEDIATE bin prediction from 50/50 prior (done at construction!)
        # This is the KEY difference from RRBT - no information gathering
        self._predicted_bin = self._rng.integers(0, 2)
        print(f"  [BASELINE] Immediate bin prediction from 50/50 prior: bin {self._predicted_bin}")
        
        # Results tracking
        self._results = {
            "true_bin": None,
            "predicted_bin": self._predicted_bin,  # Already known!
            "bin_prediction_correct": None,
            "icp_success": None,
            "icp_position": None,
            "sampled_position": None,
            "position_error": None,
            "grasp_planning_success": None,
            "rrt_success": None,  # RRT motion planning result
            "rrt_iterations": None,  # Number of RRT iterations
            "rrt_path_length": None,  # Number of waypoints in RRT path
            "grasp_execution_success": None,
            "overall_success": None,
            "failure_reason": None,
        }
        
        # Runtime parameters
        self._true_bin = None
        self._X_WM_mustard = None
        
        # Pose estimation results
        self._estimated_mustard_pose = None
        self._estimated_mustard_position = None
        
        # Large covariance (NOT reduced - this is the key difference from RRBT)
        self._initial_uncertainty = float(config.planner.mustard_position_initial_uncertainty)
        self._large_covariance = np.eye(2) * self._initial_uncertainty
        
        # Grasp planning
        self._best_grasp_pose = None
        self._pregrasp_pose = None
        self._grasp_candidates = []
        self._sampled_position = None
        
        # RRT planning
        self._q_pregrasp = None  # IK-solved pregrasp configuration (RRT goal)
        self._rrt_path = None    # RRT path from home to pregrasp
        
        # Grasp execution
        self._grasp_trajectory = None
        self._grasp_start_time = None
        self._grasp_close_time = None
        self._grasp_open_time = None
        self._grasp_end_position = None
        self._gripper_command = 0.1  # Start open
        
        # Compute IK targets
        self._compute_ik_targets()
        
        # FK context for gripper visualization
        self._fk_plant_context = self._plant.CreateDefaultContext()
        self._gripper_body = self._plant.GetBodyByName("body", self._plant.GetModelInstanceByName("wsg"))
        
        # Input port: estimated mustard pose from MustardPoseEstimatorSystem
        self._pose_input_port = self.DeclareAbstractInputPort(
            "estimated_mustard_pose",
            AbstractValue.Make(RigidTransform())
        )
        
        # Output ports
        self.DeclareVectorOutputPort(
            "iiwa_position_command",
            BasicVector(7),
            self.CalcJointCommand
        )
        
        self.DeclareVectorOutputPort(
            "wsg_position_command",
            BasicVector(1),
            self.CalcGripperCommand
        )
        
        # Output port for pose estimation trigger (1.0 only in POSE_ESTIMATION state)
        # Use time_ticket() to ensure re-evaluation at each time step
        # This breaks the algebraic loop (no INPUT port dependencies) while ensuring
        # the trigger updates as internal Python state changes
        self.DeclareVectorOutputPort(
            "pose_estimation_trigger",
            BasicVector(1),
            self.CalcPoseEstimationTrigger,
            {self.time_ticket()}
        )
        
        # NOTE: position_mean and position_covariance output ports removed
        # These now come from MustardPositionBeliefEstimatorSystem
        # The planner still uses _large_covariance internally for grasp planning
        
        print(f"BaselinePlannerSystem initialized:")
        print(f"  q_home: {self._q_home}")
        print(f"  Initial uncertainty: {self._initial_uncertainty}")
        print(f"  LARGE covariance (NOT reduced): {np.diag(self._large_covariance)}")
    
    def configure_for_execution(self, true_bin, X_WM_mustard=None):
        """Configure the planner with ground truth for evaluation."""
        self._true_bin = true_bin
        self._X_WM_mustard = X_WM_mustard
        self._results["true_bin"] = true_bin
        self._state = BaselinePlannerState.PREDICT_BIN
        print(f"BaselinePlannerSystem configured:")
        print(f"  true_bin: {true_bin}")
        print(f"  State: {self._state.name}")
    
    def get_state(self) -> BaselinePlannerState:
        return self._state
    
    def is_complete(self) -> bool:
        return self._state == BaselinePlannerState.COMPLETE
    
    def get_results(self) -> dict:
        return self._results
    
    def _compute_ik_targets(self):
        """Compute q_goal via IK."""
        tf_goal = self._config.simulation.tf_goal
        X_WG_goal = RigidTransform(
            RollPitchYaw(tf_goal.rpy).ToRotationMatrix(),
            tf_goal.translation
        )
        
        try:
            self._q_goal = np.array(solve_ik_for_pose(
                plant=self._plant,
                X_WG_target=X_WG_goal,
                q_nominal=tuple(self._q_home),
                theta_bound=0.1,
                pos_tol=0.01,
            ))
            print(f"  q_goal computed: {self._q_goal}")
        except RuntimeError as e:
            print(f"  IK failed for tf_goal: {e}")
            raise
    
    def CalcJointCommand(self, context, output):
        """State machine: calculate joint command based on current state."""
        t = context.get_time()
        q_command = self._q_home
        
        if self._state == BaselinePlannerState.IDLE:
            q_command = self._q_home
            
        elif self._state == BaselinePlannerState.PREDICT_BIN:
            # Report the prediction that was made at construction time
            print(f"\n{'='*60}")
            print("BASELINE: IMMEDIATE BIN PREDICTION (No RRBT)")
            print(f"{'='*60}")
            self._report_bin_prediction()
            self._state = BaselinePlannerState.POSE_ESTIMATION
            q_command = self._q_home
            
        elif self._state == BaselinePlannerState.POSE_ESTIMATION:
            # Skip first entry to allow trigger cache to update
            # Drake may have cached trigger=0.0 from previous state
            if not hasattr(self, '_pose_estimation_ready'):
                self._pose_estimation_ready = True
                q_command = self._q_home
                output.SetFromVector(q_command)
                return
            
            # Read estimated pose from input port
            print(f"\n{'='*60}")
            print("BASELINE: POSE ESTIMATION (ICP)")
            print(f"{'='*60}")
            
            try:
                self._estimated_mustard_pose = self._pose_input_port.Eval(context)
                self._estimated_mustard_position = self._estimated_mustard_pose.translation().copy()
                
                # Check if ICP succeeded (non-zero position)
                if np.linalg.norm(self._estimated_mustard_position) < 0.01:
                    print(f"  ICP FAILED - No mustard detected in predicted bin!")
                    print(f"  (Predicted bin {self._predicted_bin} was likely wrong)")
                    self._results["icp_success"] = False
                    self._results["failure_reason"] = "ICP failed - no mustard in predicted bin"
                    self._results["overall_success"] = False
                    self._state = BaselinePlannerState.COMPLETE
                else:
                    print(f"  ICP succeeded!")
                    print(f"  Estimated position: {self._estimated_mustard_position}")
                    self._results["icp_success"] = True
                    self._results["icp_position"] = self._estimated_mustard_position.copy()
                    
                    # Calculate position error if we have ground truth
                    if self._X_WM_mustard is not None:
                        true_pos = self._X_WM_mustard.translation()
                        error = np.linalg.norm(self._estimated_mustard_position - true_pos)
                        print(f"  True position: {true_pos}")
                        print(f"  ICP error: {error:.4f}m")
                    
                    self._state = BaselinePlannerState.GRASP_PLANNING
                    
            except Exception as e:
                # Perception failed - could be wrong bin OR other ICP issues
                print(f"  ICP FAILED with exception: {type(e).__name__}: {e}")
                if self._predicted_bin != self._true_bin:
                    print(f"  (Predicted bin {self._predicted_bin} was wrong - mustard in bin {self._true_bin})")
                    self._results["failure_reason"] = f"ICP failed - wrong bin prediction (error: {type(e).__name__})"
                else:
                    print(f"  (Predicted bin {self._predicted_bin} was CORRECT - ICP failed for other reason)")
                    self._results["failure_reason"] = f"ICP failed despite correct bin prediction (error: {type(e).__name__}: {e})"
                self._results["icp_success"] = False
                self._results["overall_success"] = False
                self._state = BaselinePlannerState.COMPLETE
            
            q_command = self._q_home
            
        elif self._state == BaselinePlannerState.GRASP_PLANNING:
            print(f"\n{'='*60}")
            print("BASELINE: GRASP PLANNING (LARGE Uncertainty)")
            print(f"{'='*60}")
            self._run_grasp_planning()
            
            if self._grasp_candidates:
                if self._compute_grasp_trajectory():
                    self._grasp_start_time = t
                    self._gripper_command = 0.1
                    self._state = BaselinePlannerState.GRASP_EXECUTING
                    print(f"  Starting grasp execution at t={t:.2f}s")
                else:
                    print(f"  All grasp candidates failed IK or RRT")
                    self._results["grasp_planning_success"] = False
                    self._results["failure_reason"] = "Grasp IK/RRT planning failed for all candidates"
                    self._results["overall_success"] = False
                    self._state = BaselinePlannerState.COMPLETE
            else:
                print(f"  No valid grasp candidates found")
                self._results["grasp_planning_success"] = False
                self._results["failure_reason"] = "No valid grasp candidates"
                self._results["overall_success"] = False
                self._state = BaselinePlannerState.COMPLETE
            
            q_command = self._q_home
            
        elif self._state == BaselinePlannerState.GRASP_EXECUTING:
            t_traj = t - self._grasp_start_time
            
            # Update gripper command
            if self._grasp_open_time is not None and t_traj >= self._grasp_open_time:
                self._gripper_command = 0.1  # Open (release)
            elif t_traj >= self._grasp_close_time:
                self._gripper_command = 0.0  # Close (grasp)
            else:
                self._gripper_command = 0.1  # Open (before grasp)
            
            if t_traj >= self._grasp_trajectory.end_time():
                self._state = BaselinePlannerState.COMPLETE
                self._results["grasp_execution_success"] = True
                self._results["overall_success"] = True
                print(f"\n{'='*60}")
                print("BASELINE: EXECUTION COMPLETE")
                print(f"{'='*60}")
                q_command = self._grasp_end_position
            else:
                q_command = self._grasp_trajectory.value(t_traj).flatten()
                
        elif self._state == BaselinePlannerState.COMPLETE:
            if self._grasp_end_position is not None:
                q_command = self._grasp_end_position
            else:
                q_command = self._q_home
        
        output.SetFromVector(q_command)
    
    def CalcGripperCommand(self, context, output):
        output.SetFromVector([self._gripper_command])

    def CalcPoseEstimationTrigger(self, context, output):
        """Output 1.0 only when in POSE_ESTIMATION state, 0.0 otherwise."""
        trigger_value = 1.0 if self._state == BaselinePlannerState.POSE_ESTIMATION else 0.0
        output.SetFromVector([trigger_value])

    def _report_bin_prediction(self):
        """
        BASELINE: Report the bin prediction made from 50/50 prior.
        
        The prediction was made at construction time (before knowing true bin).
        This is the KEY DIFFERENCE from RRBT - no information gathering.
        Expected to be wrong 50% of the time.
        """
        # Update results now that we know the true bin
        self._results["bin_prediction_correct"] = (self._predicted_bin == self._true_bin)
        
        print(f"  Prior belief: [0.5, 0.5]")
        print(f"  Predicted bin: {self._predicted_bin} (chosen at construction)")
        print(f"  True bin: {self._true_bin}")
        
        if self._predicted_bin == self._true_bin:
            print(f"  Prediction: CORRECT (lucky!)")
        else:
            print(f"  Prediction: WRONG (as expected ~50% of time)")
    
    def _run_grasp_planning(self):
        """
        BASELINE: Grasp planning with LARGE (unreduced) uncertainty.
        
        Key difference from RRBT: Uses initial large covariance, not reduced.
        The sampled position is expected to be far from truth.
        """
        print(f"  Using LARGE covariance (NO uncertainty reduction):")
        print(f"    Covariance diagonal: {np.diag(self._large_covariance)}")
        print(f"    Initial uncertainty: {self._initial_uncertainty}")
        
        # Sample position from LARGE covariance
        self._sampled_position = sample_position_from_covariance(
            mean=self._estimated_mustard_position,
            covariance=self._large_covariance,
            rng=self._rng,
            max_sigma=2.0,  # Same as RRBT
        )
        
        # Calculate offset from ICP estimate
        offset_xy = self._sampled_position[:2] - self._estimated_mustard_position[:2]
        offset_norm = np.linalg.norm(offset_xy)
        
        print(f"\n  ICP estimated position: {self._estimated_mustard_position}")
        print(f"  SAMPLED position: {self._sampled_position}")
        print(f"  X-Y offset from ICP: {offset_xy} (norm: {offset_norm:.4f}m)")
        
        self._results["sampled_position"] = self._sampled_position.copy()
        
        # Calculate error from TRUE position
        if self._X_WM_mustard is not None:
            true_pos = self._X_WM_mustard.translation()
            true_error = np.linalg.norm(self._sampled_position - true_pos)
            print(f"  True position: {true_pos}")
            print(f"  ERROR from true position: {true_error:.4f}m")
            self._results["position_error"] = true_error
        
        # Create candidate pose: sampled position + ICP rotation
        X_WM_sampled = RigidTransform(
            self._estimated_mustard_pose.rotation(),
            self._sampled_position
        )
        
        # Load mustard model and transform to sampled pose
        print(f"\n  Loading mustard model...")
        mustard_model = MustardPointCloud()
        model_pcl = mustard_model.xyzs()
        model_world_xyz = X_WM_sampled @ model_pcl
        
        # Create point cloud with normals
        grasp_cloud = PointCloud(
            model_world_xyz.shape[1],
            Fields(BaseField.kXYZs | BaseField.kNormals)
        )
        grasp_cloud.mutable_xyzs()[:] = model_world_xyz
        grasp_cloud.EstimateNormals(radius=0.05, num_closest=30)
        
        centroid = np.mean(model_world_xyz, axis=1)
        grasp_cloud.FlipNormalsTowardPoint(centroid + np.array([0, 0, 1]))
        
        # Visualize
        self._meshcat.SetObject(
            "baseline/grasp_cloud",
            grasp_cloud,
            point_size=0.003,
            rgba=Rgba(0, 1, 1, 1)
        )
        
        # Get grasp candidates
        print(f"\n  Running grasp selection...")
        self._grasp_candidates = select_best_grasp(
            meshcat=self._meshcat,
            cloud=grasp_cloud,
            rng=self._rng,
            num_candidates=1000,
            num_to_draw=0,
            num_to_return=20,
            debug=False,
        )
        
        if self._grasp_candidates:
            print(f"\n  Found {len(self._grasp_candidates)} grasp candidates")
            self._results["grasp_planning_success"] = True
        else:
            print(f"\n  No valid grasp candidates found!")
            self._results["grasp_planning_success"] = False

    def _run_rrt_to_pregrasp(self):
        """
        Use standard RRT to plan collision-free path from home to pregrasp.
        
        This is the ACTUAL RRT motion planning step:
        - Creates a collision-free path from home configuration to pregrasp configuration
        - Uses the standard RRT (Rapidly-exploring Random Tree) algorithm
        - Explores configuration space by randomly sampling and growing a tree
        
        Returns:
            True if RRT planning succeeded, False otherwise.
        """
        print(f"\n  ============================================================")
        print(f"  RRT MOTION PLANNING (Home -> Pregrasp)")
        print(f"  ============================================================")
        print(f"    Start configuration (home): {np.round(self._q_home, 3)}")
        print(f"    Goal configuration (pregrasp): {np.round(self._q_pregrasp, 3)}")
        print(f"    Algorithm: Standard RRT (Rapidly-exploring Random Tree)")
        print(f"    Max iterations: 2000")
        print(f"    Goal sampling probability: 0.1 (10%)")
        print()
        
        # Create IiwaProblem for RRT
        # This sets up the collision checking and configuration space bounds
        problem = IiwaProblem(
            q_start=tuple(self._q_home),
            q_goal=tuple(self._q_pregrasp),
            gripper_setpoint=0.1,  # Open gripper
            meshcat=self._meshcat,
            is_visualizing=False,
        )
        
        # Run RRT planning - this is the actual RRT algorithm
        print(f"    Starting RRT tree growth...")
        rrt_path, iterations = rrt_planning(
            problem,
            max_iterations=2000,
            prob_sample_q_goal=0.1,
            rng=self._rng,
            verbose=False,  # Set to True for detailed per-iteration logging
        )
        
        if rrt_path is None:
            print(f"    [RRT RESULT] FAILED - Could not find collision-free path")
            print(f"    Iterations attempted: {iterations}")
            self._results["rrt_success"] = False
            self._results["rrt_iterations"] = iterations
            return False
        
        # Record RRT success metrics
        self._results["rrt_success"] = True
        self._results["rrt_iterations"] = iterations
        self._results["rrt_path_length"] = len(rrt_path)
        
        print(f"  ============================================================")
        print(f"  RRT PLANNING COMPLETE")
        print(f"  ============================================================")
        
        self._rrt_path = rrt_path
        return True

    def _compute_grasp_trajectory(self):
        """Compute grasp execution trajectory (same logic as PlannerSystem)."""
        print(f"\nComputing grasp execution trajectory...")
        
        q_current = self._q_home
        
        if not self._grasp_candidates:
            return False
        
        q_pregrasp = None
        q_grasp = None
        q_lift = None
        q_drop = None
        
        for candidate_idx, (cost, grasp_pose) in enumerate(self._grasp_candidates):
            print(f"\n  Candidate {candidate_idx + 1}/{len(self._grasp_candidates)} (cost={cost:.3f}):")
            
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
            
            # Compute lift pose
            lift_pos = grasp_pose.translation() + np.array([0, 0, 0.3])
            X_lift = RigidTransform(grasp_pose.rotation(), lift_pos)
            
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
            
            # Compute drop pose
            drop_pos = np.array([0.5, -0.5, lift_pos[2]])
            grasp_x_world = grasp_pose.rotation().matrix()[:, 0]
            grasp_x_horizontal = np.array([grasp_x_world[0], grasp_x_world[1], 0.0])
            grasp_x_horizontal_norm = np.linalg.norm(grasp_x_horizontal)
            if grasp_x_horizontal_norm > 1e-6:
                gripper_x = grasp_x_horizontal / grasp_x_horizontal_norm
            else:
                gripper_x = np.array([1.0, 0.0, 0.0])
            
            gripper_y = np.array([0.0, 0.0, -1.0])
            gripper_z = np.cross(gripper_x, gripper_y)
            gripper_z = gripper_z / np.linalg.norm(gripper_z)
            gripper_x = np.cross(gripper_y, gripper_z)
            gripper_x = gripper_x / np.linalg.norm(gripper_x)
            
            R_drop_flat = RotationMatrix(np.column_stack([gripper_x, gripper_y, gripper_z]))
            X_drop = RigidTransform(R_drop_flat, drop_pos)
            
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
            
            # All IK checks passed - now run RRT from home to pregrasp
            # This is the actual RRT motion planning step!
            self._q_pregrasp = q_pregrasp  # Store for RRT goal
            print(f"\n    All IK checks passed! Now running RRT motion planning...")
            if not self._run_rrt_to_pregrasp():
                print(f"    RRT motion planning FAILED - trying next grasp candidate")
                continue
            
            print(f"    RRT motion planning: SUCCESS ({len(self._rrt_path)} collision-free waypoints)")
            
            # Success! Use this grasp
            print(f"\n  SUCCESS! Using grasp candidate {candidate_idx + 1}")
            
            self._best_grasp_pose = grasp_pose
            self._pregrasp_pose = pregrasp_pose
            
            # Visualize
            draw_grasp_candidate(self._meshcat, grasp_pose, prefix="baseline/best_grasp")
            AddMeshcatTriad(self._meshcat, "baseline/grasp_pose", length=0.15, radius=0.005)
            self._meshcat.SetTransform("baseline/grasp_pose", grasp_pose)
            
            break
        else:
            print(f"\n  All candidates failed IK validation!")
            return False
        
        # Build combined trajectory: RRT path (home->pregrasp) + IK segment (pregrasp->grasp->lift->drop)
        time_per_rad = 1.0
        rrt_time_per_segment = 0.05  # Time per RRT waypoint
        
        # Convert RRT path to trajectory
        rrt_trajectory = path_to_trajectory(self._rrt_path, time_per_segment=rrt_time_per_segment)
        rrt_duration = rrt_trajectory.end_time()
        
        print(f"\n  Building combined trajectory:")
        print(f"    RRT segment: {len(self._rrt_path)} waypoints, {rrt_duration:.2f}s")
        
        # Calculate times for IK segment (after RRT completes)
        dist_pregrasp_to_grasp = np.linalg.norm(q_grasp - q_pregrasp)
        dist_grasp_to_lift = np.linalg.norm(q_lift - q_grasp)
        dist_lift_to_drop = np.linalg.norm(q_drop - q_lift)
        
        t_pregrasp = rrt_duration  # RRT ends at pregrasp
        t_grasp = t_pregrasp + dist_pregrasp_to_grasp * time_per_rad
        t_hold = t_grasp + 1.0  # Hold for gripper close
        t_lift = t_hold + dist_grasp_to_lift * time_per_rad
        t_drop = t_lift + dist_lift_to_drop * time_per_rad
        t_release = t_drop + 0.5
        t_end = t_release + 0.5
        
        # Build times array: RRT waypoint times + IK segment times
        rrt_times = np.linspace(0, rrt_duration, len(self._rrt_path))
        ik_times = np.array([t_grasp, t_hold, t_lift, t_drop, t_release, t_end])
        times = np.concatenate([rrt_times, ik_times])
        
        # Build waypoints array: RRT waypoints + IK waypoints
        rrt_waypoints = np.array([np.array(q) for q in self._rrt_path]).T  # (7, n_rrt)
        ik_waypoints = np.column_stack([
            q_grasp, q_grasp, q_lift, q_drop, q_drop, q_drop
        ])
        waypoints = np.hstack([rrt_waypoints, ik_waypoints])
        
        self._grasp_trajectory = PiecewisePolynomial.FirstOrderHold(times, waypoints)
        self._grasp_close_time = t_grasp
        self._grasp_open_time = t_release
        self._grasp_end_position = q_drop
        
        print(f"    IK segment: pregrasp->grasp->lift->drop")
        print(f"    Total duration: {t_end:.1f}s")
        print(f"    Gripper closes at: {t_grasp:.1f}s")
        print(f"    Gripper opens at: {t_release:.1f}s")
        
        return True


def place_mustard_bottle_randomly_in_bin(meshcat, plant, plant_context, true_bin, np_rng):
    """Place the mustard bottle randomly within the specified bin."""
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
    
    X_BM = RigidTransform(random_rotation, [random_x, random_y, random_z])
    X_WM = X_WB.multiply(X_BM)
    
    mustard_body = plant.GetBodyByName("base_link_mustard")
    plant.SetFreeBodyPose(plant_context, mustard_body, X_WM)
    
    print(f"  Placed mustard bottle in bin{true_bin}:")
    print(f"    Bin position: {X_WB.translation()}")
    print(f"    Random offset: [{random_x:.3f}, {random_y:.3f}, {random_z:.3f}]")
    print(f"    World position: {X_WM.translation()}")
    
    return X_WM


def print_results(results):
    """Print formatted results at the end."""
    print(f"\n{'='*60}")
    print("BASELINE RRT RESULTS (No Belief-Space Planning)")
    print(f"{'='*60}")
    
    # Bin prediction
    print(f"\n[1] BIN PREDICTION (from 50/50 prior - no info gathering)")
    print(f"    True bin: {results['true_bin']}")
    print(f"    Predicted bin: {results['predicted_bin']}", end="")
    if results['bin_prediction_correct']:
        print(f"  [CORRECT - Lucky!]")
    else:
        print(f"  [WRONG - Expected ~50% of time]")
    
    # ICP / Pose estimation
    print(f"\n[2] POSE ESTIMATION (ICP)")
    print(f"    ICP success: {results['icp_success']}")
    if results['icp_position'] is not None:
        print(f"    ICP position: [{results['icp_position'][0]:.4f}, {results['icp_position'][1]:.4f}, {results['icp_position'][2]:.4f}]")
    elif not results['icp_success'] and results.get('failure_reason'):
        print(f"    Failure: {results['failure_reason']}")
    
    # Position sampling
    print(f"\n[3] POSITION SAMPLING (from LARGE unreduced covariance)")
    if results['sampled_position'] is not None:
        print(f"    Sampled position: [{results['sampled_position'][0]:.4f}, {results['sampled_position'][1]:.4f}, {results['sampled_position'][2]:.4f}]")
    else:
        print(f"    Not reached (earlier failure)")
    if results['position_error'] is not None:
        print(f"    Error from TRUE position: {results['position_error']:.4f}m")
        if results['position_error'] > 0.05:
            print(f"    [HIGH ERROR - Large covariance caused bad sample!]")
    
    # Grasp planning (includes RRT motion planning)
    print(f"\n[4] GRASP PLANNING + RRT MOTION PLANNING")
    print(f"    Method: Standard RRT (Rapidly-exploring Random Tree)")
    print(f"    Purpose: Find collision-free path from home to pregrasp pose")
    if results['grasp_planning_success'] is not None:
        print(f"    Grasp candidates found: {results['grasp_planning_success']}")
    else:
        print(f"    Grasp planning: Not reached (earlier failure)")
    
    # RRT-specific results
    if results['rrt_success'] is not None:
        print(f"    RRT motion planning: {'SUCCESS' if results['rrt_success'] else 'FAILED'}")
        if results['rrt_iterations'] is not None:
            print(f"    RRT iterations: {results['rrt_iterations']}")
        if results['rrt_path_length'] is not None:
            print(f"    RRT path waypoints: {results['rrt_path_length']}")
    else:
        print(f"    RRT motion planning: Not reached (earlier failure)")
    
    # Grasp execution
    print(f"\n[5] GRASP EXECUTION")
    if results['grasp_execution_success'] is not None:
        print(f"    Success: {results['grasp_execution_success']}")
    else:
        print(f"    Not reached (earlier failure)")
    
    # Overall result
    print(f"\n{'='*60}")
    if results['overall_success']:
        print(f"OVERALL RESULT: SUCCESS")
        print(f"  (Got lucky with prediction AND position sample!)")
    else:
        print(f"OVERALL RESULT: FAILED")
        if results['failure_reason']:
            print(f"  Reason: {results['failure_reason']}")
        print(f"\n  This demonstrates why RRBT is needed:")
        print(f"  - Without info gathering, bin prediction is random (50% wrong)")
        print(f"  - Without uncertainty reduction, position samples are unreliable")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Baseline RRT (No Belief-Space Planning)")
    parser.add_argument(
        "--visualize",
        type=str,
        nargs="?",
        const="True",
        default="True",
        help="Enable/Disable Meshcat visualization",
    )
    args = parser.parse_args()

    import sys
    
    print("=" * 60)
    print("BASELINE RRT - Vanilla RRT Without Belief-Space Planning")
    print("=" * 60)
    print("\nThis baseline demonstrates naive behavior:")
    print("  - Immediate bin prediction from 50/50 prior (no info gathering)")
    print("  - No RRBT trajectory (no uncertainty reduction)")
    print("  - Samples from LARGE initial covariance")
    print("  - Expected to FAIL most of the time\n")
    sys.stdout.flush()

    # Load configuration
    config = load_config()
    print("Loaded Configuration:")
    print(f"    > Initial uncertainty: {config.planner.mustard_position_initial_uncertainty}")
    print()
    sys.stdout.flush()

    # Start Meshcat (use port 7001 to avoid conflict with main.py)
    try:
        params = MeshcatParams()
        params.port = 7001
        meshcat = Meshcat(params=params)
    except RuntimeError as e:
        print(f"\n[ERROR] Could not start Meshcat on port 7001: {e}")
        raise

    if args.visualize == "True":
        restore_camera_pose(meshcat)

    # Load scenario
    scenario_path = Path(__file__).parent / "config" / "scenario.yaml"
    with open(scenario_path, "r") as f:
        scenario = LoadScenario(data=f.read())

    # Build diagram
    print("\nBuilding diagram...")
    builder = DiagramBuilder()
    
    station = builder.AddSystem(MakeHardwareStation(scenario=scenario, meshcat=meshcat))
    station.set_name("HardwareStation")
    plant = station.GetSubsystemByName("plant")

    # ============================================================
    # GET BIN TRANSFORMS FOR BELIEF BAR CHART POSITIONING
    # ============================================================
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
    X_bin0_chart = RigidTransform([-0.22, 0.29, 0.21])
    X_bin1_chart = RigidTransform([-0.22, -0.29, 0.21])
    
    print(f"  Bin0 position: {X_W_bin0.translation()}")
    print(f"  Bin1 position: {X_W_bin1.translation()}")

    # Add baseline planner
    planner = builder.AddSystem(BaselinePlannerSystem(plant, config, meshcat, scenario_path, rng=np_rng))
    planner.set_name("BaselinePlanner")
    
    builder.Connect(
        planner.GetOutputPort("iiwa_position_command"),
        station.GetInputPort("iiwa.position")
    )
    builder.Connect(
        planner.GetOutputPort("wsg_position_command"),
        station.GetInputPort("wsg.position")
    )

    # ============================================================
    # ADD LIGHT/DARK PERCEPTION SYSTEMS
    # ============================================================
    # Add Bin Light/Dark Perception System
    print("  Adding BinLightDarkRegionSensorSystem...")
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
    bin_perception_sys.set_name("BinLightDarkPerception")
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        bin_perception_sys.GetInputPort("iiwa.position"),
    )

    # Add Mustard Position Light/Dark Perception System
    print("  Adding MustardPositionLightDarkRegionSensorSystem...")
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

    # Add point clouds
    print("  Adding point cloud generation...")
    to_point_cloud = AddPointClouds(
        scenario=scenario,
        station=station,
        builder=builder,
        meshcat=None,
    )
    print(f"    Added point cloud converters for: {list(to_point_cloud.keys())}")

    # Export point cloud ports for visualization/debugging
    point_cloud_output_ports = {}
    for camera_name, converter in to_point_cloud.items():
        port_name = f"{camera_name}_point_cloud"
        builder.ExportOutput(converter.get_output_port(), port_name)
        point_cloud_output_ports[camera_name] = port_name

    # Add MustardPoseEstimatorSystem
    print("  Adding MustardPoseEstimatorSystem...")
    pose_estimator = builder.AddSystem(
        MustardPoseEstimatorSystem(meshcat=meshcat, n_bins=2)
    )
    pose_estimator.set_name("MustardPoseEstimator")

    # Connect cameras to pose estimator
    for i in range(6):
        camera_name = f"camera{i}"
        if camera_name in to_point_cloud:
            builder.Connect(
                to_point_cloud[camera_name].get_output_port(),
                pose_estimator.GetInputPort(f"camera{i}_point_cloud")
            )

    # ============================================================
    # ADD BELIEF ESTIMATORS
    # ============================================================
    # Add Bin Belief Estimator (Bayes Filter)
    # Note: true_bin will be set after we randomly choose it
    print("  Adding BinBeliefEstimatorSystem...")
    belief_estimator = builder.AddSystem(
        BinBeliefEstimatorSystem(
            n_bins=2,
            true_bin=0,  # Placeholder, will be updated after setup
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

    # For pose estimator: Use planner's prediction (NOT the real belief)
    # This ensures correct cameras are selected for the predicted bin
    # The real belief_estimator is still used for visualization (bar charts)
    predicted_bin = planner._predicted_bin
    if predicted_bin == 0:
        belief_for_pose_est = np.array([0.99, 0.01])
    else:
        belief_for_pose_est = np.array([0.01, 0.99])
    
    print(f"  Pose estimator using predicted bin {predicted_bin}: belief={belief_for_pose_est}")

    belief_source_for_pose = builder.AddSystem(ConstantVectorSource(belief_for_pose_est))
    belief_source_for_pose.set_name("BeliefSourceForPoseEstimator")

    builder.Connect(
        belief_source_for_pose.get_output_port(),
        pose_estimator.GetInputPort("belief")
    )
    # Connect planner's trigger output (1.0 only in POSE_ESTIMATION state)
    # This prevents pose estimation from running during initialization/settling
    builder.Connect(
        planner.GetOutputPort("pose_estimation_trigger"),
        pose_estimator.GetInputPort("estimation_trigger")
    )

    # Connect pose estimator output to planner
    builder.Connect(
        pose_estimator.GetOutputPort("estimated_pose"),
        planner.GetInputPort("estimated_mustard_pose")
    )

    # ============================================================
    # ADD MUSTARD POSITION BELIEF ESTIMATOR (Kalman Filter)
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

    # Connect estimated pose from ICP (for initial position)
    builder.Connect(
        pose_estimator.GetOutputPort("estimated_pose"),
        mustard_belief_estimator.GetInputPort("estimated_pose")
    )

    # ============================================================
    # ADD BELIEF BAR CHART VISUALIZATION
    # ============================================================
    print("  Adding BeliefBarChartSystem...")
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
    # ADD COVARIANCE ELLIPSOID VISUALIZATION
    # ============================================================
    print("  Adding CovarianceEllipsoidSystem...")
    covariance_viz = builder.AddSystem(
        CovarianceEllipsoidSystem(
            meshcat=meshcat,
            scale_factor=3.0,
            color=Rgba(1.0, 0.0, 0.0, 0.5),
        )
    )
    covariance_viz.set_name("CovarianceEllipsoid")

    # Connect position and covariance from mustard position belief estimator
    builder.Connect(
        mustard_belief_estimator.GetOutputPort("position_mean"),
        covariance_viz.GetInputPort("position")
    )
    builder.Connect(
        mustard_belief_estimator.GetOutputPort("covariance"),
        covariance_viz.GetInputPort("covariance")
    )

    # Build diagram
    diagram = builder.Build()
    diagram.set_name("BaselineRRTDiagram")

    # Initialize simulator
    print("\nInitializing simulator...")
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    # Visualize light regions
    if args.visualize == "True":
        # Visualize bin light region indicator (green)
        meshcat.SetObject(
            "bin_light_region_indicator",
            Box(*config.simulation.bin_light_size),
            Rgba(0, 1, 0, 0.3),  # Green, 0.3 Alpha
        )
        meshcat.SetTransform(
            "bin_light_region_indicator",
            RigidTransform(RotationMatrix(), config.simulation.bin_light_center),
        )
        # Visualize mustard position light region indicator (orange)
        meshcat.SetObject(
            "mustard_position_light_region_indicator",
            Box(*config.simulation.mustard_position_light_size),
            Rgba(1.0, 0.5, 0.0, 0.3),  # Orange, 0.3 Alpha
        )
        meshcat.SetTransform(
            "mustard_position_light_region_indicator",
            RigidTransform(RotationMatrix(), config.simulation.mustard_position_light_center),
        )

    # Step briefly to initialize
    simulator.AdvanceTo(0.1)

    # Setup environment
    print("\n" + "=" * 60)
    print("ENVIRONMENT SETUP")
    print("=" * 60)

    sim_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(sim_context)

    # Randomly choose true bin
    true_bin = np_rng.integers(0, 2)
    print(f"Randomly chosen true bin: {true_bin}")

    # Place mustard bottle
    X_WM_mustard = place_mustard_bottle_randomly_in_bin(
        meshcat, plant, plant_context, true_bin, np_rng
    )

    diagram.ForcedPublish(sim_context)

    # Update belief estimator with correct true_bin
    belief_estimator._true_bin = true_bin

    # Let the mustard bottle fall and settle via gravity
    # (The bottle is placed at z=0.2 above the bin floor)
    print("  Waiting for mustard bottle to settle...")
    simulator.AdvanceTo(0.8)  # Wait until t=0.8s for settling
    print("  Bottle settled.")

    # Force camera update by evaluating point clouds after settling
    # This ensures RGBD data reflects the settled mustard bottle
    print("  Updating camera point clouds...")
    sim_context = simulator.get_context()
    for camera_name, port_name in point_cloud_output_ports.items():
        pcl = diagram.GetOutputPort(port_name).Eval(sim_context)
        print(f"    {camera_name}: {pcl.size()} points")
        # Visualize immediately for debugging
        meshcat.SetObject(
            f"{camera_name}.cloud",
            pcl,
            point_size=0.003,
        )
    print("  Camera point clouds updated and visualized.")

    # Configure planner (after bottle has settled and cameras updated)
    planner.configure_for_execution(true_bin, X_WM_mustard)

    # Start recording
    meshcat.StartRecording()

    print("\nMeshcat running at http://localhost:7001")
    print("Running simulation... (Press Ctrl+C to exit)")
    print()

    # Run simulation
    try:
        while True:
            current_time = simulator.get_context().get_time()
            simulator.AdvanceTo(current_time + 0.1)
            
            if planner.is_complete():
                break
                
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")

    # Stop recording
    meshcat.StopRecording()
    meshcat.PublishRecording()

    # Print results
    print_results(planner.get_results())

    print("\nExecution complete! Replay available in Meshcat.")
    print("Press Ctrl+C to exit.")
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
