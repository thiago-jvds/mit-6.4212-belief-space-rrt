"""
Simulation Tools for IIWA Robot Planning Problems

This module provides:
- ManipulationStationSim: Drake-based collision checking and visualization
- IiwaProblem: Base class for IIWA planning problems
- IiwaProblemBinBelief: Belief-space problem with discrete Bayes filter
- IiwaProblemObjectPositionBelief: Belief-space problem with Kalman filter
"""

import numpy as np
from pydrake.all import DiagramBuilder, Meshcat
from manipulation.station import MakeHardwareStation, LoadScenario, AddPointClouds

from manipulation.exercises.trajectories.rrt_planner.robot import (
    ConfigurationSpace,
    Range,
)
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import Problem

from src.estimation.bayes_filter import (
    calculate_misclassification_risk,
    expected_posterior_all_bins,
)


class ManipulationStationSim:
    def __init__(self, meshcat: Meshcat, is_visualizing: bool = False) -> None:
        builder = DiagramBuilder()
        scenario = LoadScenario(filename="config/scenario.yaml")
        self.station = builder.AddSystem(
            MakeHardwareStation(scenario, meshcat=meshcat if is_visualizing else None)
        )

        self.to_point_cloud = AddPointClouds(
            scenario=scenario,
            station=self.station,
            builder=builder,
            meshcat=meshcat,  # type: ignore
        )
        self.plant = self.station.GetSubsystemByName("plant")  # type: ignore
        self.scene_graph = self.station.GetSubsystemByName("scene_graph")  # type: ignore
        self.is_visualizing = is_visualizing

        # scene graph query output port.
        self.query_output_port = self.scene_graph.GetOutputPort("query")

        self.diagram = builder.Build()

        # contexts
        self.context_diagram = self.diagram.CreateDefaultContext()
        self.context_station = self.diagram.GetSubsystemContext(
            self.station, self.context_diagram
        )
        self.station.GetInputPort("iiwa.position").FixValue(
            self.context_station,
            np.zeros(7),  # type: ignore
        )
        self.station.GetInputPort("wsg.position").FixValue(self.context_station, [0.1])  # type: ignore
        self.context_scene_graph = self.station.GetSubsystemContext(
            self.scene_graph, self.context_station
        )
        self.context_plant = self.station.GetMutableSubsystemContext(
            self.plant, self.context_station
        )
        # mark initial configuration
        self.q0 = self.plant.GetPositions(
            self.context_plant, self.plant.GetModelInstanceByName("iiwa")
        )
        if is_visualizing:
            self.DrawStation(self.q0, 0.1)

    def SetStationConfiguration(
        self,
        q_iiwa: np.ndarray,
        gripper_setpoint: float,
    ) -> None:
        """
        :param q_iiwa: (7,) numpy array, joint angle of robots in radian.
        :param gripper_setpoint: float, gripper opening distance in meters.
        :return:
        """
        self.plant.SetPositions(
            self.context_plant,
            self.plant.GetModelInstanceByName("iiwa"),
            q_iiwa,
        )
        self.plant.SetPositions(
            self.context_plant,
            self.plant.GetModelInstanceByName("wsg"),
            [-gripper_setpoint / 2, gripper_setpoint / 2],
        )

    def DrawStation(
        self,
        q_iiwa: np.ndarray,
        gripper_setpoint: float,
    ) -> None:
        if not self.is_visualizing:
            print("collision checker is not initialized with visualization.")
            return
        self.SetStationConfiguration(
            q_iiwa,
            gripper_setpoint,
        )
        self.diagram.ForcedPublish(self.context_diagram)

    def ExistsCollision(self, q_iiwa: np.ndarray, gripper_setpoint: float) -> bool:
        self.SetStationConfiguration(q_iiwa, gripper_setpoint)
        query_object = self.query_output_port.Eval(self.context_scene_graph)
        collision_pairs = query_object.ComputePointPairPenetration()

        return len(collision_pairs) > 0


class IiwaProblem(Problem):
    def __init__(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        gripper_setpoint: float,
        meshcat: Meshcat,
        is_visualizing=False,
    ) -> None:
        self.gripper_setpoint = gripper_setpoint
        self.is_visualizing = is_visualizing

        self.collision_checker = ManipulationStationSim(
            is_visualizing=is_visualizing, meshcat=meshcat
        )

        # Construct configuration space for IIWA.
        plant = self.collision_checker.plant
        nq = 7
        joint_limits = np.zeros((nq, 2))
        for i in range(nq):
            joint = plant.GetJointByName("iiwa_joint_%i" % (i + 1))
            joint_limits[i, 0] = joint.position_lower_limits().item()
            joint_limits[i, 1] = joint.position_upper_limits().item()

        range_list = []
        for joint_limit in joint_limits:
            range_list.append(Range(joint_limit[0], joint_limit[1]))

        def l2_distance(q: tuple):
            sum = 0
            for q_i in q:
                sum += q_i**2
            return np.sqrt(sum)

        max_steps = nq * [np.pi / 180]  # two degrees
        cspace_iiwa = ConfigurationSpace(range_list, l2_distance, max_steps)  # type: ignore

        # Call base class constructor.
        super().__init__(
            x=10,  # not used.
            y=10,  # not used.
            robot=None,  # not used.
            obstacles=None,  # not used.
            start=tuple(q_start),
            goal=tuple(q_goal),
            cspace=cspace_iiwa,
        )

    def collide(self, configuration: np.ndarray) -> bool:
        q = np.array(configuration)
        return self.collision_checker.ExistsCollision(
            q,
            self.gripper_setpoint,
        )

    def visualize_path(self, path: list[tuple]) -> None:
        if path is not None:
            # show path in meshcat
            for q in path:
                q = np.array(q)
                self.collision_checker.DrawStation(
                    q,
                    self.gripper_setpoint,
                )


class IiwaProblemBinBelief(IiwaProblem):
    """
    Belief-space planning problem using a discrete n-bin Bayes filter.
    
    The sensor model uses TPR (True Positive Rate) and FPR (False Positive Rate)
    instead of Gaussian noise covariance matrices.
    
    In light region: Informative sensor (high TPR, low FPR)
    In dark region: Uninformative sensor (TPR=0.5, FPR=0.5 - coin flip)
    
    Implements the RRBTProblem protocol for use with RRBT_Tree and RRBT_tools.
    """
    
    def __init__(
        self,
        q_start,
        q_goal,
        gripper_setpoint,
        meshcat,
        light_center,
        light_size,
        tpr_light: float = 0.80,
        fpr_light: float = 0.15,
        n_bins: int = 2,
        true_bin: int = 0,
        max_bin_uncertainty: float = 0.01,
        lambda_weight: float = 1.0
    ):
        """
        Args:
            q_start: Starting joint configuration
            q_goal: Goal joint configuration
            gripper_setpoint: Gripper opening width
            meshcat: Meshcat visualizer instance
            light_center: Center of the light region [x, y, z]
            light_size: Size of the light region [dx, dy, dz]
            tpr_light: True Positive Rate in light region (default 0.80)
            fpr_light: False Positive Rate in light region (default 0.15)
            n_bins: Number of discrete hypothesis bins (default 2)
            true_bin: Ground truth bin index for simulation (default 0)
            max_bin_uncertainty: Misclassification risk threshold for termination
            lambda_weight: Trade-off between path length and uncertainty
        """
        super().__init__(
            q_start, q_goal, gripper_setpoint, meshcat, is_visualizing=False
        )

        # Light region parameters
        self.light_center = light_center
        self.light_half = light_size / 2.0

        # --- TPR/FPR SENSOR MODEL ---
        # Light region: informative sensor
        self.tpr_light = tpr_light
        self.fpr_light = fpr_light
        
        # Dark region: uninformative sensor (coin flip, no information gain)
        self.tpr_dark = 0.5
        self.fpr_dark = 0.5
        
        # --- Bin configuration ---
        self.n_bins = n_bins
        self.true_bin = true_bin  # Ground truth for simulation
        self.max_bin_uncertainty = max_bin_uncertainty
        self.lambda_weight = lambda_weight

    def is_in_light(self, q: tuple) -> bool:
        """Check if the gripper (camera) is in the light region."""
        plant = self.collision_checker.plant
        context = self.collision_checker.context_plant
        plant.SetPositions(context, plant.GetModelInstanceByName("iiwa"), np.array(q))
        camera_body = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))
        X_Cam = plant.EvalBodyPoseInWorld(context, camera_body)
        delta = np.abs(X_Cam.translation() - self.light_center)
        return np.all(delta <= self.light_half)

    def get_sensor_model(self, q: tuple) -> tuple[float, float]:
        """
        Get the sensor model (TPR, FPR) based on robot configuration.
        
        Args:
            q: Joint configuration
            
        Returns:
            (tpr, fpr): True Positive Rate and False Positive Rate
        """
        if self.is_in_light(q):
            return self.tpr_light, self.fpr_light
        else:
            return self.tpr_dark, self.fpr_dark  # Uninformative

    # =========================================================================
    # RRBTProblem Protocol Implementation
    # =========================================================================
    
    def get_initial_belief(self) -> np.ndarray:
        """Return uniform prior over bins (maximum uncertainty)."""
        return np.ones(self.n_bins) / self.n_bins
    
    def propagate_belief(self, parent_node, q_target: tuple) -> dict | None:
        """
        Propagate belief from parent_node to q_target using discrete Bayes filter.
        
        In light region: Uses expected_posterior_all_bins() to compute
                        expected belief after measuring all bins
        In dark region: Belief unchanged (uninformative measurements)
        
        Returns dict with 'belief', 'cost', 'path_length' or None if fails.
        """
        belief_parent = parent_node.belief

        # Get sensor model (TPR, FPR) based on robot location
        tpr, fpr = self.get_sensor_model(q_target)

        # Update belief based on light/dark region
        if self.is_in_light(q_target):
            # In light: measure all bins (expected posterior for planning)
            belief_new = expected_posterior_all_bins(
                belief_parent, tpr, fpr, 
                assumed_bin=self.true_bin
            )
        else:
            # In dark: uninformative measurements, belief unchanged
            belief_new = belief_parent.copy()
        
        # Compute path length increment
        dist_increment = self.cspace.distance(parent_node.value, q_target)
        path_length_new = parent_node.path_length + dist_increment
        
        # Compute combined cost
        cost_new = self.compute_cost(path_length_new, belief_new)

        return {
            "belief": belief_new, 
            "cost": cost_new,
            "path_length": path_length_new,
        }
    
    def compute_cost(self, path_length: float, belief: np.ndarray) -> float:
        """
        Compute combined cost = path_length + lambda * misclassification_risk
        
        Where misclassification_risk = 1 - max(belief)
        """
        misclass_risk = calculate_misclassification_risk(belief)
        return path_length + self.lambda_weight * misclass_risk
    
    def node_reaches_goal(self, node, tol=None) -> bool:
        """
        Check if belief uncertainty is below threshold.
        
        For belief-space planning, goal is achieved when misclassification
        risk is low enough, NOT when robot reaches a geometric position.
        """
        misclass_risk = calculate_misclassification_risk(node.belief)
        return misclass_risk <= self.max_bin_uncertainty
    
    def sample_goal_from_belief(self, node) -> np.ndarray:
        """
        Sample goal configuration from belief using MAP estimate.
        
        Returns the goal configuration associated with the most likely bin.
        For now, returns the problem's goal configuration.
        """
        # Get the MAP estimate (bin with highest probability)
        map_bin = np.argmax(node.belief)
        
        # For now, return the true goal configuration
        # In a full implementation, this would return the bin-specific goal
        return np.array(self.goal)
    
    # Legacy alias for backward compatibility
    def cost_function(self, path_length: float, belief: np.ndarray) -> float:
        """Alias for compute_cost (backward compatibility)."""
        return self.compute_cost(path_length, belief)


class IiwaProblemMustardPositionBelief(IiwaProblem):
    """
    Belief-space planning problem using Kalman filter for 2D (X-Y) position estimation.
    
    The belief is represented as a 2x2 covariance matrix over the mustard bottle's
    X-Y position. The Z position is fixed (bottle sits on bin floor). The sensor
    model uses different noise levels in light vs dark regions.
    
    Implements the RRBTProblem protocol for use with RRBT_Tree and RRBT_tools.
    """
    
    def __init__(
        self,
        q_start,
        q_goal,
        gripper_setpoint,
        meshcat,
        light_center,
        light_size,
        scale_R_light,
        scale_R_dark,
        initial_uncertainty: float = 0.1,
        max_uncertainty: float = 0.001,
        lambda_weight: float = 1.0,
        estimated_position: np.ndarray = None,
    ):
        """
        Args:
            q_start: Starting joint configuration
            q_goal: Goal joint configuration (not used for termination in belief planning)
            gripper_setpoint: Gripper opening width
            meshcat: Meshcat visualizer instance
            light_center: Center of the light region [x, y, z]
            light_size: Size of the light region [dx, dy, dz]
            scale_R_light: Measurement noise scale in light region
            scale_R_dark: Measurement noise scale in dark region
            initial_uncertainty: Initial diagonal value for 2x2 covariance
            max_uncertainty: Termination threshold for trace(Σ)
            lambda_weight: Trade-off between path length and uncertainty
            estimated_position: 3D position estimate from ICP (X-Y uncertain, Z fixed)
        """
        super().__init__(
            q_start, q_goal, gripper_setpoint, meshcat, is_visualizing=False
        )

        self.light_center = light_center
        self.light_half = light_size / 2.0

        # --- 2D Kalman Filter Parameters (X-Y only) ---
        # State Transition (A=I): Target is static
        self.A = np.eye(2)
        
        # Observation Matrix (C=I): Full X-Y position observation
        self.C = np.eye(2)
        
        # Measurement noise matrices (2x2, X-Y only)
        self.R_light = np.eye(2) * scale_R_light
        self.R_dark = np.eye(2) * scale_R_dark
        
        # Uncertainty parameters
        self.initial_uncertainty = initial_uncertainty
        self.max_uncertainty = max_uncertainty
        self.lambda_weight = lambda_weight
        
        # 3D position estimate from ICP (X-Y uncertain, Z fixed)
        self.estimated_position = estimated_position if estimated_position is not None else np.zeros(3)

    def is_in_light(self, q: tuple) -> bool:
        """Check if the gripper (camera) is in the light region."""
        plant = self.collision_checker.plant
        context = self.collision_checker.context_plant
        plant.SetPositions(context, plant.GetModelInstanceByName("iiwa"), np.array(q))
        camera_body = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))
        X_Cam = plant.EvalBodyPoseInWorld(context, camera_body)
        delta = np.abs(X_Cam.translation() - self.light_center)
        return np.all(delta <= self.light_half)

    def get_measurement_noise(self, q: tuple) -> np.ndarray:
        """
        Get measurement noise matrix R based on robot location.
        
        Returns:
            R: 2x2 measurement noise matrix (X-Y only)
        """
        if self.is_in_light(q):
            return self.R_light
        else:
            return self.R_dark

    # =========================================================================
    # RRBTProblem Protocol Implementation
    # =========================================================================
    
    def get_initial_belief(self) -> np.ndarray:
        """Return initial 2x2 covariance matrix (high uncertainty in X-Y)."""
        return np.eye(2) * self.initial_uncertainty
    
    def propagate_belief(self, parent_node, q_target: tuple) -> dict | None:
        """
        Propagate belief from parent_node to q_target using 2D Kalman filter.
        
        The belief is a 2x2 covariance matrix over the mustard X-Y position.
        The mean (estimated_position) stays fixed; only covariance updates.
        
        Returns dict with 'belief', 'cost', 'path_length' or None if fails.
        """
        sigma_parent = parent_node.belief  # 2x2 covariance (X-Y)

        # Get measurement noise based on light/dark region
        R = self.get_measurement_noise(q_target)
        C = self.C  # Identity 2x2

        # Kalman filter covariance update
        # (Mean stays fixed at ICP estimate, only covariance shrinks)
        S = C @ sigma_parent @ C.T + R
        try:
            K = sigma_parent @ C.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return None

        sigma_new = (np.eye(2) - K @ C) @ sigma_parent
        
        # Compute path length increment
        dist_increment = self.cspace.distance(parent_node.value, q_target)
        path_length_new = parent_node.path_length + dist_increment
        
        # Compute combined cost
        cost_new = self.compute_cost(path_length_new, sigma_new)

        return {
            "belief": sigma_new, 
            "cost": cost_new,
            "path_length": path_length_new,
        }
    
    def compute_cost(self, path_length: float, belief: np.ndarray) -> float:
        """
        Compute combined cost = path_length + lambda * trace(Σ)
        
        Where Σ is the 2x2 position covariance (X-Y only).
        """
        trace_sigma = np.trace(belief)
        return path_length + self.lambda_weight * trace_sigma
    
    def node_reaches_goal(self, node, tol=None) -> bool:
        """
        Check if belief uncertainty is below threshold.
        
        Goal is achieved when trace(Σ) < max_uncertainty.
        """
        uncertainty = np.trace(node.belief)
        return uncertainty <= self.max_uncertainty
    
    def sample_goal_from_belief(self, node) -> np.ndarray:
        """
        Return the estimated 3D position from belief.
        
        Since the mean is fixed at the ICP estimate and we only track
        covariance, we return the estimated_position directly.
        """
        return self.estimated_position.copy()
