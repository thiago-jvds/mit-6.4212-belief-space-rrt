import numpy as np
from pydrake.all import DiagramBuilder, Meshcat
from manipulation.station import MakeHardwareStation, LoadScenario, AddPointClouds

from manipulation.exercises.trajectories.rrt_planner.robot import (
    ConfigurationSpace,
    Range,
)
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import Problem


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


class IiwaProblemBelief(IiwaProblem):
    """
    Belief-space planning problem using a discrete 3-bin Bayes filter.
    
    The sensor model uses TPR (True Positive Rate) and FPR (False Positive Rate)
    instead of Gaussian noise covariance matrices.
    
    In light region: Informative sensor (TPR=0.8, FPR=0.15 by default)
    In dark region: Uninformative sensor (TPR=0.5, FPR=0.5 - coin flip)
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
        n_buckets: int = 3,
        true_bucket: int = 0,
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
            n_buckets: Number of discrete hypothesis buckets (default 3)
            true_bucket: Ground truth bucket index for simulation (default 0)
        """
        super().__init__(
            q_start, q_goal, gripper_setpoint, meshcat, is_visualizing=True
        )

        # Light region parameters
        self.light_center = light_center
        self.light_half = light_size / 2.0

        # --- TPR/FPR SENSOR MODEL (replaces Kalman R matrices) ---
        # Light region: informative sensor
        self.tpr_light = tpr_light
        self.fpr_light = fpr_light
        
        # Dark region: uninformative sensor (coin flip, no information gain)
        self.tpr_dark = 0.5
        self.fpr_dark = 0.5
        
        # --- BUCKET CONFIGURATION ---
        self.n_buckets = n_buckets
        self.true_bucket = true_bucket  # Ground truth for simulation

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
