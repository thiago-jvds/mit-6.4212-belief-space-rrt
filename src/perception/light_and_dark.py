import numpy as np
from pydrake.all import LeafSystem, MultibodyPlant


class LightDarkRegionSystem(LeafSystem):
    """
    System simulating a ;light and dark regions' for a camera-mounted robot.

    - **Light** region gives full information about the desired object's
    pose (i.e., small, negligible noise).

    - **Dark** region gives no information about the desired object's 
    pose (i.e., infinite noise).
    """
    def __init__(
        self,
        plant: MultibodyPlant,
        light_region_center: np.ndarray,
        light_region_size: np.ndarray,
        sigma_light: float,
        sigma_dark: float,
    ):
        LeafSystem.__init__(self)

        # 1. Internal Plant & Context 
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")

        # 2. Get Bodies
        self._gripper_body = plant.GetBodyByName(
            "body", plant.GetModelInstanceByName("wsg")
        )
        self._target_body = plant.GetBodyByName(
            "base_link_mustard", plant.GetModelInstanceByName("target_object")
        )

        # 3. Store Region Params
        self._center = light_region_center
        self._half_size = light_region_size / 2.0
        self._rng = np.random.default_rng()
        self._sigma_light = sigma_light
        self._sigma_dark = sigma_dark

        # 4. Inputs & Outputs
        self._q_port = self.DeclareVectorInputPort("iiwa.position", 7)

        self.DeclareVectorOutputPort("target_measurement", 3, self.CalcMeasurement)
        self.DeclareVectorOutputPort("in_light_region", 1, self.CalcRegionCheck)

    def _update_internal_context(self, context):
        """Helper to sync internal plant with input 'q'"""
        q = self._q_port.Eval(context)
        
        self._plant.SetPositions(self._plant_context, self._iiwa, q)

    def CalcRegionCheck(self, context, output):
        self._update_internal_context(context)

        X_WG = self._plant.EvalBodyPoseInWorld(
            self._plant_context, self._gripper_body
        )
        p_Gripper = X_WG.translation()

        delta = np.abs(p_Gripper - self._center)
        is_light = np.all(delta <= self._half_size)

        output.SetFromVector([1.0 if is_light else 0.0])

    def CalcMeasurement(self, context, output):
        self._update_internal_context(context)

        X_WTarget = self._plant.EvalBodyPoseInWorld(
            self._plant_context, self._target_body
        )
        true_pos = X_WTarget.translation()

        X_WCamera = self._plant.EvalBodyPoseInWorld(
            self._plant_context, self._gripper_body
        )
        p_Camera = X_WCamera.translation()
        delta = np.abs(p_Camera - self._center)
        is_light = np.all(delta <= self._half_size)

        sigma = self._sigma_light if is_light else self._sigma_dark
        noise = self._rng.normal(0, sigma, size=3)

        output.SetFromVector(true_pos + noise)
