"""
LightDarkRegionSystem - A Drake LeafSystem for light/dark region perception.

This system simulates a 'light and dark regions' sensor model for a camera-mounted robot.
- Light region: Informative sensor with TPR/FPR for binary detection
- Dark region: Uninformative sensor (coin flip, no information gain)

The sensor model uses:
- TPR (True Positive Rate): P(detected | object present)
- FPR (False Positive Rate): P(detected | object absent)
"""

import numpy as np
from pydrake.all import LeafSystem, MultibodyPlant


class BinLightDarkRegionSensorSystem(LeafSystem):
    """
    System simulating 'light and dark regions' for a camera-mounted robot.

    - **Light** region: Informative binary sensor (TPR=0.8, FPR=0.15 by default)
    - **Dark** region: Uninformative sensor (TPR=0.5, FPR=0.5 - coin flip)
    
    Inputs:
        iiwa.position (7D): Current joint configuration
        
    Outputs:
        in_light_region (1D): 1.0 if in light, 0.0 if in dark
        sensor_model (2D): [TPR, FPR] for current region
    """
    
    def __init__(
        self,
        plant: MultibodyPlant,
        light_region_center: np.ndarray,
        light_region_size: np.ndarray,
        tpr_light: float = 0.80,
        fpr_light: float = 0.15,
    ):
        """
        Args:
            plant: Drake MultibodyPlant for kinematics
            light_region_center: Center of the light region [x, y, z]
            light_region_size: Size of the light region [dx, dy, dz]
            tpr_light: True Positive Rate in light region (default 0.80)
            fpr_light: False Positive Rate in light region (default 0.15)
        """
        LeafSystem.__init__(self)

        # 1. Internal Plant & Context 
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")

        # 2. Get Bodies
        self._gripper_body = plant.GetBodyByName(
            "body", plant.GetModelInstanceByName("wsg")
        )
        
        # Try to get target body (may not exist in all scenarios)
        try:
            self._target_body = plant.GetBodyByName(
                "base_link_mustard_bottle", plant.GetModelInstanceByName("mustard_bottle")
            )
        except RuntimeError:
            self._target_body = None

        # 3. Store Region Params
        self._center = light_region_center
        self._half_size = light_region_size / 2.0
        self._rng = np.random.default_rng()
        
        # TPR/FPR sensor model (replaces sigma_light/sigma_dark)
        self._tpr_light = tpr_light
        self._fpr_light = fpr_light
        # Dark region: uninformative (coin flip)
        self._tpr_dark = 0.5
        self._fpr_dark = 0.5

        # 4. Inputs & Outputs
        self._q_port = self.DeclareVectorInputPort("iiwa.position", 7)

        self.DeclareVectorOutputPort("in_light_region", 1, self.CalcRegionCheck)
        self.DeclareVectorOutputPort("sensor_model", 2, self.CalcSensorModel)

    def _update_internal_context(self, context):
        """Helper to sync internal plant with input 'q'"""
        q = self._q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)

    def _is_in_light(self, context) -> bool:
        """Check if gripper is in light region."""
        self._update_internal_context(context)
        
        X_WG = self._plant.EvalBodyPoseInWorld(
            self._plant_context, self._gripper_body
        )
        p_Gripper = X_WG.translation()

        delta = np.abs(p_Gripper - self._center)
        return np.all(delta <= self._half_size)

    def CalcRegionCheck(self, context, output):
        """Output 1.0 if in light region, 0.0 if in dark."""
        is_light = self._is_in_light(context)
        output.SetFromVector([1.0 if is_light else 0.0])

    def CalcSensorModel(self, context, output):
        """
        Output the current sensor model [TPR, FPR] based on light/dark region.
        
        This is the single source of truth for sensor parameters - downstream
        systems like BeliefEstimator should use this rather than having their
        own copies of tpr_light/fpr_light.
        
        Returns:
            sensor_model (2D): [TPR, FPR] for the current region
        """
        is_light = self._is_in_light(context)
        
        if is_light:
            output.SetFromVector([self._tpr_light, self._fpr_light])
        else:
            output.SetFromVector([self._tpr_dark, self._fpr_dark])


class MustardPositionLightDarkRegionSensorSystem(LeafSystem):
    """
    System simulating a light and dark regions' for a mustard bottle position estimation.

    - **Light** region gives full information about the desired object's
    position (i.e., small, negligible noise).

    - **Dark** region gives no information about the desired object's 
    position (i.e., infinite noise).
    """
    def __init__(
        self,
        plant: MultibodyPlant,
        light_region_center: np.ndarray,
        light_region_size: np.ndarray,
        meas_noise_light: float,
        meas_noise_dark: float,
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
            "base_link_mustard", plant.GetModelInstanceByName("mustard")
        )

        # 3. Store Region Params
        self._center = light_region_center
        self._half_size = light_region_size / 2.0
        self._rng = np.random.default_rng()
        self._sigma_light = meas_noise_light
        self._sigma_dark = meas_noise_dark

        # 4. Inputs & Outputs
        self._q_port = self.DeclareVectorInputPort("iiwa.position", 7)

        self.DeclareVectorOutputPort("target_measurement", 3, self.CalcMeasurement)
        self.DeclareVectorOutputPort("in_light_region", 1, self.CalcRegionCheck)
        self.DeclareVectorOutputPort("measurement_variance", 1, self.CalcMeasurementVariance)

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

    def CalcMeasurementVariance(self, context, output):
        """
        Output the current measurement variance (sigma^2) based on light/dark region.
        
        This is the single source of truth for measurement noise - downstream
        systems like BeliefEstimator should use this rather than having their
        own copies of sigma_light/sigma_dark.
        
        Returns:
            variance (1D): sigma^2 for the current region
        """
        self._update_internal_context(context)

        X_WG = self._plant.EvalBodyPoseInWorld(
            self._plant_context, self._gripper_body
        )
        p_Gripper = X_WG.translation()

        delta = np.abs(p_Gripper - self._center)
        is_light = np.all(delta <= self._half_size)

        sigma = self._sigma_light if is_light else self._sigma_dark
        variance = sigma ** 2
        
        output.SetFromVector([variance])