import yaml
import numpy as np
from pathlib import Path
from types import SimpleNamespace


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)


def load_config(filename="config.yaml"):
    """
    Loads the YAML config and converts lists to numpy arrays where needed.
    """
    # Find the file relative to the project root
    # Assuming this script is in src/utils/
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert specific lists to numpy arrays for convenience
    if "simulation" in config:
        config["simulation"]["bin_light_center"] = np.array(
            config["simulation"]["bin_light_center"]
        )
        config["simulation"]["bin_light_size"] = np.array(
            config["simulation"]["bin_light_size"]
        )
        config["simulation"]["mustard_position_light_center"] = np.array(
            config["simulation"]["mustard_position_light_center"]
        )
        config["simulation"]["mustard_position_light_size"] = np.array(
            config["simulation"]["mustard_position_light_size"]
        )
        config["simulation"]["q_home"] = np.array(config["simulation"]["q_home"])
        
        # Parse tf_goal (task-space goal) if present
        if "tf_goal" in config["simulation"]:
            tf_goal = config["simulation"]["tf_goal"]
            tf_goal["translation"] = np.array(tf_goal["translation"])
            tf_goal["rpy"] = np.array(tf_goal["rpy"])
        
        # Legacy support: convert q_goal if present (optional fallback)
        if "q_goal" in config["simulation"]:
            config["simulation"]["q_goal"] = np.array(config["simulation"]["q_goal"])

    # Convert camera pose and chart position to numpy arrays if present
    if "visualization" in config:
        if "camera_position" in config["visualization"]:
            config["visualization"]["camera_position"] = np.array(
                config["visualization"]["camera_position"]
            )
        if "camera_target" in config["visualization"]:
            config["visualization"]["camera_target"] = np.array(
                config["visualization"]["camera_target"]
            )
        if "chart_position" in config["visualization"]:
            config["visualization"]["chart_position"] = np.array(
                config["visualization"]["chart_position"]
            )

    return NestedNamespace(config)
