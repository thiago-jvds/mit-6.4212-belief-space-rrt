import yaml
import json
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


def load_rrbt_config(filename="config.yaml"):
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
        config["simulation"]["light_center"] = np.array(
            config["simulation"]["light_center"]
        )
        config["simulation"]["light_size"] = np.array(
            config["simulation"]["light_size"]
        )
        config["simulation"]["q_home"] = np.array(config["simulation"]["q_home"])
        config["simulation"]["q_goal"] = np.array(config["simulation"]["q_goal"])

    # Convert camera pose to numpy arrays if present
    if "visualization" in config:
        if "camera_position" in config["visualization"]:
            config["visualization"]["camera_position"] = np.array(
                config["visualization"]["camera_position"]
            )
        if "camera_target" in config["visualization"]:
            config["visualization"]["camera_target"] = np.array(
                config["visualization"]["camera_target"]
            )

    return NestedNamespace(config)
