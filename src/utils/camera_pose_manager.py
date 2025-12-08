"""
Camera Pose Manager for Meshcat

This module provides utilities to save and restore Meshcat camera poses.
The camera pose is saved in config.yaml and restored when the simulation is re-run.

Usage:
    1. Open Meshcat with ?tracked_camera=on parameter in the URL
    2. Move the camera to your desired position
    3. The camera pose can be polled and saved using save_camera_pose()
    4. On next run, the camera pose will be automatically restored using load_camera_pose()
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from pydrake.geometry import Meshcat
from pydrake.math import RigidTransform


def get_camera_pose(meshcat: Meshcat) -> Optional[RigidTransform]:
    """
    Get the current camera pose from Meshcat.
    
    Note: This requires the Meshcat browser to be opened with ?tracked_camera=on
    in the URL. For example: http://localhost:7000/?tracked_camera=on
    
    Args:
        meshcat: Meshcat instance
        
    Returns:
        RigidTransform representing the camera pose, or None if not available
    """
    try:
        camera_pose = meshcat.GetTrackedCameraPose()
        return camera_pose
    except Exception as e:
        print(f"Warning: Could not get camera pose: {e}")
        return None


def save_camera_pose(
    meshcat: Meshcat,
    filepath: Optional[Path] = None,
    camera_position: Optional[np.ndarray] = None,
    target_position: Optional[np.ndarray] = None,
) -> bool:
    """
    Save the current camera pose to config.yaml.
    
    This function can work in two modes:
    1. If camera_position and target_position are provided, save those directly
    2. Otherwise, try to get the camera pose from Meshcat using GetTrackedCameraPose()
    
    Args:
        meshcat: Meshcat instance
        filepath: Path to config.yaml file. If None, uses default location.
        camera_position: Optional camera position (3D array). If None, tries to get from Meshcat.
        target_position: Optional target position (3D array). If None, tries to get from Meshcat.
        
    Returns:
        True if saved successfully, False otherwise
    """
    if filepath is None:
        project_root = Path(__file__).resolve().parents[2]
        filepath = project_root / "config" / "config.yaml"
    
    # Try to get camera pose from Meshcat if not provided
    if camera_position is None or target_position is None:
        camera_pose = get_camera_pose(meshcat)
        if camera_pose is None:
            print("Warning: Could not get camera pose from Meshcat.")
            print("  Make sure you opened Meshcat with ?tracked_camera=on in the URL")
            print("  Example: http://localhost:7000/?tracked_camera=on")
            return False
        
        # Extract camera position
        camera_position = camera_pose.translation()
        
        # For target, we'll use a point 1 meter in front of the camera
        # (along the camera's forward direction)
        # The camera's forward direction is typically -Z in camera frame
        # We need to transform this to world frame
        R = camera_pose.rotation().matrix()
        # Camera forward is typically -Z, so world forward = -R[:, 2]
        forward_direction = -R[:, 2]
        target_position = camera_position + forward_direction * 1.0
    
    # Ensure numpy arrays
    camera_position = np.array(camera_position).flatten()
    target_position = np.array(target_position).flatten()
    
    if len(camera_position) != 3 or len(target_position) != 3:
        print(f"Error: Invalid camera or target position dimensions")
        return False
    
    try:
        # Load existing config
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
        
        # Ensure visualization section exists
        if "visualization" not in config:
            config["visualization"] = {}
        
        # Update camera pose values
        config["visualization"]["camera_position"] = camera_position.tolist()
        config["visualization"]["camera_target"] = target_position.tolist()
        
        # Save updated config
        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Camera pose saved to {filepath}")
        print(f"  Camera position: {camera_position}")
        print(f"  Target position: {target_position}")
        return True
    except Exception as e:
        print(f"Error saving camera pose: {e}")
        return False


def load_camera_pose(
    filepath: Optional[Path] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load camera pose from config.yaml.
    
    Args:
        filepath: Path to config.yaml file. If None, uses default location.
        
    Returns:
        Tuple of (camera_position, target_position) as numpy arrays, or None if not found
    """
    if filepath is None:
        project_root = Path(__file__).resolve().parents[2]
        filepath = project_root / "config" / "config.yaml"
    
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
        
        if "visualization" not in config:
            return None
        
        viz_config = config["visualization"]
        if "camera_position" not in viz_config or "camera_target" not in viz_config:
            return None
        
        camera_position = np.array(viz_config["camera_position"])
        target_position = np.array(viz_config["camera_target"])
        
        return (camera_position, target_position)
    except Exception as e:
        print(f"Error loading camera pose: {e}")
        return None


def restore_camera_pose(meshcat: Meshcat, filepath: Optional[Path] = None) -> bool:
    """
    Restore the camera pose from config.yaml.
    
    Args:
        meshcat: Meshcat instance
        filepath: Path to config.yaml file. If None, uses default location.
        
    Returns:
        True if restored successfully, False otherwise
    """
    camera_data = load_camera_pose(filepath)
    if camera_data is None:
        return False
    
    camera_position, target_position = camera_data
    
    try:
        meshcat.SetCameraPose(camera_position, target_position)
        print(f"✓ Camera pose restored")
        print(f"  Camera position: {camera_position}")
        print(f"  Target position: {target_position}")
        return True
    except Exception as e:
        print(f"Error restoring camera pose: {e}")
        return False


def poll_and_save_camera_pose(
    meshcat: Meshcat,
    filepath: Optional[Path] = None,
    interval_seconds: float = 1.0,
    max_iterations: Optional[int] = None,
) -> None:
    """
    Continuously poll the camera pose and save it when it changes.
    
    This is useful for interactively saving camera poses as you move the camera
    in the Meshcat browser.
    
    Args:
        meshcat: Meshcat instance
        filepath: Path to config.yaml file. If None, uses default location.
        interval_seconds: How often to check for camera pose changes (in seconds)
        max_iterations: Maximum number of iterations. If None, runs indefinitely.
    """
    import time
    
    # Set default filepath to config.yaml if not provided
    if filepath is None:
        project_root = Path(__file__).resolve().parents[2]
        filepath = project_root / "config" / "config.yaml"
    
    print("\n" + "=" * 60)
    print("Camera Pose Polling Started")
    print("=" * 60)
    print("Move the camera in Meshcat to save its pose.")
    print("Press Ctrl+C to stop polling.")
    print("=" * 60 + "\n")
    
    last_pose = None
    iteration = 0
    pose_available = False
    warning_printed = False
    
    try:
        while max_iterations is None or iteration < max_iterations:
            camera_pose = get_camera_pose(meshcat)
            
            if camera_pose is not None:
                pose_available = True
                if warning_printed:
                    print("  ✓ Camera pose tracking is now active!")
                    warning_printed = False
                
                # Check if pose has changed
                current_translation = camera_pose.translation()
                
                if last_pose is None or not np.allclose(
                    current_translation, last_pose.translation(), atol=1e-3
                ):
                    # Pose has changed, save it
                    # Compute target position from camera pose (1m in front of camera)
                    R = camera_pose.rotation().matrix()
                    forward_direction = -R[:, 2]  # Camera forward is -Z
                    target_pos = camera_pose.translation() + forward_direction * 1.0
                    
                    if save_camera_pose(meshcat, filepath, 
                                       camera_position=camera_pose.translation(),
                                       target_position=target_pos):
                        last_pose = camera_pose
                        print(f"  [Iteration {iteration}] Camera pose updated and saved")
            else:
                if not warning_printed and iteration == 0:
                    print("\n⚠ WARNING: Camera pose tracking is not active!")
                    print("  To enable camera pose tracking:")
                    print(f"    1. Open Meshcat in your browser with ?tracked_camera=on:")
                    print(f"       {meshcat.web_url()}?tracked_camera=on")
                    print("    2. The camera pose will be tracked as you move it")
                    print("    3. Once tracking is active, the pose will be saved automatically\n")
                    warning_printed = True
                elif not warning_printed and iteration > 0 and iteration % 5 == 0:
                    print("  ⚠ Still waiting for camera pose tracking...")
                    print(f"     Make sure you opened: {meshcat.web_url()}?tracked_camera=on")
            
            time.sleep(interval_seconds)
            iteration += 1
            
    except KeyboardInterrupt:
        print("\n\nCamera pose polling stopped by user.")
        
        # Try to save the last known pose when stopping
        if last_pose is not None:
            print("Saving final camera pose...")
            # Compute target position from camera pose (1m in front of camera)
            R = last_pose.rotation().matrix()
            forward_direction = -R[:, 2]  # Camera forward is -Z
            target_pos = last_pose.translation() + forward_direction * 1.0
            
            if save_camera_pose(meshcat, filepath, 
                               camera_position=last_pose.translation(),
                               target_position=target_pos):
                print("  ✓ Final camera pose saved successfully!")
            else:
                print("  ✗ Failed to save final camera pose.")
        elif pose_available:
            # We had a pose at some point, try to get current one
            final_pose = get_camera_pose(meshcat)
            if final_pose is not None:
                print("Saving current camera pose...")
                # Compute target position from camera pose (1m in front of camera)
                R = final_pose.rotation().matrix()
                forward_direction = -R[:, 2]  # Camera forward is -Z
                target_pos = final_pose.translation() + forward_direction * 1.0
                
                if save_camera_pose(meshcat, filepath,
                                   camera_position=final_pose.translation(),
                                   target_position=target_pos):
                    print("  ✓ Final camera pose saved successfully!")
                else:
                    print("  ✗ Failed to save final camera pose.")
            else:
                print("  ⚠ No camera pose available to save.")
                print("  Make sure you opened Meshcat with ?tracked_camera=on in the URL")
        else:
            print("  ⚠ No camera pose was ever retrieved.")
            print("  Make sure you opened Meshcat with ?tracked_camera=on in the URL")
            print(f"  Example: {meshcat.web_url()}?tracked_camera=on")
