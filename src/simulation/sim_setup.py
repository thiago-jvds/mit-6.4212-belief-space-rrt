import numpy as np
from src.simulation.simulation_tools import IiwaProblemBelief
from typing import List
from pydrake.all import Sphere, Rgba, RigidTransform, RotationMatrix


def simulate_execution(problem: IiwaProblemBelief, path: List[List[float]], trials=20):
    """
    Simulates the robot trying to follow 'path' with noise.

    Noise is defined as drift from the actual estimated position by
    """
    success_count = 0
    print(f"   > Simulating execution ({trials} trials)...")

    for t in range(trials):
        true_q = np.array(problem.start)
        est_q = np.array(problem.start)
        est_sigma = np.eye(7) * 1e-6
        crashed = False

        for i in range(len(path) - 1):
            target_q = np.array(path[i + 1])

            # --- PHYSICS (Drift) ---
            cmd = target_q - est_q
            drift = np.random.multivariate_normal(np.zeros(7), problem.Q)
            true_q = true_q + cmd + drift

            if problem.collide(true_q):
                crashed = True
                break

            # --- SENSING (Vision) ---
            A, Q, C, R = problem.get_dynamics_and_observation(true_q)
            noise_sensor = np.random.multivariate_normal(np.zeros(7), R)
            z = true_q + noise_sensor

            # --- ESTIMATION (Kalman Filter) ---
            # Predict
            est_q_pred = est_q + cmd
            est_sigma_pred = A @ est_sigma @ A.T + Q
            # Update
            S = C @ est_sigma_pred @ C.T + R
            K = est_sigma_pred @ C.T @ np.linalg.inv(S)
            est_q = est_q_pred + K @ (z - est_q_pred)  # Correction
            est_sigma = (np.eye(7) - K @ C) @ est_sigma_pred

        if not crashed:
            if np.linalg.norm(true_q - np.array(problem.goal)) < 0.2:
                success_count += 1

    return (success_count / trials) * 100.0

def visualize_belief_path(problem, path, meshcat, belief_nodes=None):
    """
    Visualizes the planned path with uncertainty indicators.
    
    Args:
        problem: IiwaProblemBelief instance
        path: List of configurations [q_start, ..., q_goal]
        meshcat: Meshcat instance
        belief_nodes: Optional list of BeliefNodes (if available)
    """
    plant = problem.collision_checker.plant
    context = problem.collision_checker.context_plant
    iiwa = plant.GetModelInstanceByName("iiwa")
    wsg_body = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))
    
    # Simulate belief propagation if nodes not provided
    sigma = np.eye(7) * 1e-6
    
    for i, q in enumerate(path):
        # Get gripper position via forward kinematics
        plant.SetPositions(context, iiwa, np.array(q))
        X_Gripper = plant.EvalBodyPoseInWorld(context, wsg_body)
        gripper_pos = X_Gripper.translation()
        
        # Propagate belief to get uncertainty at this waypoint
        A, Q, C, R = problem.get_dynamics_and_observation(q)
        sigma_pred = A @ sigma @ A.T + Q
        S = C @ sigma_pred @ C.T + R
        K = sigma_pred @ C.T @ np.linalg.inv(S)
        sigma = (np.eye(7) - K @ C) @ sigma_pred
        uncertainty = np.trace(sigma)
        
        # Check if in light or dark
        in_light = problem.is_in_light(q)
        
        # Color: Green if in light, Red if in dark
        color = Rgba(0, 1, 0, 0.5) if in_light else Rgba(1, 0, 0, 0.5)
        
        # Sphere radius proportional to uncertainty (scaled for visibility)
        radius = max(0.01, min(0.1, uncertainty * 10))  # Clamp between 1cm and 10cm
        
        # Draw sphere at gripper position
        sphere_name = f"path_uncertainty_{i}"
        meshcat.SetObject(sphere_name, Sphere(radius), color)
        meshcat.SetTransform(sphere_name, RigidTransform(RotationMatrix(), gripper_pos))
        
        # Draw path line segment
        if i > 0:
            prev_q = path[i-1]
            plant.SetPositions(context, iiwa, np.array(prev_q))
            prev_pos = plant.EvalBodyPoseInWorld(context, wsg_body).translation()
            
            # Draw line segment (as thin cylinder or use SetLine)
            line_name = f"path_line_{i}"
            meshcat.SetLine(line_name, np.column_stack([prev_pos, gripper_pos]), 
                           rgba=Rgba(1, 1, 0, 1))  # Yellow path
    
    # Print path statistics
    print(f"\n📊 Path Statistics:")
    print(f"   Total waypoints: {len(path)}")
    print(f"   Final uncertainty: {uncertainty:.6f}")
    print(f"   Max uncertainty threshold: {problem.collision_checker.MAX_UNCERTAINTY if hasattr(problem.collision_checker, 'MAX_UNCERTAINTY') else 'N/A'}")

def visualize_noisy_execution(problem, path, meshcat):
    """
    Executes path with Simulated Noise and uses problem.collision_checker.DrawStation
    to update the visualizer reliably.
    """
    print("\n--- STARTING NOISY SIMULATION & RECORDING ---")
    
    # 1. Setup Recording
    meshcat.DeleteRecording()
    meshcat.StartRecording()
    
    # 2. Access the Internal Simulator from the Problem
    # This ensures we use the exact same context that visualize_path uses
    sim_wrapper = problem.collision_checker
    
    # 3. Initialization
    true_q = np.array(path[0])      # The physical robot
    est_q = np.array(path[0])       # The robot's belief
    est_sigma = np.eye(7) * 1e-6    # Initial uncertainty
    
    # Animation Timing
    dt = 0.05 
    current_time = 0.0
    
    # --- VISUALIZATION LOOP ---
    for i in range(len(path) - 1):
        target_q = np.array(path[i+1])
        steps = 10 
        step_vec = (target_q - est_q) / steps 
        
        for s in range(steps):
            # A. Update Time (Crucial for Recording!)
            current_time += (dt / steps)
            
            # Update the context time inside the problem's internal simulator
            sim_wrapper.context_diagram.SetTime(current_time)

            # B. Physics (Process Noise)
            process_noise = np.random.multivariate_normal(np.zeros(7), problem.Q * 0.1)
            true_q = true_q + step_vec + process_noise
            
            # Use the existing method from IiwaProblem
            sim_wrapper.DrawStation(true_q, 0.1) 
            
            # Collision Check
            if problem.collide(true_q):
                print(f"\n💥 CRASH DETECTED at step {i}/{len(path)}!")
                meshcat.StopRecording()
                meshcat.PublishRecording()
                return 
            
            # D. Sensing & Estimation (Kalman Filter)
            A, Q, C, R = problem.get_dynamics_and_observation(true_q)
            meas_noise = np.random.multivariate_normal(np.zeros(7), R)
            z_meas = true_q + meas_noise
            
            est_q_pred = est_q + step_vec
            est_sigma_pred = A @ est_sigma @ A.T + Q
            S = C @ est_sigma_pred @ C.T + R
            K = est_sigma_pred @ C.T @ np.linalg.inv(S)
            est_q = est_q_pred + K @ (z_meas - C @ est_q_pred)
            est_sigma = (np.eye(7) - K @ C) @ est_sigma_pred
            
            # Print Status
            in_light = problem.is_in_light(true_q)
            status = "LIGHT" if in_light else "DARK "
            print(f"\rRecording... T={current_time:.2f}s | {status}", end="")

    # 3. Finalize Recording
    meshcat.StopRecording()
    meshcat.PublishRecording()
    
    print("\n\n✓ RECORDING COMPLETE!")
    
    dist_error = np.linalg.norm(true_q - np.array(path[-1]))
    print(f"  Final Position Error: {dist_error*100:.2f} cm")