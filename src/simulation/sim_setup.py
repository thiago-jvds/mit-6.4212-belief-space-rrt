import numpy as np
from src.simulation.simulation_tools import IiwaProblemBelief
from typing import List
from pydrake.all import Sphere, Rgba, RigidTransform, RotationMatrix, Ellipsoid


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
            prev_q = path[i - 1]
            plant.SetPositions(context, iiwa, np.array(prev_q))
            prev_pos = plant.EvalBodyPoseInWorld(context, wsg_body).translation()

            # Draw line segment (as thin cylinder or use SetLine)
            line_name = f"path_line_{i}"
            meshcat.SetLine(
                line_name,
                np.column_stack([prev_pos, gripper_pos]),
                rgba=Rgba(1, 1, 0, 1),
            )  # Yellow path

    # Print path statistics
    print(f"\n📊 Path Statistics:")
    print(f"   Total waypoints: {len(path)}")
    print(f"   Final uncertainty: {uncertainty:.6f}")
    print(
        f"   Max uncertainty threshold: {problem.collision_checker.MAX_UNCERTAINTY if hasattr(problem.collision_checker, 'MAX_UNCERTAINTY') else 'N/A'}"
    )


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
    true_q = np.array(path[0])  # The physical robot
    est_q = np.array(path[0])  # The robot's belief
    est_sigma = np.eye(7) * 1e-6  # Initial uncertainty

    # Animation Timing
    dt = 0.05
    current_time = 0.0

    # --- VISUALIZATION LOOP ---
    for i in range(len(path) - 1):
        target_q = np.array(path[i + 1])
        steps = 10
        step_vec = (target_q - est_q) / steps

        for s in range(steps):
            # A. Update Time (Crucial for Recording!)
            current_time += dt / steps

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
    print(f"  Final Position Error: {dist_error * 100:.2f} cm")


def visualize_target_uncertainty(problem, path, meshcat, true_goal_config=None):
    """
    Visualizes the robot path and the SHRINKING UNCERTAINTY
    around the estimated q_goal.
    """
    print("\n   > Visualizing Goal Uncertainty Evolution...")
    meshcat.DeleteRecording()
    meshcat.StartRecording()

    sim_wrapper = problem.collision_checker
    plant = sim_wrapper.plant

    # Context for Visualization (Robot Motion)
    viz_context = sim_wrapper.context_plant

    # --- FIX: Create a Separate Context for Math (Goal FK) ---
    # We use this to calculate X_Est without moving the actual robot visual
    math_context = plant.CreateDefaultContext()

    iiwa_model = plant.GetModelInstanceByName("iiwa")
    wsg_body = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))

    # ---------------------------------------------------------
    # 1. CALCULATE TRUTH (The Mean we are uncertain about)
    # ---------------------------------------------------------
    if true_goal_config is not None:
        true_q_goal = np.array(true_goal_config) # Use explicit truth
    else:
        true_q_goal = np.array(problem.goal)     # Fallback

    # Calculate True 3D Position using the MATH context
    plant.SetPositions(math_context, iiwa_model, true_q_goal)
    X_TrueGoal = plant.EvalBodyPoseInWorld(math_context, wsg_body)

    # ---------------------------------------------------------
    # 2. INITIALIZE BELIEF
    # ---------------------------------------------------------
    est_q_goal = true_q_goal + np.random.uniform(-0.5, 0.5, size=7)
    est_sigma = np.eye(7) * 1.0

    # Setup Meshcat Objects
    meshcat.SetObject("uncertainty/ellipsoid", Sphere(1.0), Rgba(1, 0.2, 0, 0.3))
    meshcat.SetObject("uncertainty/estimate", Sphere(0.03), Rgba(0, 0, 1, 1.0))
    meshcat.SetObject("uncertainty/truth", Sphere(0.03), Rgba(0, 1, 0, 0.5))

    meshcat.SetTransform("uncertainty/truth", X_TrueGoal)

    # Simulation Params
    dt = 0.05
    current_time = 0.0
    vis_interval = 2
    steps = 10

    for i in range(len(path) - 1):
        q_curr = np.array(path[i])
        q_next = np.array(path[i + 1])
        step_vec = (q_next - q_curr) / steps

        for s in range(steps):
            current_time += dt / steps
            sim_wrapper.context_diagram.SetTime(current_time)

            # A. Robot Move
            q_robot = q_curr + step_vec * s

            # B. Belief Update (Kalman Filter)
            A, _, C, R = problem.get_dynamics_and_observation(q_robot)
            sigma_pred = est_sigma
            S = C @ sigma_pred @ C.T + R
            K = sigma_pred @ C.T @ np.linalg.inv(S)
            est_sigma = (np.eye(7) - K @ C) @ sigma_pred

            # C. Simulate Convergence
            info_gain = np.trace(sigma_pred) - np.trace(est_sigma)
            if info_gain > 1e-6:
                alpha = 0.1
                est_q_goal = est_q_goal * (1 - alpha) + true_q_goal * alpha

            # D. Visualization
            if s % vis_interval == 0:
                # 1. Update Robot Visuals (Uses Main/Viz Context)
                sim_wrapper.DrawStation(q_robot, 0.1)

                # 2. Calculate Estimate Position (Uses Math Context)
                # This does NOT disturb the robot's position in the visualizer
                plant.SetPositions(math_context, iiwa_model, est_q_goal)
                X_Est = plant.EvalBodyPoseInWorld(math_context, wsg_body)

                # 3. Update Transforms
                meshcat.SetTransform("uncertainty/estimate", X_Est)

                uncertainty_radius = np.trace(est_sigma) * 0.2
                uncertainty_radius = max(0.05, uncertainty_radius)

                meshcat.SetObject(
                    "uncertainty/ellipsoid",
                    Sphere(uncertainty_radius),
                    Rgba(1, 0.5, 0, 0.3),
                )
                meshcat.SetTransform("uncertainty/ellipsoid", X_Est)

                # 4. Publish (This sends the state of Viz Context, which is at q_robot)
                sim_wrapper.diagram.ForcedPublish(sim_wrapper.context_diagram)

            if s == 0:
                dist_err = np.linalg.norm(est_q_goal - true_q_goal)
                print(
                    f"\rUncertainty: {np.trace(est_sigma):.4f} | Err: {dist_err:.4f}",
                    end="",
                )

    meshcat.StopRecording()
    meshcat.PublishRecording()
    print("\n✓ Visualization Complete.")


def visualize_belief_tree(rrbt_tree, problem, meshcat, iteration):
    """
    Visualizes the RRBT tree in Meshcat for debugging.
    Shows only key structural nodes: root, leaves, and junctions.

    Args:
        rrbt_tree: RRBT_Tree instance containing the belief tree
        problem: IiwaProblemBelief instance (for FK and light region check)
        meshcat: Meshcat instance
        iteration: Current iteration number (for display)
    """
    # Clear previous tree visualization
    meshcat.Delete("rrbt_tree")

    plant = problem.collision_checker.plant
    context = problem.collision_checker.context_plant
    iiwa = plant.GetModelInstanceByName("iiwa")
    wsg_body = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg"))

    def get_gripper_pos(q):
        """Helper to compute gripper position via forward kinematics."""
        plant.SetPositions(context, iiwa, np.array(q))
        X_Gripper = plant.EvalBodyPoseInWorld(context, wsg_body)
        return X_Gripper.translation()

    # DEBUG: Sanity check parent pointers
    if iteration == 1:
        print("\n--- DEBUG TREE STRUCTURE ---")
        root = rrbt_tree.nodes[0]
        print(f"Root ID: {id(root)}")

        # check how many nodes point to root
        children_of_root = [n for n in rrbt_tree.nodes if n.parent == root]
        print(f"Nodes pointing to root: {len(children_of_root)}")

        # Check a random child of root
        if children_of_root:
            child = children_of_root[0]
            print(f"Child ID: {id(child)}, Child.parent ID: {id(child.parent)}")
            print(f"Parent match: {child.parent is root}")

            # Check grandchildren
            grandchild = [n for n in rrbt_tree.nodes if n.parent == child]
            print(f"Grandchildren of first child: {len(grandchild)}")
            if len(grandchild) > 0:
                print(
                    f"First child IS Intermediate (or Junction). is_key_node should be False."
                )
            else:
                print(f"First child IS Leaf. is_key_node should be True.")

        # Check for orphaned parents
        orphans = 0
        for n in rrbt_tree.nodes:
            if n.parent and n.parent not in rrbt_tree.nodes:
                orphans += 1
        print(f"Nodes with parents NOT in tree list: {orphans}")
        print("----------------------------\n")

    def is_key_node(node, all_nodes):
        """Check if node is structurally important: root, leaf, or junction."""
        is_root = node.parent is None

        # Always count children by traversing all nodes to ensure accuracy
        # (The .children attribute might be out of sync or uninitialized)
        num_children = sum(1 for n in all_nodes if n.parent == node)

        is_leaf = num_children == 0
        is_junction = num_children >= 2

        return is_root or is_leaf or is_junction

    # Collect key nodes and their positions
    key_nodes = [n for n in rrbt_tree.nodes if is_key_node(n, rrbt_tree.nodes)]

    # DEBUG: Count types
    n_roots = sum(1 for n in key_nodes if n.parent is None)
    n_leaves = sum(
        1 for n in key_nodes if sum(1 for x in rrbt_tree.nodes if x.parent == n) == 0
    )
    n_junctions = len(key_nodes) - n_roots - n_leaves

    print(
        f"\r   > RRBT Iter {iteration}: {len(rrbt_tree.nodes)} nodes -> {len(key_nodes)} keys "
        f"(R:{n_roots}, L:{n_leaves}, J:{n_junctions})",
        end="",
        flush=True,
    )

    # Draw key nodes as spheres
    for i, node in enumerate(key_nodes):
        pos = get_gripper_pos(node.value)
        in_light = problem.is_in_light(node.value)
        uncertainty = node.cost

        # Color based on light/dark region (green for light, red for dark)
        # Alpha varies with uncertainty (more uncertain = more transparent)
        alpha = max(0.3, min(0.9, 1.0 - uncertainty * 5))
        if in_light:
            color = Rgba(0, 0.8, 0.2, alpha)  # Green for light
        else:
            color = Rgba(0.9, 0.2, 0.1, alpha)  # Red for dark

        # Sphere radius proportional to uncertainty (clamped for visibility)
        radius = max(0.005, min(0.03, uncertainty * 2))

        # Ensure root is slightly more visible
        if node.parent is None:
            radius = max(radius, 0.02)  # Root should be at least 2cm

        meshcat.SetObject(f"rrbt_tree/node_{i}", Sphere(radius), color)
        meshcat.SetTransform(
            f"rrbt_tree/node_{i}", RigidTransform(RotationMatrix(), pos)
        )

    # Draw edges: trace from each leaf/junction back to previous key node
    edge_idx = 0
    for node in key_nodes:
        if node.parent is None:
            continue

        # Walk up to find the previous key node
        end_pos = get_gripper_pos(node.value)
        curr = node.parent
        while curr is not None and not is_key_node(curr, rrbt_tree.nodes):
            curr = curr.parent

        if curr is not None:
            start_pos = get_gripper_pos(curr.value)
            in_light = problem.is_in_light(node.value)
            edge_color = (
                Rgba(0.5, 1.0, 0.5, 0.7) if in_light else Rgba(1.0, 0.5, 0.5, 0.7)
            )
            meshcat.SetLine(
                f"rrbt_tree/edge_{edge_idx}",
                np.column_stack([start_pos, end_pos]),
                rgba=edge_color,
            )
            edge_idx += 1

    # (Print moved to top)
