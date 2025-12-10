# Main.py Architecture and Design Documentation

This document provides a comprehensive explanation of the high-level design and logical flow of events in `main.py`, the entry point for the MIT 6.4212 Belief-Space RRT planning system.

## Overview

The system implements a **belief-space planning** approach for robot manipulation under uncertainty. It uses a Rapidly-exploring Random Belief Tree (RRBT) algorithm to plan robot motions that actively reduce uncertainty about object location before attempting to grasp.

The core insight is that the robot operates in a "light and dark" environment:
- **Light regions**: Areas where sensors provide high-quality measurements (informative observations)
- **Dark regions**: Areas where sensors provide poor/no information (uninformative observations)

By planning paths through light regions, the robot can actively gather information to reduce uncertainty about the target object.

---

## System Architecture

### Drake-Based Simulation

The system is built on the [Drake](https://drake.mit.edu/) robotics simulation framework. It uses Drake's `DiagramBuilder` pattern to construct a computational graph of interconnected systems (called `LeafSystem`s) that run together in a synchronized simulation loop.

### Detailed Wiring Diagram

The following diagram shows the actual port-level connections in `UnifiedPlanningDiagram` (derived from `system_diagram_full.dot`):

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    UnifiedPlanningDiagram                                            │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              HardwareStation (RobotDiagram)                                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  ┌─────────────────────────────────────┐ │  │
│  │  │MultibodyPlant│  │ SceneGraph  │  │ SimIiwaDriver  │  │ 6× RgbdSensor (camera0-5)          │ │  │
│  │  │   (plant)   │  │             │  │ + WsgController│  │ + MeshcatVisualizers                │ │  │
│  │  └─────────────┘  └─────────────┘  └────────────────┘  └─────────────────────────────────────┘ │  │
│  └────────┬───────────────────────────────────┬───────────────────────────────┬──────────────────┘  │
│           │iiwa.position_measured             │ body_poses                    │ camera images       │
│           │                                   │                               │                     │
│           ▼                                   │                               ▼                     │
│  ┌─────────────────────┐   ┌─────────────────────┐          ┌─────────────────────────────────────┐ │
│  │ BinLightDarkRegion  │   │MustardPositionLight │          │ 6× DepthImageToPointCloud           │ │
│  │    SensorSystem     │   │ DarkRegionSensor    │          │ + ExtractPose (camera pose)         │ │
│  │ (LightDarkPerception│   │  (MustardPosition   │          └──────────────────┬──────────────────┘ │
│  │                     │   │   Perception)       │                             │ point_cloud (x6)   │
│  └──────────┬──────────┘   └──────────┬──────────┘                             │                    │
│             │ sensor_model            │ measurement_variance                   │                    │
│             ▼                         │                                        │                    │
│  ┌─────────────────────┐              │                                        │                    │
│  │ BinBeliefEstimator  │              │                                        │                    │
│  │   System            │              │                                        │                    │
│  │ (Discrete Bayes)    │              │                                        │                    │
│  └─────┬───────┬───────┘              │                                        │                    │
│        │       │                      │                                        │                    │
│        │       │ belief               │                                        │                    │
│        │       │ belief_confident     │                                        │                    │
│        │       │                      │                                        │                    │
│        │       ▼                      │                                        ▼                    │
│        │  ┌────────────────────────────────────────────────────────────────────────────────────┐   │
│        │  │                          MustardPoseEstimatorSystem                                │   │
│        │  │                               (ICP Pose Estimation)                                │   │
│        │  │  Inputs: camera0-5_point_cloud, belief, estimation_trigger                         │   │
│        │  └─────────────────────────────────────────────┬──────────────────────────────────────┘   │
│        │                                                │ estimated_pose                           │
│        │                                                ├──────────────────────────┐               │
│        │                                                │                          │               │
│        │                                                ▼                          │               │
│        │                           ┌────────────────────────────────────────┐      │               │
│        │                           │  MustardPositionBeliefEstimatorSystem  │◀─────┘               │
│        │                           │           (Kalman Filter)              │◀── measurement_      │
│        │                           │                                        │    variance          │
│        │                           └────────────┬───────────────┬───────────┘                      │
│        │                                        │               │                                  │
│        │                           position_mean│               │covariance                        │
│        │                                        │               │                                  │
│        │                                        ▼               │                                  │
│        │                           ┌────────────────────────┐   │                                  │
│        │                           │CovarianceEllipsoidSystem│◀─┘                                  │
│        │                           │    (Visualization)     │                                      │
│        │                           └────────────────────────┘                                      │
│        │                                                                                           │
│        │  belief                                                                                   │
│        ▼                                                                                           │
│  ┌─────────────────────┐                                                                           │
│  │ BeliefBarChartSystem│                                                                           │
│  │   (Visualization)   │                                                                           │
│  └─────────────────────┘                                                                           │
│                                                                                                    │
│                   ┌────────────────────────────────────────────────────────────┐                   │
│                   │                      PlannerSystem                         │                   │
│                   │                    (State Machine)                         │                   │
│                   │                                                            │                   │
│                   │  Inputs:   estimated_mustard_pose  ◀── MustardPoseEstimator│                   │
│                   │            position_covariance     ◀── MustardPositionBeliefEstimator          │
│                   │                                                            │                   │
│                   │  Outputs:  iiwa_position_command   ──▶ HardwareStation     │                   │
│                   │            wsg_position_command    ──▶ HardwareStation     │                   │
│                   └────────────────────────────────────────────────────────────┘                   │
│                                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Abstracted System Diagram

The following diagram abstracts the system into five functional categories while preserving the accurate data flow:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              UnifiedPlanningDiagram                                     │
│                                                                                         │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                         ROBOT SIMULATION LAYER                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      HardwareStation                                        │  │  │
│  │  │  • IIWA arm + WSG gripper (physics, control)                                │  │  │
│  │  │  • 6 RGB-D cameras                                                          │  │  │
│  │  │  • Meshcat visualization                                                    │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────┬─────────────────────────┬─────────────────────────┬───────────────────┘  │
│              │                         │                         │                      │
│              │ joint state             │ body poses              │ camera images        │
│              ▼                         ▼                         ▼                      │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                           PERCEPTION LAYER                                        │  │
│  │                                                                                   │  │
│  │   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────────────┐  │  │
│  │   │  Bin Light/Dark  │   │ Position Light/  │   │  Point Cloud Pipeline        │  │  │
│  │   │  Sensor          │   │ Dark Sensor      │   │  (DepthImage → PointCloud)   │  │  │
│  │   │                  │   │                  │   │                              │  │  │
│  │   │  Outputs:        │   │  Outputs:        │   │  Outputs:                    │  │  │
│  │   │  • sensor_model  │   │  • meas_variance │   │  • 6× point clouds           │  │  │
│  │   │    (TPR, FPR)    │   │                  │   │                              │  │  │
│  │   └────────┬─────────┘   └────────┬─────────┘   └─────────────┬────────────────┘  │  │
│  └────────────│─────────────────────│───────────────────────────│────────────────────┘  │
│               │                      │                           │                      │
│               ▼                      │                           ▼                      │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                           ESTIMATION LAYER                                        │  │
│  │                                                                                   │  │
│  │   ┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐   │  │
│  │   │  Bin Belief      │        │  Pose Estimator  │        │ Position Belief  │   │  │
│  │   │  Estimator       │───────▶│  (ICP)           │───────▶│ Estimator        │   │  │
│  │   │                  │ belief │                  │ pose   │                  │   │  │
│  │   │  Discrete Bayes  │ +trig  │  6 point clouds  │        │  Kalman Filter   │◀──│───│── meas_variance
│  │   │  Filter          │        │  → 6-DOF pose    │        │  (Gaussian)      │   │  │
│  │   │                  │        │                  │        │                  │   │  │
│  │   │  Out: belief,    │        │  Out: pose       │        │  Out: mean, cov  │   │  │
│  │   │       confident  │        │                  │        │                  │   │  │
│  │   └────────┬─────────┘        └────────┬─────────┘        └───────┬──────────┘   │  │
│  └────────────│────────────────────────────│─────────────────────────│──────────────┘  │
│               │                            │                         │                  │
│               │ belief                     │ pose                    │ covariance       │
│               ▼                            ▼                         ▼                  │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                            PLANNING LAYER                                         │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────────┐ │  │
│  │   │                         PlannerSystem                                       │ │  │
│  │   │                                                                             │ │  │
│  │   │   State Machine: IDLE → RRBT_PLAN → RRBT_EXEC → POSE_EST → RRBT2_PLAN →    │ │  │
│  │   │                  RRBT2_EXEC → GRASP_PLAN → GRASP_EXEC → COMPLETE            │ │  │
│  │   │                                                                             │ │  │
│  │   │   Inputs:  estimated_pose, position_covariance                              │ │  │
│  │   │   Outputs: joint commands, gripper commands                                 │ │  │
│  │   └─────────────────────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────┬───────────────────────────────────────┘  │
│                                              │                                          │
│                                              │ joint + gripper commands                 │
│                                              ▼                                          │
│                                    ┌─────────────────┐                                  │
│                                    │ HardwareStation │ (closes the control loop)        │
│                                    └─────────────────┘                                  │
│                                                                                         │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                          VISUALIZATION LAYER                                      │  │
│  │   ┌──────────────────────┐              ┌──────────────────────┐                  │  │
│  │   │  BeliefBarChart      │◀── belief    │  CovarianceEllipsoid │◀── mean, cov    │  │
│  │   │  (bin probabilities) │              │  (position uncertainty)                │  │
│  │   └──────────────────────┘              └──────────────────────┘                  │  │
│  │                                                                                   │  │
│  │   Both render to Meshcat for real-time 3D visualization                          │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

**Key Abstractions:**

| Layer | Systems | Role |
|-------|---------|------|
| **Robot Simulation** | HardwareStation | Physics, control, cameras, rendering |
| **Perception** | BinLightDark, PositionLightDark, PointCloud pipeline | Sensor models and raw measurements |
| **Estimation** | BinBeliefEstimator, PoseEstimator, PositionBeliefEstimator | Belief state maintenance and updates |
| **Planning** | PlannerSystem | Motion planning, trajectory execution, state machine |
| **Visualization** | BeliefBarChart, CovarianceEllipsoid | Real-time Meshcat overlays |

**Data Flow Summary:**
1. **Robot → Perception**: Joint state determines light/dark region; cameras provide depth images
2. **Perception → Estimation**: Sensor models (TPR/FPR, variance) inform belief updates; point clouds enable ICP
3. **Estimation → Planning**: Pose and covariance drive grasp planning decisions
4. **Planning → Robot**: Joint/gripper commands close the control loop

### Connection Summary Table

| Source System | Output Port | Destination System | Input Port |
|---------------|-------------|-------------------|------------|
| HardwareStation | `iiwa.position_measured` | BinLightDarkRegionSensorSystem | `iiwa.position` |
| HardwareStation | `iiwa.position_measured` | MustardPositionLightDarkRegionSensorSystem | `iiwa.position` |
| BinLightDarkRegionSensorSystem | `sensor_model` | BinBeliefEstimatorSystem | `sensor_model` |
| BinBeliefEstimatorSystem | `belief` | BeliefBarChartSystem | `belief` |
| BinBeliefEstimatorSystem | `belief` | MustardPoseEstimatorSystem | `belief` |
| BinBeliefEstimatorSystem | `belief_confident` | MustardPoseEstimatorSystem | `estimation_trigger` |
| HardwareStation | `camera{0-5}.depth_image` | DepthImageToPointCloud | `depth_image` |
| HardwareStation | `camera{0-5}.rgb_image` | DepthImageToPointCloud | `color_image` |
| HardwareStation | `body_poses` | ExtractPose (x6) | `poses` |
| ExtractPose | `pose` | DepthImageToPointCloud | `camera_pose` |
| DepthImageToPointCloud (x6) | `point_cloud` | MustardPoseEstimatorSystem | `camera{0-5}_point_cloud` |
| MustardPoseEstimatorSystem | `estimated_pose` | PlannerSystem | `estimated_mustard_pose` |
| MustardPoseEstimatorSystem | `estimated_pose` | MustardPositionBeliefEstimatorSystem | `estimated_pose` |
| MustardPositionLightDarkRegionSensorSystem | `measurement_variance` | MustardPositionBeliefEstimatorSystem | `measurement_variance` |
| MustardPositionBeliefEstimatorSystem | `covariance` | PlannerSystem | `position_covariance` |
| MustardPositionBeliefEstimatorSystem | `position_mean` | CovarianceEllipsoidSystem | `position` |
| MustardPositionBeliefEstimatorSystem | `covariance` | CovarianceEllipsoidSystem | `covariance` |
| PlannerSystem | `iiwa_position_command` | HardwareStation | `iiwa.position` |
| PlannerSystem | `wsg_position_command` | HardwareStation | `wsg.position` |

---

## Execution Phases

The system progresses through a well-defined sequence of phases, driven by the `PlannerSystem` state machine. Here is the complete flow:

### Phase 0: Initialization and Setup

**Location in code**: Lines 192-576

1. **Configuration Loading** (`load_config()`)
   - Loads parameters from `config/config.yaml`
   - Includes physics parameters (noise scales, TPR/FPR), planner settings, simulation parameters

2. **Random Seed Setup**
   - Sets global random seeds for `numpy` and `random` modules
   - Ensures reproducible results across runs

3. **Meshcat Visualization**
   - Starts the 3D visualization server on port 7000
   - Restores any saved camera poses

4. **Scenario Loading**
   - Loads the robot and environment configuration from `config/scenario.yaml`
   - Defines the IIWA robot arm, WSG gripper, bins, cameras, and mustard bottle

5. **Diagram Building** (Lines 241-489)
   - Constructs the Drake `Diagram` by adding and connecting all LeafSystems
   - See [System Components](#system-components) below for details

6. **Environment Setup** (Lines 522-550)
   - Randomly chooses which bin contains the mustard bottle (ground truth)
   - Places the mustard bottle at a random position within the chosen bin
   - Updates the belief estimator with the true bin (for simulation feedback)

7. **Planner Configuration** (Line 560)
   - Calls `planner.configure_for_execution(true_bin, X_WM_mustard)`
   - Transitions the planner from `IDLE` to `RRBT_PLANNING` state
   - Stores ground truth information for simulation

---

### Phase 1: RRBT Planning (Bin Belief)

**State**: `PlannerState.RRBT_PLANNING`

**Purpose**: Plan a path that will reduce uncertainty about *which bin* contains the target object.

**Key Actions**:
1. Creates an `IiwaProblemBinBelief` problem instance
2. Runs the RRBT algorithm (`rrbt_planning()`)
3. The planner searches for paths through the **bin light region** where observations are informative
4. Uses a discrete Bayes filter to propagate belief along candidate paths
5. Selects the path that achieves sufficient confidence (misclassification risk < threshold)

**Outputs**:
- A trajectory (`_rrbt_trajectory`) that moves the robot to/through the light region
- A predicted goal configuration based on which bin has highest posterior probability

**Transition**: When planning completes → `RRBT_EXECUTING`

---

### Phase 2: RRBT Execution (Bin Belief)

**State**: `PlannerState.RRBT_EXECUTING`

**Purpose**: Execute the planned trajectory to gather observations and update bin belief.

**Key Actions**:
1. The robot follows the time-parameterized trajectory
2. As the robot enters the **bin light region**:
   - `BinLightDarkRegionSensorSystem` outputs informative sensor model (TPR=0.8, FPR=0.15)
   - `BinBeliefEstimatorSystem` performs Bayes updates on the belief vector
3. The belief bar chart visualization updates in real-time in Meshcat

**Belief Update (Discrete Bayes Filter)**:
```
Prior: P(bin) = [0.5, 0.5]  (uniform, maximum entropy)

For each observation in light region:
    P(bin|obs) ∝ P(obs|bin) × P(bin)
    
    where P(obs|bin) uses TPR/FPR:
    - P(detected|object_present) = TPR = 0.8
    - P(detected|object_absent) = FPR = 0.15

Posterior: P(bin) ≈ [0.95, 0.05] or [0.05, 0.95]
```

**Confidence Check**:
- `belief_confident` signal goes high when: `max(belief) > (1 - max_bin_uncertainty)`
- This triggers the pose estimation phase

**Transition**: When trajectory completes → `POSE_ESTIMATION`

---

### Phase 3: Pose Estimation

**State**: `PlannerState.POSE_ESTIMATION`

**Purpose**: Now that we know *which bin* to look in, estimate the precise 6-DOF pose of the target object.

**Key Actions**:
1. `MustardPoseEstimatorSystem` is triggered by the `belief_confident` signal
2. Uses point clouds from 6 cameras (RGB-D sensors in simulation)
3. Performs ICP (Iterative Closest Point) registration:
   - Segments the point cloud to isolate the target object
   - Aligns a known model of the mustard bottle to the segmented points
   - Outputs the estimated pose `X_WM` (world frame to mustard frame)

**Output**: `estimated_mustard_pose` (RigidTransform)

**Transition**: Immediately → `RRBT2_PLANNING`

---

### Phase 4: RRBT2 Planning (Position Belief)

**State**: `PlannerState.RRBT2_PLANNING`

**Purpose**: Plan a path that will reduce uncertainty about the *precise position* of the target object.

**Key Actions**:
1. Creates an `IiwaProblemMustardPositionBelief` problem instance
2. Uses a **Kalman filter** (continuous Gaussian belief) instead of discrete Bayes
3. The initial position uncertainty comes from ICP estimation error
4. Plans paths through the **mustard position light region** (different from bin light region)
5. Goal: Reduce position covariance below `max_uncertainty` threshold

**Belief Representation**:
```
State: x = [x, y, z] position
Belief: N(μ, Σ)  (3D Gaussian)

Light region: R = σ²_light × I  (low measurement noise)
Dark region:  R = σ²_dark × I   (high measurement noise)

Kalman update reduces Σ when in light region:
    K = Σ × H^T × (H × Σ × H^T + R)^{-1}
    Σ' = (I - K × H) × Σ
```

**Transition**: When planning completes → `RRBT2_EXECUTING`

---

### Phase 5: RRBT2 Execution (Position Belief)

**State**: `PlannerState.RRBT2_EXECUTING`

**Purpose**: Execute the position-uncertainty-reducing trajectory.

**Key Actions**:
1. Robot follows the RRBT2 trajectory
2. `MustardPositionBeliefEstimatorSystem` performs Kalman filter updates
3. `CovarianceEllipsoidSystem` visualizes the shrinking uncertainty ellipsoid in Meshcat
4. As the robot enters the light region, measurement noise decreases, and the covariance shrinks

**Visualization**: A red 3-sigma ellipsoid around the estimated position shrinks as uncertainty decreases.

**Transition**: When trajectory completes → `GRASP_PLANNING`

---

### Phase 6: Grasp Planning

**State**: `PlannerState.GRASP_PLANNING`

**Purpose**: Plan a grasp that accounts for remaining position uncertainty.

**Key Actions**:
1. **Sample Position**: Sample a position from the final covariance ellipsoid
   - Uses `sample_position_from_covariance()` with truncation to stay conservative
2. **Build Grasp Cloud**: Transform the known mustard model to the sampled world position
3. **Grasp Selection**: Run antipodal grasp selection on the point cloud
   - Samples candidate grasps
   - Scores based on antipodal quality
   - Returns top N candidates
4. **IK Validation**: For each candidate grasp:
   - Compute pregrasp pose (30cm offset in gripper Z)
   - Solve inverse kinematics for pregrasp, grasp, lift, and drop poses
   - Use first candidate that passes all IK checks

**Outputs**:
- `_best_grasp_pose`: The selected gripper pose for grasping
- `_pregrasp_pose`: Approach pose above the grasp
- `_grasp_candidates`: Ranked list of grasp options

**Transition**: When planning completes → `GRASP_EXECUTING`

---

### Phase 7: Grasp Execution

**State**: `PlannerState.GRASP_EXECUTING`

**Purpose**: Execute the pick-and-place operation.

**Trajectory Waypoints**:
```
current → [home] → pregrasp → grasp → hold → lift → [transfer waypoints] → drop → release
```

**Key Actions**:
1. **Approach**: Move to pregrasp pose (30cm above grasp)
2. **Grasp**: Descend to grasp pose
3. **Close Gripper**: Hold position while gripper closes (1 second)
4. **Lift**: Raise straight up (30cm) with intermediate waypoints for smooth motion
5. **Transfer**: Move horizontally to drop location above square bin
6. **Release**: Open gripper to drop object

**Gripper Control**:
- Gripper starts open (0.1m)
- Closes at grasp waypoint (0.0m)
- Opens at release waypoint (0.1m)

**Transition**: When trajectory completes → `COMPLETE`

---

### Phase 8: Complete

**State**: `PlannerState.COMPLETE`

**Purpose**: Mission accomplished—hold final position.

**Key Actions**:
1. Holds at the drop position
2. Publishes Meshcat recording for replay
3. Main loop detects completion and prints summary
4. Waits for user to exit (Ctrl+C)

---

## System Components

### HardwareStation (RobotDiagram)
- **Type**: Drake `RobotDiagram` from `manipulation` library
- **Role**: Simulates the robot hardware (IIWA arm, WSG gripper), physics, cameras, and visualization
- **Contains**:
  - `MultibodyPlant` (physics simulation)
  - `SceneGraph` (geometry and collision)
  - `SimIiwaDriver` (arm controller)
  - `SchunkWsgPositionController` (gripper controller)
  - 6× `RgbdSensor` (camera0-5)
  - 3× `MeshcatVisualizer` (illustration, inertia, proximity)
  - `ContactVisualizer`
- **Key Inputs**: `iiwa.position`, `wsg.position`
- **Key Outputs**: `iiwa.position_measured`, `body_poses`, `camera{0-5}.rgb_image`, `camera{0-5}.depth_image`

### PlannerSystem
- **Type**: Custom Drake `LeafSystem`
- **Role**: State machine that orchestrates all planning and execution phases
- **Inputs**: `estimated_mustard_pose` (from MustardPoseEstimator), `position_covariance` (from MustardPositionBeliefEstimator)
- **Outputs**: `iiwa_position_command`, `wsg_position_command`

### BinLightDarkRegionSensorSystem
- **Type**: Custom Drake `LeafSystem`
- **Role**: Determines if gripper is in light/dark region for **bin detection**
- **Input**: `iiwa.position` (from HardwareStation)
- **Outputs**: `in_light_region`, `sensor_model` (TPR, FPR tuple)

### MustardPositionLightDarkRegionSensorSystem
- **Type**: Custom Drake `LeafSystem`
- **Role**: Determines if gripper is in light/dark region for **position estimation**
- **Input**: `iiwa.position` (from HardwareStation)
- **Outputs**: `target_measurement`, `in_light_region`, `measurement_variance`

### BinBeliefEstimatorSystem
- **Type**: Custom Drake `LeafSystem`
- **Role**: Maintains and updates discrete belief over bins (Bayes filter)
- **Input**: `sensor_model` (from BinLightDarkRegionSensorSystem)
- **Outputs**: `belief` (probability vector), `belief_confident` (1.0 when confidence threshold met)

### MustardPoseEstimatorSystem
- **Type**: Custom Drake `LeafSystem`
- **Role**: ICP-based 6-DOF pose estimation from point clouds
- **Inputs**: `camera{0-5}_point_cloud` (x6), `belief` (from BinBeliefEstimator), `estimation_trigger` (from BinBeliefEstimator.belief_confident)
- **Output**: `estimated_pose` (RigidTransform)

### MustardPositionBeliefEstimatorSystem
- **Type**: Custom Drake `LeafSystem`
- **Role**: Maintains and updates continuous belief over position (Kalman filter)
- **Inputs**: `measurement_variance` (from MustardPositionLightDarkRegionSensorSystem), `estimated_pose` (from MustardPoseEstimator)
- **Outputs**: `position_mean` (3D), `covariance` (2×2 flattened to 4)

### BeliefBarChartSystem
- **Type**: Custom Drake `LeafSystem`
- **Role**: Visualizes bin belief as 3D bar charts above each bin in Meshcat
- **Input**: `belief` (from BinBeliefEstimator)

### CovarianceEllipsoidSystem
- **Type**: Custom Drake `LeafSystem`
- **Role**: Visualizes position uncertainty as 3D ellipsoid in Meshcat
- **Inputs**: `position` (from MustardPositionBeliefEstimator.position_mean), `covariance` (from MustardPositionBeliefEstimator)

### Point Cloud Pipeline (per camera)
- **DepthImageToPointCloud**: Converts depth + color images to point cloud
  - Inputs: `depth_image`, `color_image`, `camera_pose`
  - Output: `point_cloud`
- **ExtractPose**: Extracts camera body pose from `body_poses`
  - Input: `poses` (from HardwareStation.body_poses)
  - Output: `pose` (for camera_pose input)

---

## Data Flow Summary

The data flows through three major pipelines that converge at the `PlannerSystem`:

### Pipeline 1: Bin Belief (Discrete)
```
HardwareStation ──(iiwa.position_measured)──▶ BinLightDarkRegionSensorSystem
                                                        │
                                                        │ sensor_model (TPR, FPR)
                                                        ▼
                                              BinBeliefEstimatorSystem
                                                   │         │
                              belief ─────────────┘         └───── belief_confident
                                 │                                      │
                                 ▼                                      │
                        BeliefBarChartSystem                            │
                          (Visualization)                               │
                                                                        ▼
                                                            MustardPoseEstimatorSystem
                                                               (triggers ICP)
```

### Pipeline 2: Pose Estimation (ICP)
```
HardwareStation ──(camera images, body_poses)──▶ DepthImageToPointCloud (x6)
                                                          │
                                                          │ point_cloud (x6)
                                                          ▼
BinBeliefEstimatorSystem ──(belief, belief_confident)──▶ MustardPoseEstimatorSystem
                                                                   │
                                                                   │ estimated_pose
                                                      ┌────────────┴────────────┐
                                                      ▼                         ▼
                                               PlannerSystem      MustardPositionBeliefEstimatorSystem
```

### Pipeline 3: Position Belief (Continuous Kalman)
```
HardwareStation ──(iiwa.position_measured)──▶ MustardPositionLightDarkRegionSensorSystem
                                                          │
                                                          │ measurement_variance
                                                          ▼
MustardPoseEstimatorSystem ──(estimated_pose)──▶ MustardPositionBeliefEstimatorSystem
                                                          │
                                             ┌────────────┴────────────┐
                                             │                         │
                                    position_mean              covariance
                                             │                    │    │
                                             ▼                    │    ▼
                                  CovarianceEllipsoidSystem ◀─────┘  PlannerSystem
                                       (Visualization)
```

### Control Loop (Closed)
```
PlannerSystem ──(iiwa_position_command, wsg_position_command)──▶ HardwareStation
      ▲                                                                │
      │                                                                │
      └────────────(estimated_pose, covariance)────────────────────────┘
```

---

## Configuration Parameters

Key parameters from `config/config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `physics.tpr_light` | True Positive Rate in bin light region (default: 0.8) |
| `physics.fpr_light` | False Positive Rate in bin light region (default: 0.15) |
| `physics.meas_noise_light` | Measurement noise σ in position light region |
| `physics.meas_noise_dark` | Measurement noise σ in position dark region |
| `planner.max_bin_uncertainty` | Misclassification risk threshold for RRBT |
| `planner.max_iterations` | Maximum RRBT iterations |
| `planner.bias_prob_sample_q_bin_light` | Bias probability to sample toward light region |
| `simulation.bin_light_center` | Center of bin observation light region |
| `simulation.bin_light_size` | Size of bin observation light region |
| `simulation.mustard_position_light_center` | Center of position observation light region |
| `simulation.mustard_position_light_size` | Size of position observation light region |

---

## Visualization Outputs

The system generates several visualizations:

1. **Meshcat (http://localhost:7000)**
   - 3D robot and environment rendering
   - Light region indicators (green/orange boxes)
   - Belief bar charts above bins
   - Covariance ellipsoid around estimated position
   - Point clouds from cameras
   - Grasp pose triads

2. **Saved Images**
   - `prior_belief.png` - Initial uniform belief distribution
   - `posterior_belief_rrbt.png` - Belief after RRBT execution (with confidence threshold)

---

## Block Diagram Generation

- Use the new flag to export the full Drake diagram (all subsystems and ports) to Graphviz:
  - `python main.py --generate-block-diagram`
  - Optional custom path/stem: `--diagram-output-stem diagrams/unified_diagram`
- Outputs:
  - `.dot` with the raw `Diagram.GetGraphvizString()` (e.g., `system_diagram_full.dot`)
  - `.png` rendered via Graphviz `dot` if installed (e.g., `system_diagram_full.png`)
- The generated diagram includes the station (RobotDiagram), planner, perception (light/dark sensors), belief estimators (bin + position), pose estimator, visualization systems, and all exported ports. It reflects exactly what `main.py` wires together at build time.

---

## Entry Point

```bash
python main.py [--visualize True/False]
```

The main simulation loop (lines 579-634):
1. Advances the simulator in 0.1-second increments
2. Monitors planner state transitions
3. Generates posterior belief plot when RRBT completes
4. Visualizes point clouds once
5. Checks for completion and exits gracefully

---

## Key Design Decisions

1. **Two-Stage RRBT**: Separates bin identification (discrete) from position refinement (continuous)
2. **Single Source of Truth**: Sensor parameters come from perception systems, not duplicated in estimators
3. **Light/Dark Abstraction**: Cleanly separates informative vs. uninformative sensing regions
4. **Drake Integration**: Leverages Drake's diagram architecture for modular, testable components
5. **Reproducibility**: Global random seeds ensure deterministic runs for debugging and comparison
