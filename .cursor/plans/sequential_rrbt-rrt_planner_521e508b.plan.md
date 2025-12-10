---
name: Sequential RRBT-RRT Planner
overview: "Create a PlannerSystem LeafSystem with staged planning: first RRBT plans and executes for information gathering, then RRT plans and executes from the RRBT endpoint to the goal."
todos:
  - id: create-planner-system
    content: Create src/planning/planner_system.py with PlannerSystem LeafSystem and state machine (IDLE -> RRBT_PLANNING -> RRBT_EXECUTING -> RRT_PLANNING -> RRT_EXECUTING -> COMPLETE)
    status: completed
  - id: refactor-main
    content: Refactor main.py to use PlannerSystem, remove execution diagram construction, wire planner output to station
    status: completed
---

# Sequential RRBT-RRT Planner System

## Goal

Create a unified Drake simulation with a `PlannerSystem` that executes planning in stages:

1. RRBT plans, then robot executes RRBT trajectory (information gathering)
2. RRT plans from RRBT endpoint to goal, then robot executes RRT trajectory

## State Machine

```
┌─────────────────┐
│   IDLE          │  Output: q_home
│ (waiting setup) │
└────────┬────────┘
         │ configure_for_execution() called
         v
┌─────────────────┐
│  RRBT_PLANNING  │  Output: q_home (blocking while RRBT runs)
│                 │
└────────┬────────┘
         │ RRBT completes, trajectory created
         v
┌─────────────────┐
│ RRBT_EXECUTING  │  Output: rrbt_trajectory(t)
│                 │
└────────┬────────┘
         │ RRBT trajectory finished (t >= end_time)
         v
┌─────────────────┐
│  RRT_PLANNING   │  Output: hold at rrbt_trajectory end (blocking while RRT runs)
│                 │
└────────┬────────┘
         │ RRT completes, trajectory created
         v
┌─────────────────┐
│  RRT_EXECUTING  │  Output: rrt_trajectory(t)
│                 │
└────────┬────────┘
         │ RRT trajectory finished
         v
┌─────────────────┐
│    COMPLETE     │  Output: hold at final position (q_goal)
└─────────────────┘
```

## Implementation

### 1. Create `src/planning/planner_system.py`

```python
from enum import Enum, auto
from pydrake.all import LeafSystem, BasicVector, PiecewisePolynomial
import numpy as np

class PlannerState(Enum):
    IDLE = auto()              # Waiting for setup
    RRBT_PLANNING = auto()     # Running RRBT (blocking)
    RRBT_EXECUTING = auto()    # Playing RRBT trajectory
    RRT_PLANNING = auto()      # Running RRT (blocking)  
    RRT_EXECUTING = auto()     # Playing RRT trajectory
    COMPLETE = auto()          # Done, holding at goal

class PlannerSystem(LeafSystem):
    def __init__(self, plant, config, meshcat, scenario_path):
        LeafSystem.__init__(self)
        
        # Configuration
        self._plant = plant
        self._config = config
        self._meshcat = meshcat
        self._scenario_path = scenario_path
        
        # Positions (computed in constructor via IK)
        self._q_home = np.array(config.simulation.q_home)
        self._q_goal = None      # Computed via IK from tf_goal
        self._q_light_hint = None  # Computed via IK from light_center
        
        # State machine
        self._state = PlannerState.IDLE
        
        # RRBT stage
        self._rrbt_trajectory = None
        self._rrbt_start_time = None
        self._rrbt_end_position = None
        self._pred_q_goal = None  # Goal predicted by RRBT belief
        
        # RRT stage  
        self._rrt_trajectory = None
        self._rrt_start_time = None
        
        # Runtime parameters (set by configure_for_execution)
        self._true_bin = None
        self._X_WM_mustard = None
        
        # Compute q_goal and q_light_hint via IK
        self._compute_ik_targets()
        
        # Output port
        self.DeclareVectorOutputPort(
            "iiwa_position_command", 
            BasicVector(7), 
            self.CalcJointCommand
        )
    
    def configure_for_execution(self, true_bin, X_WM_mustard=None):
        """Called after mustard is placed to enable planning."""
        self._true_bin = true_bin
        self._X_WM_mustard = X_WM_mustard
        self._state = PlannerState.RRBT_PLANNING
    
    def CalcJointCommand(self, context, output):
        t = context.get_time()
        
        if self._state == PlannerState.IDLE:
            output.SetFromVector(self._q_home)
            
        elif self._state == PlannerState.RRBT_PLANNING:
            # Run RRBT (blocking)
            self._run_rrbt_planning()
            self._rrbt_start_time = t
            self._state = PlannerState.RRBT_EXECUTING
            output.SetFromVector(self._q_home)
            
        elif self._state == PlannerState.RRBT_EXECUTING:
            t_traj = t - self._rrbt_start_time
            if t_traj >= self._rrbt_trajectory.end_time():
                # RRBT trajectory complete, transition to RRT planning
                self._state = PlannerState.RRT_PLANNING
                self._rrbt_end_position = self._rrbt_trajectory.value(
                    self._rrbt_trajectory.end_time()
                ).flatten()
                output.SetFromVector(self._rrbt_end_position)
            else:
                q = self._rrbt_trajectory.value(t_traj).flatten()
                output.SetFromVector(q)
                
        elif self._state == PlannerState.RRT_PLANNING:
            # Run RRT from RRBT endpoint to predicted goal (blocking)
            self._run_rrt_planning()
            self._rrt_start_time = t
            self._state = PlannerState.RRT_EXECUTING
            output.SetFromVector(self._rrbt_end_position)
            
        elif self._state == PlannerState.RRT_EXECUTING:
            t_traj = t - self._rrt_start_time
            if t_traj >= self._rrt_trajectory.end_time():
                # RRT trajectory complete
                self._state = PlannerState.COMPLETE
                output.SetFromVector(self._pred_q_goal)
            else:
                q = self._rrt_trajectory.value(t_traj).flatten()
                output.SetFromVector(q)
                
        elif self._state == PlannerState.COMPLETE:
            output.SetFromVector(self._pred_q_goal)
    
    def _run_rrbt_planning(self):
        """Execute RRBT planning, populate _rrbt_trajectory and _pred_q_goal."""
        # Creates IiwaProblemBinBelief and calls rrbt_planning()
        # Converts path to PiecewisePolynomial trajectory
        # Stores predicted goal from belief
        ...
    
    def _run_rrt_planning(self):
        """Execute RRT from _rrbt_end_position to _pred_q_goal."""
        # Creates IiwaProblem and calls rrt_planning()
        # Converts path to PiecewisePolynomial trajectory
        ...
    
    def _compute_ik_targets(self):
        """Compute q_goal and q_light_hint from task-space targets via IK."""
        ...
```

### 2. Key Methods Detail

**`_run_rrbt_planning()`** - Adapts lines 410-452 from main.py:

```python
def _run_rrbt_planning(self):
    problem = IiwaProblemBinBelief(
        q_start=tuple(self._q_home),
        q_goal=tuple(self._q_goal),
        gripper_setpoint=0.1,
        meshcat=self._meshcat,
        light_center=self._config.simulation.light_center,
        light_size=self._config.simulation.light_size,
        tpr_light=float(self._config.physics.tpr_light),
        fpr_light=float(self._config.physics.fpr_light),
        n_bins=2,
        true_bin=self._true_bin,
        max_bin_uncertainty=float(self._config.planner.max_bin_uncertainty),
        lambda_weight=float(self._config.planner.bin_lambda_weight),
    )
    
    rrbt_result, _ = rrbt_planning(
        problem,
        max_iterations=int(self._config.planner.max_iterations),
        bias_prob_sample_q_goal=float(self._config.planner.bias_prob_sample_q_goal),
        bias_prob_sample_q_bin_light=float(self._config.planner.bias_prob_sample_q_bin_light),
        q_light_hint=self._q_light_hint,
    )
    
    if rrbt_result:
        path_to_info, self._pred_q_goal = rrbt_result
        self._rrbt_trajectory = path_to_trajectory(path_to_info)
    else:
        # RRBT failed - create minimal trajectory to stay at home
        self._rrbt_trajectory = path_to_trajectory([tuple(self._q_home)])
        self._pred_q_goal = self._q_goal  # Fallback to original goal
```

**`_run_rrt_planning()`** - Adapts lines 458-477 from main.py:

```python
def _run_rrt_planning(self):
    problem = IiwaProblem(
        q_start=tuple(self._rrbt_end_position),
        q_goal=tuple(self._pred_q_goal),
        gripper_setpoint=0.1,
        meshcat=self._meshcat,
    )
    
    path_to_grasp, _ = rrt_planning(
        problem, 
        max_iterations=1000,
        prob_sample_q_goal=0.25,
    )
    
    if path_to_grasp:
        self._rrt_trajectory = path_to_trajectory(path_to_grasp)
    else:
        # RRT failed - create trajectory to hold position
        self._rrt_trajectory = path_to_trajectory([tuple(self._rrbt_end_position)])
```

### 3. Modify `main.py`

**Remove:**

- Execution diagram construction (lines 483-599)
- Planning logic currently in main() (lines 364-480)
- ConstantVectorSource for iiwa.position

**Simplified main() structure:**

```python
def main():
    # 1. Setup (meshcat, config, scenario) - same as before
    config = load_rrbt_config()
    meshcat = Meshcat(...)
    
    # 2. Build unified diagram
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(...))
    plant = station.GetSubsystemByName("plant")
    
    # 3. Add PlannerSystem (replaces ConstantVectorSource)
    planner = builder.AddSystem(PlannerSystem(plant, config, meshcat, scenario_path))
    builder.Connect(
        planner.GetOutputPort("iiwa_position_command"),
        station.GetInputPort("iiwa.position")
    )
    
    # 4. Add perception and belief systems (same as before)
    perception_sys = builder.AddSystem(LightDarkRegionSystem(...))
    belief_estimator = builder.AddSystem(BeliefEstimatorSystem(...))
    belief_viz = builder.AddSystem(BeliefBarChartSystem(...))
    # ... wire connections ...
    
    # 5. WSG gripper (constant open)
    wsg_source = builder.AddSystem(ConstantVectorSource([0.1]))
    builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))
    
    # 6. Build and initialize
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    
    # 7. Place mustard bottle (requires simulator context)
    sim_context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(sim_context)
    true_bin = np.random.randint(0, 2)
    X_WM = place_mustard_bottle_randomly_in_bin(meshcat, plant, plant_context, true_bin, np_rng)
    diagram.ForcedPublish(sim_context)
    
    # 8. Configure planner and run
    planner.configure_for_execution(true_bin, X_WM)
    
    meshcat.StartRecording()
    
    # Run until complete (or use a reasonable max time)
    while True:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)
        # Could check planner state to know when done
    
    meshcat.StopRecording()
    meshcat.PublishRecording()
```

## Execution Timeline Example

```
Time 0.0s:  IDLE → q_home
Time 0.1s:  configure_for_execution() called
Time 0.1s:  RRBT_PLANNING (blocking ~5-10s) → q_home
Time 0.1s:  RRBT_EXECUTING starts → trajectory playback
Time 2.5s:  RRBT trajectory ends
Time 2.5s:  RRT_PLANNING (blocking ~1-3s) → hold at RRBT end
Time 2.5s:  RRT_EXECUTING starts → trajectory playback  
Time 4.0s:  RRT trajectory ends → COMPLETE, hold at goal
```

## File Changes Summary

| File | Action |

|------|--------|

| `src/planning/planner_system.py` | **Create** - PlannerSystem with state machine |

| `main.py` | **Modify** - Use PlannerSystem, remove dual-diagram architecture |