---
name: Refactor to 3-Bin Bayes Filter
overview: Refactor the RRBT planning system from continuous Kalman Filter belief (7x7 covariance matrix) to a discrete 3-bin Bayes filter (3-element probability vector), using expected posteriors during planning and measuring all 3 buckets each timestep in the light region.
todos:
  - id: create-bayes-module
    content: Create `src/estimation/bayes_filter.py` with `calculate_misclassification_risk()`, `bayes_update_single()`, `bayes_update_all_buckets()`, and `expected_posterior_all_buckets()`
    status: completed
  - id: modify-simulation-tools
    content: Modify `IiwaProblemBelief` to use TPR/FPR sensor model, add `get_sensor_model()`, add `n_buckets` and `true_bucket` attributes
    status: completed
  - id: modify-rrbt-tree
    content: Change `BeliefNode.sigma` to `BeliefNode.belief`, update `RRBT_Tree.Propagate()` to use `expected_posterior_all_buckets()`
    status: completed
  - id: modify-rrt-tools
    content: Update `node_reaches_goal()` to use `calculate_misclassification_risk(belief) < threshold`
    status: completed
  - id: modify-belief-estimator
    content: Change `BeliefEstimatorSystem` to discrete Bayes filter using `bayes_update_all_buckets()`
    status: completed
  - id: modify-perception
    content: Update `LightDarkRegionSystem` to output `[TPR, FPR]` instead of variance
    status: completed
  - id: create-bar-chart-viz
    content: Create `src/visualization/belief_bar_chart.py` to render belief as 3 bars in Meshcat
    status: completed
  - id: update-config
    content: Update `config.yaml` with `tpr_light`, `fpr_light`, `n_buckets`, `true_bucket`
    status: completed
  - id: update-main
    content: Update `main.py` to wire new systems and replace ellipsoid visualizer with bar chart
    status: completed
---

# Refactor RRBT to 3-Bin Bayes Filter

## Design Decisions

1. **Buckets are abstract** - 3 discrete hypotheses; will be grounded later
2. **Measure all 3 buckets each timestep in light** - When in light, take 3 measurements (one per bucket) and update posterior for each
3. **Use expected posterior during planning** - Assume object is in one bucket; planner uses expected posterior which converges to correct hypothesis
4. **Bar chart visualization** - Plot belief probabilities during execution

## Current vs. Target Architecture

| Aspect | Current (Kalman) | Target (3-bin Bayes) |

|--------|------------------|---------------------|

| Belief | 7x7 covariance Σ | 3-element `[P(A), P(B), P(C)]` |

| Uncertainty Metric | `trace(Σ)` | `1 - max(belief)` (misclassification risk) |

| Sensor Model | Gaussian variance R | TPR/FPR binary sensor |

| Light Region | Low R (0.01) | Informative: TPR=0.8, FPR=0.15 |

| Dark Region | High R (1e9) | Uninformative: TPR=0.5, FPR=0.5 |

| Measurements | Single observation | 3 measurements (all buckets) per timestep in light |

---

## Files to Modify

### 1. Create `src/estimation/bayes_filter.py` (NEW FILE)

Extract and extend core Bayes functions:

```python
import numpy as np

def calculate_misclassification_risk(belief: np.ndarray) -> float:
    """Risk = 1 - P(MAP estimate)"""
    return 1.0 - np.max(belief)

def bayes_update_single(belief: np.ndarray, measured_bucket: int, 
                        observation: bool, tpr: float, fpr: float) -> np.ndarray:
    """Update belief after measuring ONE bucket."""
    likelihood = np.zeros(len(belief))
    for i in range(len(belief)):
        if i == measured_bucket:
            likelihood[i] = tpr if observation else (1 - tpr)
        else:
            likelihood[i] = fpr if observation else (1 - fpr)
    
    posterior = likelihood * belief
    return posterior / np.sum(posterior)

def bayes_update_all_buckets(belief: np.ndarray, true_bucket: int,
                             tpr: float, fpr: float) -> np.ndarray:
    """
    Measure ALL buckets and update belief sequentially.
    Used when robot is in light region.
    
    Args:
        belief: Current belief [P(A), P(B), P(C)]
        true_bucket: Index of bucket where object actually is (for simulation)
        tpr: True positive rate
        fpr: False positive rate
    """
    n_buckets = len(belief)
    updated_belief = belief.copy()
    
    for bucket_idx in range(n_buckets):
        # Simulate observation for this bucket
        is_object_here = (bucket_idx == true_bucket)
        if is_object_here:
            observation = np.random.rand() < tpr  # Detection with prob TPR
        else:
            observation = np.random.rand() < fpr  # False alarm with prob FPR
        
        # Update belief with this measurement
        updated_belief = bayes_update_single(updated_belief, bucket_idx, 
                                              observation, tpr, fpr)
    
    return updated_belief

def expected_posterior_all_buckets(belief: np.ndarray, tpr: float, fpr: float) -> np.ndarray:
    """
    Compute EXPECTED posterior after measuring all buckets.
    Used during PLANNING (no actual observations, just expectations).
    
    For each bucket measurement, we compute:
    E[posterior] = P(obs=1) * posterior(obs=1) + P(obs=0) * posterior(obs=0)
    """
    n_buckets = len(belief)
    expected_belief = belief.copy()
    
    for measured_bucket in range(n_buckets):
        # Probability of positive observation when measuring this bucket
        # P(obs=1) = sum over hypotheses: P(obs=1|hypothesis) * P(hypothesis)
        p_obs_positive = 0.0
        for hyp in range(n_buckets):
            if hyp == measured_bucket:
                p_obs_positive += tpr * expected_belief[hyp]
            else:
                p_obs_positive += fpr * expected_belief[hyp]
        p_obs_negative = 1.0 - p_obs_positive
        
        # Compute posterior for each observation outcome
        post_if_positive = bayes_update_single(expected_belief, measured_bucket, True, tpr, fpr)
        post_if_negative = bayes_update_single(expected_belief, measured_bucket, False, tpr, fpr)
        
        # Expected posterior
        expected_belief = p_obs_positive * post_if_positive + p_obs_negative * post_if_negative
    
    return expected_belief
```

---

### 2. Modify `src/simulation/simulation_tools.py`

**`IiwaProblemBelief.__init__`:**

```python
def __init__(self, q_start, q_goal, gripper_setpoint, meshcat,
             light_center, light_size, 
             tpr_light=0.80, fpr_light=0.15,  # NEW: replaces scale_R_light/dark
             n_buckets=3, true_bucket=0):     # NEW: bucket config
    # ... existing init ...
    
    # TPR/FPR sensor model (replaces R_light, R_dark)
    self.tpr_light = tpr_light
    self.fpr_light = fpr_light
    self.tpr_dark = 0.5   # Uninformative (coin flip)
    self.fpr_dark = 0.5   # Uninformative (coin flip)
    
    # Bucket configuration
    self.n_buckets = n_buckets
    self.true_bucket = true_bucket  # Ground truth for simulation
```

**Replace `get_dynamics_and_observation()`:**

```python
def get_sensor_model(self, q: tuple) -> tuple[float, float]:
    """Returns (TPR, FPR) based on light/dark region."""
    if self.is_in_light(q):
        return self.tpr_light, self.fpr_light
    else:
        return self.tpr_dark, self.fpr_dark  # Uninformative
```

---

### 3. Modify `src/planning/rrbt_tree.py`

**`BeliefNode` class - change sigma to belief vector:**

```python
class BeliefNode(TreeNode):
    def __init__(self, value, parent=None, belief=None, cost=0.0, path_length=0.0):
        super().__init__(value, parent)
        self.belief = belief  # [P(A), P(B), P(C)] - replaces sigma
        self.path_length = path_length
        self.cost = cost
```

**`RRBT_Tree.__init__` - initialize with uniform prior:**

```python
def __init__(self, problem, root_value, max_uncertainty, 
             initial_uncertainty=None, lambda_weight=1.0):
    # ...
    n_buckets = problem.n_buckets
    init_belief = np.ones(n_buckets) / n_buckets  # Uniform prior
    init_risk = calculate_misclassification_risk(init_belief)  # 0.667 for 3 buckets
    init_cost = 0.0 + lambda_weight * init_risk
    
    self.root = BeliefNode(root_value, parent=None, belief=init_belief,
                           cost=init_cost, path_length=0.0)
```

**`RRBT_Tree.Propagate()` - use expected posterior:**

```python
def Propagate(self, parent_node, q_target):
    belief_parent = parent_node.belief
    
    # Get sensor model
    tpr, fpr = self.problem.get_sensor_model(q_target)
    
    # If in light: measure all buckets (expected posterior)
    # If in dark: belief unchanged (uninformative measurements)
    if self.problem.is_in_light(q_target):
        belief_new = expected_posterior_all_buckets(belief_parent, tpr, fpr)
    else:
        belief_new = belief_parent.copy()  # No information gain in dark
    
    # Compute cost
    dist_increment = self.cspace.distance(parent_node.value, q_target)
    path_length_new = parent_node.path_length + dist_increment
    misclass_risk = calculate_misclassification_risk(belief_new)
    cost_new = path_length_new + self.lambda_weight * misclass_risk
    
    return {"belief": belief_new, "cost": cost_new, "path_length": path_length_new}
```

---

### 4. Modify `src/planning/rrt_tools.py`

**`RRBT_tools.node_reaches_goal()`:**

```python
def node_reaches_goal(self, node, tol=None):
    """Terminate when misclassification risk is sufficiently low."""
    from src.estimation.bayes_filter import calculate_misclassification_risk
    misclass_risk = calculate_misclassification_risk(node.belief)
    return misclass_risk <= self.MAX_UNCERTAINTY  # e.g., 0.01
```

---

### 5. Modify `src/estimation/belief_estimator.py`

Change from Kalman to discrete Bayes filter:

```python
class BeliefEstimatorSystem(LeafSystem):
    def __init__(self, n_buckets=3, true_bucket=0, update_period=0.01):
        LeafSystem.__init__(self)
        
        self._n_buckets = n_buckets
        self._true_bucket = true_bucket  # For simulation
        
        # Input: TPR and FPR from perception
        self._sensor_port = self.DeclareVectorInputPort("sensor_model", 2)
        
        # Discrete state: belief vector (n_buckets elements)
        initial_belief = np.ones(n_buckets) / n_buckets
        self._belief_state_index = self.DeclareDiscreteState(initial_belief)
        
        # Output: belief vector
        self.DeclareVectorOutputPort("belief", n_buckets, self._CalcBelief)
        
        # Periodic update
        self.DeclarePeriodicDiscreteUpdateEvent(update_period, 0.0, self._DoBayesUpdate)
    
    def _DoBayesUpdate(self, context, discrete_state):
        tpr, fpr = self._sensor_port.Eval(context)
        belief = discrete_state.get_mutable_vector(self._belief_state_index)
        current_belief = belief.get_mutable_value()
        
        # Only update if informative (in light)
        if tpr != 0.5:  # Not in dark
            updated = bayes_update_all_buckets(current_belief, self._true_bucket, tpr, fpr)
            belief.set_value(updated)
```

---

### 6. Modify `src/perception/light_and_dark.py`

**Change output from variance to TPR/FPR:**

```python
def __init__(self, plant, light_region_center, light_region_size,
             tpr_light=0.80, fpr_light=0.15):  # NEW signature
    # ...
    self._tpr_light = tpr_light
    self._fpr_light = fpr_light
    
    # Replace measurement_variance output with sensor_model
    self.DeclareVectorOutputPort("sensor_model", 2, self.CalcSensorModel)

def CalcSensorModel(self, context, output):
    """Output [TPR, FPR] based on light/dark region."""
    is_light = self._is_in_light(context)
    if is_light:
        output.SetFromVector([self._tpr_light, self._fpr_light])
    else:
        output.SetFromVector([0.5, 0.5])  # Uninformative
```

---

### 7. Create `src/visualization/belief_bar_chart.py` (NEW FILE)

```python
class BeliefBarChartSystem(LeafSystem):
    """Visualize belief as bar chart in Meshcat."""
    
    def __init__(self, meshcat, n_buckets=3):
        LeafSystem.__init__(self)
        self._meshcat = meshcat
        self._n_buckets = n_buckets
        self._belief_port = self.DeclareVectorInputPort("belief", n_buckets)
        
        self.DeclarePeriodicPublishEvent(0.1, 0.0, self._Publish)
    
    def _Publish(self, context):
        belief = self._belief_port.Eval(context)
        # Create 3 boxes in Meshcat with heights proportional to belief
        for i, p in enumerate(belief):
            box_height = max(0.01, p * 0.5)  # Scale for visibility
            # Update Meshcat box geometry...
```

---

### 8. Update `config/config.yaml`

```yaml
physics:
  # TPR/FPR sensor model (replaces meas_noise_light/dark)
  tpr_light: 0.80   # True Positive Rate in light
  fpr_light: 0.15   # False Positive Rate in light
  # Dark region: TPR=FPR=0.5 (hardcoded, coin flip)

planner:
  # Bucket configuration
  n_buckets: 3
  true_bucket: 0    # Ground truth for simulation (bucket A)
  
  # Termination: misclassification_risk < max_uncertainty
  max_uncertainty: 0.01
```

---

### 9. Update `main.py`

- Pass `n_buckets`, `true_bucket`, `tpr_light`, `fpr_light` to `IiwaProblemBelief`
- Replace `BeliefVisualizerSystem` (ellipsoid) with `BeliefBarChartSystem`
- Update `debug_path_beliefs()` to print belief vector and misclassification risk instead of trace(Σ)