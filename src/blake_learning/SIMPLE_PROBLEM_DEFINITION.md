Here is the rigorous problem definition. This specification contains all the mathematical and logical components required to implement this simulation in Python, MATLAB, or C++.

### 1. The Environment & Agents

**The Map ($\mathcal{W}$)**
* **Bounds:** A rectangular domain $\mathcal{W} \subset \mathbb{R}^2$ defined by $[x_{min}, x_{max}] \times [y_{min}, y_{max}]$.
* **The Divider:** An invisible vertical line at $x = x_{split}$.
    * **Region A (Blind Zone):** $x < x_{split}$.
    * **Region B (Observation Zone):** $x \geq x_{split}$.
* **The Landmark ($L$):** A static point feature at unknown coordinates $L = [L_x, L_y]^T$.
    * *Ground Truth:* The true position $L_{true}$ exists in Region B, but is unknown to the robot initially.

**The Robot ($R$)**
* **State:** The robot knows its own pose perfectly at all times.
    * $q_k = [x_k, y_k, \theta_k]^T$ (Position and Heading).
* **Dynamics:** Holonomic and Deterministic. The robot can move in any direction $(v_x, v_y)$ and rotate $\omega$ instantly.
    * Motion Model: $q_{k+1} = q_k + u_k \cdot \Delta t$
    * Control Input: $u_k = [\Delta x, \Delta y, \Delta \theta]^T$.

---

### 2. The Belief State (What the RRT Plans Over)

Since the robot's pose is known, the "Belief" applies **only** to the location of the landmark.

**The Belief State $b_k$**
$$b_k = (\mu_k, \Sigma_k)$$
* **$\mu_k$ (Mean):** The estimated 2D position of the landmark $[ \hat{L}_x, \hat{L}_y ]^T$.
* **$\Sigma_k$ (Covariance):** A $2 \times 2$ symmetric positive-definite matrix representing the uncertainty of the landmark location.
    $$\Sigma_k = \begin{bmatrix} \sigma_{xx}^2 & \sigma_{xy} \\ \sigma_{yx} & \sigma_{yy}^2 \end{bmatrix}$$

**Initial Conditions ($k=0$)**
* **Robot Start:** $q_0$ is in Region A ($x_0 < x_{split}$).
* **Initial Belief:**
    * $\mu_0$: Arbitrary guess (e.g., center of Region B).
    * $\Sigma_0$: Effectively infinite (representing total ignorance). In code, initialize as a diagonal matrix with very large values (e.g., $10^6 \cdot I$).

---

### 3. The Sensor Model (Stochastic)

The robot is equipped with a camera that provides Range ($r$) and Bearing ($\phi$) measurements relative to the robot's frame.

**A. Field of View Constraint**
The robot receives a measurement $z_k$ if and only if **all** the following are true:
1.  **Zone Check:** The robot is in Region B ($x_k \geq x_{split}$).
2.  **Range Check:** $|| L - (x_k, y_k) || \leq R_{max}$.
3.  **Angle Check:** The landmark is within the camera's angle of view $\pm \alpha_{fov}$ relative to the robot's heading $\theta_k$.

If constraints are not met, $z_k = \emptyset$.

**B. Observation Equation**
If the sensor is active, the measurement $z_k$ is:
$$z_k = h(q_k, L) + v_k$$
$$z_k = \begin{bmatrix} r \\ \phi \end{bmatrix} = \begin{bmatrix} \sqrt{(L_x - x_k)^2 + (L_y - y_k)^2} \\ \text{atan2}(L_y - y_k, L_x - x_k) - \theta_k \end{bmatrix} + v_k$$

**C. Noise Model ($v_k$)**
The noise is zero-mean Gaussian $v_k \sim \mathcal{N}(0, R_{noise})$.
Crucially, you specified "depth noise." Usually, range noise scales with distance.
$$R_{noise} = \begin{bmatrix} \sigma_r^2 & 0 \\ 0 & \sigma_{\phi}^2 \end{bmatrix}$$
* $\sigma_r$: Standard deviation of range (e.g., $0.1 \cdot r_{measured}$ or a constant).
* $\sigma_{\phi}$: Standard deviation of bearing (constant, e.g., 0.05 radians).

---

### 4. Belief Dynamics (The "Filter" for the RRT)

When the RRT expands a node, it must simulate how $\Sigma$ changes. Since $L$ is static, we use the measurement update of the Extended Kalman Filter (EKF).

**Step 1: The Jacobian ($H_k$)**
We need the Jacobian of the observation function $h$ with respect to the landmark state $L$:
$$H_k = \frac{\partial h}{\partial L} = \begin{bmatrix} \frac{L_x - x_k}{r} & \frac{L_y - y_k}{r} \\ \frac{-(L_y - y_k)}{r^2} & \frac{L_x - x_k}{r^2} \end{bmatrix}$$

**Step 2: Covariance Update**
If the sensor is active (in Region B), the covariance contracts. The computationally efficient form (Joseph form or Information form) for the update is:
$$\Sigma_{k+1} = (\Sigma_k^{-1} + H_k^T R_{noise}^{-1} H_k)^{-1}$$
*(Note: If the sensor is inactive/blind, $\Sigma_{k+1} = \Sigma_k$)*.

---

### 5. The Planning Problem

**The Objective**
Find a sequence of controls $u_{0:T}$ that results in a trajectory where the final belief covariance satisfies a "tightness" constraint.

**RRT Node Structure**
Each node $N$ in the tree contains:
1.  **Robot Pose:** $q$
2.  **Landmark Covariance:** $\Sigma$
3.  **Cost:** $J$

**The Goal Condition**
The planner succeeds when a node is added that satisfies:
$$\text{Trace}(\Sigma_{final}) < \epsilon$$
Where $\epsilon$ is a small user-defined threshold (e.g., 0.1).
*Alternatively, use the Determinant (volume of uncertainty ellipse).*

**Cost Function (for optimizing path)**
$$Cost = \sum_{k=0}^{T} ||q_{k+1} - q_k||$$
*Note: While standard RRT finds any path, heuristic guidance usually adds a penalty for high uncertainty to encourage the robot to enter Region B quickly.*

### Summary for Implementation
1.  **Initialize** root node with infinite $\Sigma$.
2.  **Loop:**
    * Sample random pose $q_{rand}$.
    * Find nearest node in tree.
    * Steer robot towards $q_{rand}$ (deterministic motion).
    * **Check Region:** If in Region B, calculate Jacobian $H$ and update $\Sigma_{new}$ using the math in Section 4.
    * Add new node $(q_{new}, \Sigma_{new})$.
    * **Check Goal:** Is $\text{Trace}(\Sigma_{new}) < \epsilon$? If yes, backtrace and return path.

Would you like me to generate the pseudocode for the `Extend` function specifically, as that is where the covariance math lives?