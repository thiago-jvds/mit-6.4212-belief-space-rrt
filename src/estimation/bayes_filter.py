"""
Discrete Bayes Filter for 3-Bin Belief Space Planning.

This module implements a simple discrete Bayes filter for categorical
hypothesis testing with a binary sensor model (TPR/FPR).

The belief state is a probability vector [P(A), P(B), P(C)] representing
the probability that the object is in each of the 3 bins.

Sensor Model:
- TPR (True Positive Rate): P(detected | object present)
- FPR (False Positive Rate): P(detected | object absent)

In light region: Informative sensor (TPR=0.8, FPR=0.15)
In dark region: Uninformative sensor (TPR=0.5, FPR=0.5)
"""

import numpy as np


def calculate_misclassification_risk(belief: np.ndarray) -> float:
    """
    Compute the Misclassification Risk Cost.
    
    This measures the probability that the true state is NOT the 
    Maximum A Posteriori (MAP) estimate.
    
    Risk = 1 - P(x_MAP) = 1 - max(belief)
    
    Args:
        belief: Probability vector [P(A), P(B), P(C)]
        
    Returns:
        Misclassification risk (0.0 = certain, approaches 1.0 = uncertain)
    """
    return 1.0 - np.max(belief)


def bayes_update_single(
    belief: np.ndarray,
    measured_bin: int,
    observation: bool,
    tpr: float,
    fpr: float,
) -> np.ndarray:
    """
    Update belief after measuring ONE bin with a binary sensor.
    
    Applies Bayes' rule:
        posterior(x) ∝ likelihood(z|x) × prior(x)
    
    Args:
        belief: Current belief vector [P(A), P(B), P(C)]
        measured_bin: Index of the bin being measured (0, 1, or 2)
        observation: True if detection, False if no detection
        tpr: True Positive Rate P(detected | object present)
        fpr: False Positive Rate P(detected | object absent)
        
    Returns:
        Normalized posterior belief vector
    """
    n_bins = len(belief)
    likelihood = np.zeros(n_bins)
    
    for hypothesis_idx in range(n_bins):
        if hypothesis_idx == measured_bin:
            # Hypothesis: object is in the bin we just measured
            # P(observation | object here)
            if observation:
                likelihood[hypothesis_idx] = tpr  # True positive
            else:
                likelihood[hypothesis_idx] = 1 - tpr  # False negative
        else:
            # Hypothesis: object is in a DIFFERENT bin
            # P(observation | object NOT here)
            if observation:
                likelihood[hypothesis_idx] = fpr  # False positive
            else:
                likelihood[hypothesis_idx] = 1 - fpr  # True negative
    
    # Bayes update
    unnormalized_posterior = likelihood * belief
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
    
    return posterior


def bayes_update_all_bins(
    belief: np.ndarray,
    true_bin: int,
    tpr: float,
    fpr: float,
) -> np.ndarray:
    """
    Measure ALL bins and update belief sequentially.
    
    Used during EXECUTION when robot is in the light region.
    Each bin is measured once, and observations are simulated
    based on the true_bin (ground truth).
    
    Args:
        belief: Current belief vector [P(A), P(B), P(C)]
        true_bin: Index of bin where object actually is (ground truth)
        tpr: True Positive Rate
        fpr: False Positive Rate
        
    Returns:
        Updated belief after measuring all bins
    """
    n_bins = len(belief)
    updated_belief = belief.copy()
    
    for bin_idx in range(n_bins):
        # Simulate observation for this bin based on ground truth
        is_object_here = (bin_idx == true_bin)
        
        if is_object_here:
            # Object is in this bin: detection with probability TPR
            observation = np.random.rand() < tpr
        else:
            # Object not in this bin: false alarm with probability FPR
            observation = np.random.rand() < fpr
        
        # Update belief with this measurement
        updated_belief = bayes_update_single(
            updated_belief, bin_idx, observation, tpr, fpr
        )
    
    return updated_belief


def expected_posterior_all_bins(
    belief: np.ndarray,
    tpr: float,
    fpr: float,
    assumed_bin: int = 0,
) -> np.ndarray:
    """
    Compute EXPECTED posterior after measuring all bins, assuming the
    object is in a specific bin.
    
    Used during PLANNING. We assume the object is in `assumed_bin` and
    compute the expected posterior given that assumption. This allows the
    planner to reason about information gain - if we go to the light region,
    the expected belief will converge toward certainty about `assumed_bin`.
    
    For each bin measurement:
    - If measuring the assumed_bin: expect observation=True with prob TPR
    - If measuring other bins: expect observation=True with prob FPR
    
    Args:
        belief: Current belief vector [P(A), P(B), P(C)]
        tpr: True Positive Rate
        fpr: False Positive Rate
        assumed_bin: Which bin to assume the object is in (for planning)
        
    Returns:
        Expected posterior after measuring all bins
    """
    n_bins = len(belief)
    expected_belief = belief.copy()
    
    for measured_bin in range(n_bins):
        # Determine if this is the bin where we assume the object is
        is_assumed_bin = (measured_bin == assumed_bin)
        
        if is_assumed_bin:
            # We're measuring the bin where (we assume) the object is
            # Expect positive observation with probability TPR
            p_obs_positive = tpr
        else:
            # We're measuring a bin where (we assume) the object is NOT
            # Expect positive observation with probability FPR (false alarm)
            p_obs_positive = fpr
        
        p_obs_negative = 1.0 - p_obs_positive
        
        # Compute posterior for each observation outcome
        posterior_if_positive = bayes_update_single(
            expected_belief, measured_bin, True, tpr, fpr
        )
        posterior_if_negative = bayes_update_single(
            expected_belief, measured_bin, False, tpr, fpr
        )
        
        # Expected posterior (weighted average given our assumption)
        expected_belief = (
            p_obs_positive * posterior_if_positive +
            p_obs_negative * posterior_if_negative
        )
    
    return expected_belief

