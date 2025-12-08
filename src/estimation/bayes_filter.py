"""
Discrete Bayes Filter for 3-Bin Belief Space Planning.

This module implements a simple discrete Bayes filter for categorical
hypothesis testing with a binary sensor model (TPR/FPR).

The belief state is a probability vector [P(A), P(B), P(C)] representing
the probability that the object is in each of the 3 buckets.

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
    measured_bucket: int,
    observation: bool,
    tpr: float,
    fpr: float,
) -> np.ndarray:
    """
    Update belief after measuring ONE bucket with a binary sensor.
    
    Applies Bayes' rule:
        posterior(x) ∝ likelihood(z|x) × prior(x)
    
    Args:
        belief: Current belief vector [P(A), P(B), P(C)]
        measured_bucket: Index of the bucket being measured (0, 1, or 2)
        observation: True if detection, False if no detection
        tpr: True Positive Rate P(detected | object present)
        fpr: False Positive Rate P(detected | object absent)
        
    Returns:
        Normalized posterior belief vector
    """
    n_buckets = len(belief)
    likelihood = np.zeros(n_buckets)
    
    for hypothesis_idx in range(n_buckets):
        if hypothesis_idx == measured_bucket:
            # Hypothesis: object is in the bucket we just measured
            # P(observation | object here)
            if observation:
                likelihood[hypothesis_idx] = tpr  # True positive
            else:
                likelihood[hypothesis_idx] = 1 - tpr  # False negative
        else:
            # Hypothesis: object is in a DIFFERENT bucket
            # P(observation | object NOT here)
            if observation:
                likelihood[hypothesis_idx] = fpr  # False positive
            else:
                likelihood[hypothesis_idx] = 1 - fpr  # True negative
    
    # Bayes update
    unnormalized_posterior = likelihood * belief
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
    
    return posterior


def bayes_update_all_buckets(
    belief: np.ndarray,
    true_bucket: int,
    tpr: float,
    fpr: float,
) -> np.ndarray:
    """
    Measure ALL buckets and update belief sequentially.
    
    Used during EXECUTION when robot is in the light region.
    Each bucket is measured once, and observations are simulated
    based on the true_bucket (ground truth).
    
    Args:
        belief: Current belief vector [P(A), P(B), P(C)]
        true_bucket: Index of bucket where object actually is (ground truth)
        tpr: True Positive Rate
        fpr: False Positive Rate
        
    Returns:
        Updated belief after measuring all buckets
    """
    n_buckets = len(belief)
    updated_belief = belief.copy()
    
    for bucket_idx in range(n_buckets):
        # Simulate observation for this bucket based on ground truth
        is_object_here = (bucket_idx == true_bucket)
        
        if is_object_here:
            # Object is in this bucket: detection with probability TPR
            observation = np.random.rand() < tpr
        else:
            # Object not in this bucket: false alarm with probability FPR
            observation = np.random.rand() < fpr
        
        # Update belief with this measurement
        updated_belief = bayes_update_single(
            updated_belief, bucket_idx, observation, tpr, fpr
        )
    
    return updated_belief


def expected_posterior_all_buckets(
    belief: np.ndarray,
    tpr: float,
    fpr: float,
    assumed_bucket: int = 0,
) -> np.ndarray:
    """
    Compute EXPECTED posterior after measuring all buckets, assuming the
    object is in a specific bucket.
    
    Used during PLANNING. We assume the object is in `assumed_bucket` and
    compute the expected posterior given that assumption. This allows the
    planner to reason about information gain - if we go to the light region,
    the expected belief will converge toward certainty about `assumed_bucket`.
    
    For each bucket measurement:
    - If measuring the assumed_bucket: expect observation=True with prob TPR
    - If measuring other buckets: expect observation=True with prob FPR
    
    Args:
        belief: Current belief vector [P(A), P(B), P(C)]
        tpr: True Positive Rate
        fpr: False Positive Rate
        assumed_bucket: Which bucket to assume the object is in (for planning)
        
    Returns:
        Expected posterior after measuring all buckets
    """
    n_buckets = len(belief)
    expected_belief = belief.copy()
    
    for measured_bucket in range(n_buckets):
        # Determine if this is the bucket where we assume the object is
        is_assumed_bucket = (measured_bucket == assumed_bucket)
        
        if is_assumed_bucket:
            # We're measuring the bucket where (we assume) the object is
            # Expect positive observation with probability TPR
            p_obs_positive = tpr
        else:
            # We're measuring a bucket where (we assume) the object is NOT
            # Expect positive observation with probability FPR (false alarm)
            p_obs_positive = fpr
        
        p_obs_negative = 1.0 - p_obs_positive
        
        # Compute posterior for each observation outcome
        posterior_if_positive = bayes_update_single(
            expected_belief, measured_bucket, True, tpr, fpr
        )
        posterior_if_negative = bayes_update_single(
            expected_belief, measured_bucket, False, tpr, fpr
        )
        
        # Expected posterior (weighted average given our assumption)
        expected_belief = (
            p_obs_positive * posterior_if_positive +
            p_obs_negative * posterior_if_negative
        )
    
    return expected_belief

