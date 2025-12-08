import numpy as np
import matplotlib.pyplot as plt


def calculate_entropy(belief_vector):
    """Calculates the Shannon Entropy (in bits) of a belief vector."""
    
    # 1. Filter out zeros, because log(0) is undefined. 
    # (P(x) * log(P(x)) approaches 0 as P(x) approaches 0)
    probabilities = belief_vector[belief_vector > 0]
    
    # 2. Apply the formula: H = - sum(P * log2(P))
    # np.log2 is the logarithm base 2
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def calculate_misclassification_risk(belief_vector: np.ndarray) -> float:
    """
    Implements the Misclassification Risk Cost, which measures the probability 
    that the true state is NOT the Maximum A Posteriori (MAP) estimate.

    Cost = 1 - P(x_MAP)

    Args:
        belief_vector: A NumPy array representing the categorical belief state 
                       [P(x_A), P(x_B), P(x_C), ...].

    Returns:
        The Misclassification Risk Cost (a float between 0.0 and 1.0).
    """
    
    # 1. Find the probability of the Maximum A Posteriori (MAP) estimate
    # This is the highest probability in the belief vector.
    max_probability = np.max(belief_vector)
    
    # 2. Calculate the risk: 1 minus the probability of the best guess
    misclassification_risk = 1.0 - max_probability
    
    return misclassification_risk

def run_bayes_simulation():
    # ---------------------------------------------------------
    # 1. Setup & Initialization
    # ---------------------------------------------------------
    buckets = ['A', 'B', 'C']
    n_buckets = len(buckets)
    
    # Maximum Entropy Prior (Uniform Distribution)
    # Belief[i] = P(Item is in Bucket i)
    belief = np.array([1.0/n_buckets] * n_buckets)
    
    # Hidden State: Randomly instantiate the problem
    true_bucket_idx = np.random.randint(0, n_buckets)
    true_bucket_name = buckets[true_bucket_idx]
    
    # ---------------------------------------------------------
    # 2. Define Measurement / Sensor Model
    # ---------------------------------------------------------
    # True Positive Rate (Sensitivity): P(Detected | Item Present)
    TPR = 0.80 
    # False Positive Rate (False Alarm): P(Detected | Item Absent)
    FPR = 0.15 
    
    # ---------------------------------------------------------
    # 3. Simulation Loop: Collect N Measurements
    # ---------------------------------------------------------
    N = 6  # Number of measurements to collect
    
    print(f"--- SIMULATION START ---")
    print(f"Hidden Truth: The item is actually in Bucket {true_bucket_name}")
    print(f"Prior Belief: {belief}\n")
    
    step = 0
    while calculate_misclassification_risk(belief) > 0.01:
        step += 1
        # A. Action: Decide which bucket to measure
        # Strategy: Randomly pick a bucket to scan
        measure_idx = np.random.randint(0, n_buckets)
        measure_name = buckets[measure_idx]
        
        # B. Observation: Simulate the sensor reading
        # Determine if the item is actually in the bucket we are measuring
        is_item_present = (measure_idx == true_bucket_idx)
        
        if is_item_present:
            # We are scanning the correct bucket.
            # Sensor returns Positive with probability TPR.
            observation = np.random.rand() < TPR
        else:
            # We are scanning an empty bucket.
            # Sensor returns Positive with probability FPR (Noise).
            observation = np.random.rand() < FPR
            
        obs_str = "DETECTED" if observation else "NOTHING"
        
        # C. Update: Apply Bayes' Rule
        # Posterior(x) = Likelihood(z|x) * Prior(x) (normalized)
        
        likelihood_vector = np.zeros(n_buckets)
        
        for hypothesis_idx in range(n_buckets):
            # We are evaluating: P(Observation | Item is in hypothesis_idx)
            
            if hypothesis_idx == measure_idx:
                # Hypothesis: The item is in the bucket we just scanned.
                # So we expect to see it (TPR logic).
                if observation: # We saw it
                    likelihood_vector[hypothesis_idx] = TPR
                else:           # We missed it
                    likelihood_vector[hypothesis_idx] = 1 - TPR
            else:
                # Hypothesis: The item is in a DIFFERENT bucket than the one we scanned.
                # So the bucket we scanned should be empty (FPR logic).
                if observation: # False Positive
                    likelihood_vector[hypothesis_idx] = FPR
                else:           # True Negative
                    likelihood_vector[hypothesis_idx] = 1 - FPR
        
        # Update and Normalize
        unnormalized_posterior = likelihood_vector * belief
        belief = unnormalized_posterior / np.sum(unnormalized_posterior)
        
        # Log step
        print(f"Step {step}: Scanned Bucket {measure_name} -> Got '{obs_str}'")
        print(f"   New Belief:  {np.round(belief, 4)}")
        print(f"   Likelihoods: {likelihood_vector}")
        print(f"   Entropy:    {calculate_entropy(belief):.4f} bits")
        print(f"   Misclassification Risk: {calculate_misclassification_risk(belief):.4f}")
        print("-" * 40)
    
    # ---------------------------------------------------------
    # 4. Plotting the Posterior
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    # Create bar chart
    bars = plt.bar(buckets, belief, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
    
    # Highlight the true bucket in the plot logic or just title
    plt.title(f'Posterior Belief after {N} Measurements\nTrue Location: Bucket {true_bucket_name}', fontsize=14)
    plt.ylabel('Probability', fontsize=12)
    plt.xlabel('Bucket Location', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig('bayes_posterior_buckets.png')

if __name__ == "__main__":
    run_bayes_simulation()