import numpy as np

def sample_npz(input_npz, output_npz, sample_ratio=0.1, seed=42):
    """
    Randomly sample a subset of data from an existing .npz file and save it to a new .npz file.
    Args:
        input_npz (str): Path to the input .npz file.
        output_npz (str): Path to the output .npz file.
        sample_ratio (float): Proportion of data to sample (0 < sample_ratio <= 1).
        seed (int): Random seed for reproducibility.
    """
    print(f"Loading data from {input_npz}...")
    data = np.load(input_npz)
    X = data["hidden"]  # Hidden representations
    Y = data["pos"]     # POS labels

    print(f"Original dataset size: {X.shape[0]} samples.")

    # Random sampling
    np.random.seed(seed)
    indices = np.random.choice(len(X), int(len(X) * sample_ratio), replace=False)
    X_sampled = X[indices]
    Y_sampled = Y[indices]

    print(f"Sampled dataset size: {X_sampled.shape[0]} samples.")

    # Save sampled data to a new .npz file
    print(f"Saving sampled data to {output_npz}...")
    np.savez(output_npz, hidden=X_sampled, pos=Y_sampled)
    print("Sampling complete.")

def main():
    input_npz = "hidden_representations.npz"  # Input file path
    output_npz = "sampled_hidden_representations.npz"  # Output file path
    sample_ratio = 0.1  # Proportion of data to sample (e.g., 10%)
    seed = 42  # Random seed for reproducibility

    sample_npz(input_npz, output_npz, sample_ratio, seed)

if __name__ == "__main__":
    main()
