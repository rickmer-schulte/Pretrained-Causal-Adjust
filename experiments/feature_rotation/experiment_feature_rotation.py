import os
import numpy as np
from feature_extraction.pretrained_models_xrv import load_torchxrayvision_model, extract_features_from_folder
from utils.project import set_root

from helpers.feature_rotation import rotate_latent_features
from helpers.linear_probing import train_linear_probe
from visualization.plotting import plot_non_zero_coefs


def analyze_feature_rotation(features, labels, iterations=10, dimension_pairs=5):
    """
    Rotate latent features and perform linear probing iteratively.
    Parameters
    ----------
    features : np.ndarray
        Original latent features of shape (N, D).
    labels : np.ndarray
        Corresponding labels of shape (N,).
    iterations : int
        Number of iterations to rotate and analyze.
    dimension_pairs : int
        Number of dimension pairs to rotate along in each iteration.
    Returns
    -------
    non_zero_counts : list of int
        List of non-zero coefficient counts over iterations.
    """
    non_zero_counts = []
    for i in range(iterations):
        # Train linear probe on current features
        model, _, _, _, _, _ = train_linear_probe(features, labels)
        # Calculate the number of non-zero coefficients
        non_zero_count = np.count_nonzero(model.coef_)
        non_zero_counts.append(non_zero_count)

        features = rotate_latent_features(features, angle_degrees = None, num_pairs=dimension_pairs)
     
    return non_zero_counts


def main():
    # Define directories for saving plots
    set_root()
    plots_dir = os.path.join("results/feature_rotation")
    os.makedirs(plots_dir, exist_ok=True)

    # Define dataset directory
    dataset_dir = "data/pneumonia/test"

    # 1. Load Pre-trained Model
    print("Loading pre-trained torchxrayvision model...")
    model_name = "densenet121-res224-rsna"
    model = load_torchxrayvision_model(model_name)

    # 2. Extract Features from X-ray Dataset
    print(f"Extracting features from dataset at '{dataset_dir}'...")
    all_features, labels = extract_features_from_folder(dataset_dir, model, device='cpu', batch_size=32)
    d_features = all_features.shape[1]
    print(f"Extracted features shape: {all_features.shape}")

    # Number of Zero Coefficients over Random Rotations
    print("Analyzing feature rotation and non-zero coefficients...")
    non_zero_counts = analyze_feature_rotation(all_features, labels, iterations=51, dimension_pairs=100)
    plot_non_zero_coefs(non_zero_counts, d_features,
                        save_path=os.path.join(plots_dir, "non-zero-coefs.pdf"))
    print("Analysis complete. Plots saved.")


if __name__ == "__main__":
    main()