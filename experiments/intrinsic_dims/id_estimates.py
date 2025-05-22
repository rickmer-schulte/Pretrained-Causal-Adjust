import os
import numpy as np
import pandas as pd
from skdim import id
from visualization.plotting import plot_intrinsic_dimension_barplot
from feature_extraction.pretrained_models_xrv import load_torchxrayvision_model, extract_features_from_folder

def main(
    base_dir="data/representations/pneumonia",
    dataset_dir="data/pneumonia",
    output_dir="results/intrinsic_dimensions",
    verbose=True
):
    model_names = [
        "densenet121-res224-pc",
        "densenet121-res224-chex",
        "densenet121-res224-all",
        "densenet121-res224-nih",
        "densenet121-res224-rsna"
    ]
    if verbose:
        print("Processing models:", model_names)

    results = []
    estimators = {
        'MLE': id.MLE(),
        'ESS': id.ESS(),
        'lPCA': id.lPCA()
    }

    for model_name in model_names:
        save_dir = os.path.join(base_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)
        features_path = os.path.join(save_dir, "latent_features.npy")
        labels_path = os.path.join(save_dir, "labels.npy")

        # Check if features exist, otherwise extract and save them
        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            if verbose:
                print(f"Extracting features for {model_name} ...")
            model = load_torchxrayvision_model(model_name)
            all_features, labels = extract_features_from_folder(
                dataset_dir, model, device='cpu', batch_size=32, save_path=save_dir
            )
            np.save(features_path, all_features)
            np.save(labels_path, labels)
            if verbose:
                print(f"Saved features and labels in {save_dir}")
        else:
            if verbose:
                print(f"Features and labels already exist for {model_name}. Skipping extraction.")

        features = np.load(features_path)
        if verbose:
            print(f"Loaded features for model: {model_name} with shape {features.shape}")

        # Estimate intrinsic dimensions using different estimators
        if verbose:
            print(f"Estimating intrinsic dimensions for {model_name}...")
        model_results = {'Model': model_name}
        for est_name, estimator in estimators.items():
            try:
                if est_name == 'lPCA':
                    id_estimator = estimator.fit_pw(features, n_neighbors=50)
                    mean_id = np.mean(id_estimator.dimension_pw_)
                elif est_name == 'ESS':
                    id_estimator = estimator.fit(features, n_neighbors=25)
                    mean_id = np.mean(id_estimator.dimension_pw_)
                elif est_name == 'MLE':
                    id_estimator = estimator.fit(features, n_neighbors=5)
                    mean_id = np.mean(id_estimator.dimension_pw_)
                else:
                    raise ValueError(f"Estimator {est_name} not found.")

                model_results[est_name] = mean_id
                if verbose:
                    print(f"Estimated ID for {model_name} using {est_name}: {mean_id:.2f}")
            except Exception as e:
                if verbose:
                    print(f"Error estimating ID for {model_name} using {est_name}: {e}")
                model_results[est_name] = np.nan

        results.append(model_results)

    df_int_dim = pd.DataFrame(results)
    df_int_dim_melted = df_int_dim.melt(id_vars='Model', var_name='Estimator', value_name='Intrinsic Dimension')

    # Plot and save the results
    plot_intrinsic_dimension_barplot(df_int_dim_melted, output_dir="results/intrinsic_dimensions")
    if verbose:
        print(f"Results saved to {output_dir}")

if __name__ == "__main__":

    main()
