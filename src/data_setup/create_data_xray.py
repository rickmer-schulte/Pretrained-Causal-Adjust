import os
import numpy as np
import argparse
from utils.project import set_root
from feature_extraction.pretrained_models_xrv import load_torchxrayvision_model, extract_features_from_folder

def extract_and_save_features(model_names, base_dir="data/xray/representations", dataset_dir="data/xray/processed/all_unique", verbose=True):
    for model_name in model_names:
        save_dir = os.path.join(base_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)
        features_path = os.path.join(save_dir, "latent_features.npy")
        labels_path = os.path.join(save_dir, "labels.npy")

        # Extract and save features if not already existing
        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            if verbose:
                print(f"Extracting features for {model_name} ...")
            model = load_torchxrayvision_model(model_name)
            all_features, labels = extract_features_from_folder(
                dataset_dir, model, device='cpu', batch_size=32, save_path=save_dir
            )
            np.save(features_path, all_features)
            np.save(labels_path, labels)
        else:
            if verbose:
                print(f"Features and labels already exist for {model_name}. Skipping extraction.")

        features = np.load(features_path)
        if verbose:
            print(f"Loaded features for model: {model_name} with shape {features.shape}")

if __name__ == "__main__":
    set_root()
    parser = argparse.ArgumentParser(description="Extract and save pre-trained representations from specified models.")
    parser.add_argument('model_names', nargs='+', help='List of pre-trained model names to process')
    parser.add_argument('--base_dir', default="data/xray/representations", help="Base directory for storing representations")
    parser.add_argument('--dataset_dir', default="data/xray/processed/all_unique", help="Directory containing processed all_unique dataset")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help="Disable verbose output (default: verbose is True)")
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
    extract_and_save_features(args.model_names, args.base_dir, args.dataset_dir, args.verbose)