import os
import pickle

def save_results(run_dir, est_dict, ci_dict):
    """Save estimates and confidence intervals as pickle files."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'estimates_dict.pkl'), 'wb') as f:
        pickle.dump(est_dict, f)
    with open(os.path.join(run_dir, 'cis_dict.pkl'), 'wb') as f:
        pickle.dump(ci_dict, f)

def load_results(run_dir):
    """Load estimates and confidence intervals from pickle files."""
    with open(os.path.join(run_dir, 'estimates_dict.pkl'), 'rb') as f:
        est_dict = pickle.load(f)
    with open(os.path.join(run_dir, 'cis_dict.pkl'), 'rb') as f:
        ci_dict = pickle.load(f)
    return est_dict, ci_dict

def save_pickle(obj, filename):
    """Generic save pickle utility."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    """Generic load pickle utility."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
