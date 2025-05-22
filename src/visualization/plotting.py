# Visualization functions

import matplotlib.pyplot as plt
import seaborn as sns

def plot_non_zero_coefs(non_zero_counts, d_features, save_path=None):
    """
    Plot the number of non-zero coefficients over iterations.

    Parameters
    ----------
    non_zero_counts : list of int
        List of non-zero coefficient counts over iterations.

    d_features : int
        Number of features in the latent space.

    Returns
    -------
    None
    """
    plt.style.use("default")  
    dpi_val = 300  

    plt.figure(figsize=(10, 4), dpi=dpi_val) 
    plt.hlines(y=d_features, xmin=0, xmax=len(non_zero_counts)-1, colors='dimgrey', linestyles='dashed')
    plt.text(x=1, y=d_features - 0.05 * d_features, s="Latent Feature Dimension", fontsize=12, color='black')
    sns.lineplot(x=range(len(non_zero_counts)), y=non_zero_counts, marker="o", linewidth=2, markersize=8)
    plt.xlabel("No. of random feature rotations", fontsize=16)
    plt.ylabel("Non-zero coefficients", fontsize=16)
    plt.grid(linestyle='--', alpha=0.7)
    plt.xlim(min(range(len(non_zero_counts))) - 1, max(range(len(non_zero_counts))) + 1)
    plt.tight_layout(pad=2)
        
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()