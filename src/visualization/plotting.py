# Visualization functions

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Non-zero coefficients plot

def plot_non_zero_coefs(non_zero_counts, d_features, save_path=None):
    """
    Plot the number of non-zero coefficients over iterations.

    Parameters
    ----------
    non_zero_counts : list of int
        List of non-zero coefficient counts over iterations.

    d_features : int
        Number of features in the latent space.
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


# Intrinsic dimension barplot

import matplotlib.pyplot as plt
import seaborn as sns

def plot_intrinsic_dimension_barplot(
    df_id,
    output_dir=None,
    model_name_mapping=None,
    model_order=None,
    save_path=None,
    verbose=True
):
    """
    Plot intrinsic dimension estimates for different ID estimators of representations of several pre-trained models.

    Parameters
    ----------
    df_id : pd.DataFrame
        DataFrame with columns ['Model', 'Estimator', 'Intrinsic Dimension'].
    output_dir : str, optional
        Directory to save the plot. Used only if save_path is not provided.
    model_name_mapping : dict, optional
        Mapping from model codes to display names. Uses defaults if None.
    model_order : list of str, optional
        List of display names for x-axis order. Uses defaults if None.
    save_path : str, optional
        Full path (including filename) to save the plot. If None, uses output_dir or just displays plot.
    verbose : bool, optional
        Print path when saving. Default True.
    """

    # Seaborn style for white background and grid
    sns.set(style="whitegrid")

    dpi_val = 300
    fontsize_plot = 25

    if model_name_mapping is None:
        model_name_mapping = {
            'densenet121-res224-pc': 'PC',
            'densenet121-res224-chex': 'CheXpert',
            'densenet121-res224-all': 'All',
            'densenet121-res224-nih': 'NIH',
            'densenet121-res224-rsna': 'RSNA'
        }
    if model_order is None:
        model_order = ["PC", "CheXpert", "All", "NIH", "RSNA"]

    df_plot = df_id.copy()
    df_plot['Model'] = df_plot['Model'].map(model_name_mapping)
    df_plot['Model'] = pd.Categorical(df_plot['Model'], categories=model_order, ordered=True)
    df_plot = df_plot.dropna(subset=['Model'])

    plt.figure(figsize=(14, 7), dpi=dpi_val)
    ax = sns.barplot(
        data=df_plot,
        x='Model',
        y='Intrinsic Dimension',
        hue='Estimator',
        palette=sns.color_palette("Set2")
    )

    plt.xlabel('Pre-Trained Representations from Model', fontsize=fontsize_plot)
    plt.ylabel('Intrinsic Dimension Estimates', fontsize=fontsize_plot)
    plt.xticks(fontsize=fontsize_plot, fontweight='bold')
    plt.yticks(fontsize=17)
    plt.legend(title='ID Estimator', fontsize=19, title_fontsize=19)
    plt.tight_layout()
    
    # Save or show plot
    if save_path is None and output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        save_path = f"{output_dir}/intrinsic_dimension_estimates.pdf"

    if save_path:
        plt.savefig(save_path, dpi=dpi_val, bbox_inches='tight', facecolor='white', format='pdf')
        plt.close()
        if verbose:
            print(f"Plot saved as {save_path}")
    else:
        plt.show()