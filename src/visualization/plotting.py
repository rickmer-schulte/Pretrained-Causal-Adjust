# Visualization functions

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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
        os.makedirs(output_dir, exist_ok=True)
        save_path = f"{output_dir}/intrinsic_dimension_estimates.pdf"

    if save_path:
        plt.savefig(save_path, dpi=dpi_val, bbox_inches='tight', facecolor='white', format='pdf')
        plt.close()
        if verbose:
            print(f"Plot saved as {save_path}")
    else:
        plt.show()


# ATE estimates plot

def plot_ate_estimates(
    estimates_dict,
    cis_dict,
    plot_name,
    save_dir,
    ate_true=2.0,
    n_runs=5,
    figsize=(14, 7),
    vert_diff=0.03,
    label_break=True,
    verbose=True
):
    """
    Plot ATE estimates with confidence intervals across multiple estimation methods.

    Parameters
    ----------
    estimates_dict : dict
        Dictionary with methods as keys and lists of ATE estimates as values.

    cis_dict : dict
        Dictionary with methods as keys and another dict {'lower': list, 'upper': list} as values for CIs.

    plot_name : str
        Name of the plot file (without extension).

    save_dir : str
        Directory to save the plot.

    ate_true : float, optional
        True ATE value for reference line. Default is 2.0.

    n_runs : int, optional
        Number of experiment runs per method. Default is 5.

    figsize : tuple, optional
        Size of the plot figure. Default is (14, 7).

    vert_diff : float, optional
        Vertical offset for the true ATE label. Default is 0.03.

    label_break : bool, optional
        Whether to add line breaks in method labels for better readability. Default is True.

    verbose : bool, optional
        Whether to show the saved plot. Default is True.

    Returns
    -------
    None
    """
    # Extract methods from estimates_dict
    methods = list(estimates_dict.keys())

    # Generate method labels with line breaks
    if label_break == True:
        methods_labels = [method if '(' not in method else method.replace(' (', '\n(') for method in methods]
    else:
        methods_labels = methods

    # Define estimator colors lookup
    estimator_colors = {
        "Naive": "#4D4D4D", 
        "Oracle": "#145A32", 
        "S-Learner (Linear)": "#8B0000", "S-Learner (NN)": "#8B0000", 
        "S-Learner (RF)": "#CC5500",  
        "S-Learner (Lasso)": "#FFA07A",
        "DML (Linear)": "#003366", "DML (Pre-trained)": "#003366", "DML (NN)": "#003366", 
        "DML (RF)": "#4682B4", "DML (CNN)": "#4682B4",
        "DML (Lasso)": "#87CEEB",  
    }

    # Aggregate data into DataFrame
    estimates_df = pd.DataFrame(estimates_dict)
    cis_lower_df = pd.DataFrame({method: cis_dict[method]['lower'] for method in methods})
    cis_upper_df = pd.DataFrame({method: cis_dict[method]['upper'] for method in methods})

    plot_data = []
    for method in methods:
        for run in range(n_runs):
            estimate = estimates_df.loc[run, method]
            if not np.isnan(estimate):
                plot_data.append({
                    'Method': method,
                    'Run': run + 1,
                    'Estimate': estimate,
                    'CI Lower': cis_lower_df.loc[run, method],
                    'CI Upper': cis_upper_df.loc[run, method]
                })

    plot_df = pd.DataFrame(plot_data)

    # Plot setup
    sns.set(style="whitegrid")
    plt.figure(figsize=figsize, facecolor='white')
    ax = plt.gca()

    fontsize_plot = 25
    method_positions = {method: idx for idx, method in enumerate(methods)}

    # Offsets for clearer visualization
    offset_range = 0.2
    offsets = np.linspace(-offset_range, offset_range, n_runs)

    # Plot data points and CIs
    for _, row in plot_df.iterrows():
        method = row['Method']
        run = row['Run']
        x = method_positions[method] + offsets[run - 1]
        estimate = row['Estimate']
        ci_lower = row['CI Lower']
        ci_upper = row['CI Upper']
        color = estimator_colors.get(method, "black") 

        ax.errorbar(
            x, estimate, 
            yerr=[[estimate - ci_lower], [ci_upper - estimate]],
            fmt='none', ecolor=color, elinewidth=4, capsize=5, alpha=0.8
        )
        ax.plot(x, estimate, marker='o', markerfacecolor='white', markeredgecolor=color, markersize=10)

    # True ATE reference line
    plt.axhline(y=ate_true, color='red', linestyle='--', linewidth=3)
    plt.text(x=-0.3, y=ate_true + vert_diff, s="True ATE", fontsize=19, color='red')

    plt.xticks(ticks=list(method_positions.values()), labels=methods_labels, fontsize=fontsize_plot)
    plt.yticks(fontsize=18)
    plt.ylabel("Estimated ATE", fontsize=fontsize_plot)

    plt.tight_layout()

    # Save plot
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = f"{save_dir}/{plot_name}.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        if verbose:
            print(f"Plot saved at {plot_path}")

    if verbose:
        plt.show()
    else:
        plt.close()
