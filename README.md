# Adjustment for Confounding using Pre-Trained Representations

This repository contains the code for the experiments and figures of the **ICML 2025** paper **["Adjustment for Confounding using Pre-Trained Representations"](https://openreview.net/forum?id=p4CHBlYxYj)**.

## Project Structure

```bash
.
│
├── data/ # Raw and processed data
│ └── xray
│   ├── raw/
│   └── representations/
│ └── imdb/
│
├── experiments/ # Scripts and notebooks for experiments
│ ├── xray/ # X-Ray: Label and complex confounding experiments
│ ├── imdb/ # IMDb: Label and complex confounding experiments
│ ├── dml_cnn/ # DML with/out pre-training 
│ ├── intrinsic_dims/ # Intrinsic dimension estimation
│ ├── feature_rotation/ # Noninvariance of sparsity to feature rotations
│ ├── asym_norm/ # Asymptototic normality experiments 
│ └── hcm/ # Exploring HCM
│
├── results/ # Folder for all generated results
│
├── src/
│ ├── feature_extraction/ # Scripts extracting pre-trained representations
│ ├── dml_utils/ # Modules for DML (CNN)
│ ├── helpers/ # Specfic helper functions experiments
│ ├── utils/ # General utility functions
│ └── visualization/ # Plotting functions
│
├── pyproject.toml # Project config
└── ...
```

## Python Setup
```python
git clone
cd pretrained-causal-adjust
# Possibly create a new virtual environment and activate it
poetry install --no-root
```

## Citation

```bibtex
@inproceedings{
  schulte2025adjustment,
  title={Adjustment for Confounding using Pre-Trained Representations},
  author={Schulte, Rickmer and R{\"u}gamer, David and Nagler, Thomas},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025},
  publisher={PMLR},
}

