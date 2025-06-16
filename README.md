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
│ ├── asym_norm/ # Asymptotic normality experiments 
│ └── hcm/ # Exploring HCM
│
├── results/ # Folder for all generated results
│
├── src/
│ ├── data_setup/ # Scripts for data creation and preprocessing
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

## Data Setup
#### X-Ray
1. Download the chest X-ray dataset from [Data Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/2) or [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 
2. Run `remove_duplicates.py` to remove mulitple x-rays per patient
3. Run `create_data_xray.py` to extract pre-trained representations 

Download the Chest X-Ray image dataset from either of the two sources. Unzip the downloaded data file and place the data folders (train/test/val) in the folder `data/xray/raw`. Subsequently run `src/data_setup/remove_duplicates.py` to remove mulitple x-rays per patient (otherwise there would be multiple X-rays for certain patients in the dataset). The dataset will be stored at `data/processed/all_unique`. Afterwards run `src/data_setup/create_data_xray.py` to extract representations from a user-specified pre-trained model for each X-ray image in the dataset. The label that indicates the presence of the disease pneumomina for each x-ray will also be stored. The data will be stored in `data/xray/representations`.

#### IMDb
1. Run `create_data_imdb.py`
  
The download and extraction of pre-trained representations for the IMDb data is performed by running the script at `src/data_setup/create_data_imdb.py`. This will create a CSV file including the text review, the related pre-trained representations, and sentiment label at `data/imdb/imdb_with_hidden_states_sentiment.csv`.

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

