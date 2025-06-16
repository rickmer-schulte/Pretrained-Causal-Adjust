# Adjustment for Confounding using Pre-Trained Representations

This repository contains the code for the experiments and figures of the **ICML 2025** paper **["Adjustment for Confounding using Pre-Trained Representations"](https://openreview.net/forum?id=p4CHBlYxYj)**.

## Project Structure

```bash
.
│
├── data/ # Raw and processed data
│ └── xray
│   ├── raw/
│   ├── processed/
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
### X-Ray
1. Download the chest X-ray dataset from [Data Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/2) or [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 
2. Run `remove_duplicates.py` to remove mulitple x-rays per patient:
```python
python src/data_setup/remove_duplicates.py
```
3. Run `create_data_xray.py MODEL_NAMES` to extract representations from pre-trained model(s), i.e.:
```python
python src/data_setup/create_data_xray.py densenet121-res224-all 
```

**Details:** Download the Chest X-Ray image dataset from either of the two sources. Unzip the downloaded data file and place the data folders (train/test/val) in the folder `data/xray/raw`. Subsequently run `remove_duplicates.py` to remove mulitple x-rays per patient (otherwise there would be multiple X-rays for certain patients in the dataset). The dataset will be stored at `data/processed/all_unique`. Afterwards run `create_data_xray.py` to extract representations from user-specified pre-trained model(s) for each X-ray image in the dataset. Choose different pre-trained models from the [TorchXRayVision](https://github.com/mlmed/torchxrayvision) library by varying the dataset name in `densenet121-res224-DATASET`. The representations as well the the labels that indicates the presence of the disease pneumomina for each x-ray will be stored at `data/xray/representations`.

### IMDb
1. Run `python src/data_setup/create_data_imdb.py`
  
**Details:** The download and extraction of pre-trained representations for the IMDb data is performed by running `create_data_imdb.py`. This will create a CSV file including the text review, the related pre-trained representations, and sentiment label at `data/imdb/imdb_with_hidden_states_sentiment.csv`.

## Experiments
After the data setup, each experiment can be conducted by running the respective python file or notebook from the `experiments` folder. All results will be stored in a dedicated folder in the `results` folder.

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

