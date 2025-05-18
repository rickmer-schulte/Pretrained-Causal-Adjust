import os
import torchxrayvision as xrv
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def load_torchxrayvision_model(model_name="densenet121-res224-all"):
    """
    Load a pretrained model from torchxrayvision.

    Parameters
    ----------
    model_name : str
        Name of the model to load.

    Returns
    -------
    model : torch.nn.Module
        Pretrained torchxrayvision model.
    """
    model = xrv.models.get_model(model_name)
    model.eval()  # Set to evaluation mode
    return model

def preprocess_image(img):
    """
    Preprocess a single x-ray image for the torchxrayvision model.

    Parameters
    ----------
    img : np.ndarray
        The x-ray image array.

    Returns
    -------
    torch.Tensor
        Preprocessed image tensor.
    """
    img = xrv.datasets.normalize(img, 255)  # Convert to [-1024, 1024]
    if img.ndim == 2:  # Grayscale check
        img = img[:, :, np.newaxis]
    img = img.mean(2)[None, ...]  # Single color channel
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = transform(img)
    return torch.from_numpy(img)

def extract_features_from_folder(folder_path, model, device='cpu', batch_size=32, save_path=None):
    """
    Extract features from all images in a folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing subfolders "NORMAL" and "PNEUMONIA".
    model : torch.nn.Module
        Pretrained torchxrayvision model.
    device : str
        Device to use ('cpu' or 'cuda').
    batch_size : int
        Batch size for processing.
    save_path : str or None
        Path to save the features. If None, features are not saved.

    Returns
    -------
    features : np.ndarray
        Latent features for all images.
    labels : np.ndarray
        Corresponding labels (0 for NORMAL, 1 for PNEUMONIA).
    """
    from pathlib import Path
    from PIL import Image

    categories = {"NORMAL": 0, "PNEUMONIA": 1}
    data = []
    labels = []

    for category, label in categories.items():
        category_path = Path(folder_path) / category
        for img_path in category_path.glob("*.jpeg"):
            img = np.array(Image.open(img_path))
            img = preprocess_image(img)
            data.append(img)
            labels.append(label)

    data = torch.stack(data)  # (N, 1, H, W)
    labels = np.array(labels)

    model.to(device)
    data = data.to(device)

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0]
            features = model.features2(images)
            all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        features_path = os.path.join(save_path, "latent_features.npy")
        labels_path = os.path.join(save_path, "labels.npy")
        np.save(features_path, all_features)
        np.save(labels_path, labels)
        print(f"Saved features and labels to '{save_path}'.")

    return all_features, labels