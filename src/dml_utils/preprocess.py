# preprocess.py
import torchvision.transforms as transforms

# Define the preprocessing pipeline for DML models
def dml_preprocess_pipeline():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])