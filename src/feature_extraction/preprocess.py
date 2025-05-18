# preprocess.py
import torchvision.transforms as transforms

def get_preprocess_pipeline():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])