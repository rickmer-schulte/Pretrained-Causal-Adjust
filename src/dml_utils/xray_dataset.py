import os
from PIL import Image
from torch.utils.data import Dataset

# Define the XRayDataset
class XRayDataset(Dataset):
    """
    Dataset for loading chest X-ray images from the pneumonia dataset
    having a folder structure with subfolders 'NORMAL' and 'PNEUMONIA'.
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []  # Each element is (file_path, label, filename)
        self.categories = ['NORMAL', 'PNEUMONIA']
        self.transform = transform
        
        print(f"Loading images from: {root_dir}")
        for cat in self.categories:
            cat_dir = os.path.join(root_dir, cat)
            if not os.path.isdir(cat_dir):
                print(f"Warning: Category folder '{cat_dir}' not found.")
                continue
            print(f"Scanning category: {cat}")
            for fname in os.listdir(cat_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(cat_dir, fname)
                    self.samples.append((file_path, cat, fname))
                else:
                    print(f"Skipping non-image file: {fname}")
        print(f"Found {len(self.samples)} samples in '{root_dir}'.")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label, fname = self.samples[idx]
        # Open the image and convert to grayscale (1 channel)
        image = Image.open(file_path).convert("L")
        if self.transform:
            image = self.transform(image)
        # Return image tensor, label string, and original filename
        return image, label, fname