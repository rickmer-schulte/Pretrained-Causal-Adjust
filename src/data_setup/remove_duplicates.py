import os
import re
import shutil
from collections import defaultdict
from utils.project import set_root

RAW_DATA_DIR = "data/xray/raw"
ALL_DATA_DIR = "data/xray/processed/all"
PROCESSED_DIR = "data/xray/processed/all_unique"

def collect_images(input_dirs):
    images = defaultdict(list) 
    for folder in input_dirs:
        for label in ["NORMAL", "PNEUMONIA"]:
            label_dir = os.path.join(folder, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(".jpeg"):
                    images[label].append((os.path.join(label_dir, fname), fname))
    return images

def copy_all_images(images, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for label, files in images.items():
        label_target_dir = os.path.join(target_dir, label)
        os.makedirs(label_target_dir, exist_ok=True)
        for src, fname in files:
            shutil.copy2(src, os.path.join(label_target_dir, fname))

def remove_duplicates_and_copy(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for label in ["NORMAL", "PNEUMONIA"]:
        label_source_dir = os.path.join(source_dir, label)
        label_target_dir = os.path.join(target_dir, label)
        os.makedirs(label_target_dir, exist_ok=True)
        seen_ids = set()
        for fname in sorted(os.listdir(label_source_dir)):
            src = os.path.join(label_source_dir, fname)
            if label == "NORMAL":
                match = re.search(r'IM-(\d{4})', fname)
                unique_id = match.group(1) if match else fname
            elif label == "PNEUMONIA":
                parts = fname.split('_')
                unique_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 3 else fname
            else:
                continue

            if unique_id in seen_ids:
                print(f"Duplicate {label} image removed: {fname}")
                continue

            seen_ids.add(unique_id)
            shutil.copy2(src, os.path.join(label_target_dir, fname))


if __name__ == "__main__":
    set_root()
    data_folders = [os.path.join(RAW_DATA_DIR, f) for f in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, f))]
    images = collect_images(data_folders)

    # Copy all images into the aggregated folder
    copy_all_images(images, ALL_DATA_DIR)

    # Remove duplicates from aggregated folder
    remove_duplicates_and_copy(ALL_DATA_DIR, PROCESSED_DIR)

    print("Finished processing images. Unique images are stored in:", PROCESSED_DIR)