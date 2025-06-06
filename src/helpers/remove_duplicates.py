import os
import re
import shutil

def process_normal_folder(normal_folder_path):
    """
    Processes the 'all_unique/NORMAL' folder to ensure each unique person appears only once.
    """
    seen_ids = set()
    files = sorted(os.listdir(normal_folder_path))
    for filename in files:
        if not filename.lower().endswith('.jpeg'):
            continue  # Skip non-jpeg files
        match = re.search(r'IM-(\d{4})', filename)
        if match:
            unique_id = match.group(1)
            if unique_id in seen_ids:
                file_path = os.path.join(normal_folder_path, filename)
                os.remove(file_path)
                print(f"Removed duplicate NORMAL image: {filename}")
            else:
                seen_ids.add(unique_id)
        else:
            print(f"Filename does not match expected pattern: {filename}")

def process_pneumonia_folder(pneumonia_folder_path):
    """
    Processes the 'all_unique/PNEUMONIA' folder to ensure each unique person appears only once.
    """
    seen_ids = set()
    files = sorted(os.listdir(pneumonia_folder_path))
    for filename in files:
        if not filename.lower().endswith('.jpeg'):
            continue  # Skip non-jpeg files
        parts = filename.split('_')
        if len(parts) >= 3:
            unique_id = f"{parts[0]}_{parts[1]}"
            if unique_id in seen_ids:
                file_path = os.path.join(pneumonia_folder_path, filename)
                os.remove(file_path)
                print(f"Removed duplicate PNEUMONIA image: {filename}")
            else:
                seen_ids.add(unique_id)
        else:
            print(f"Filename does not match expected pattern: {filename}")

def main():
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    original_folder = os.path.join(BASE_PATH, 'data/xray/raw/all')
    unique_folder = os.path.join(BASE_PATH, 'data/xray/raw/all_unique')

    if not os.path.exists(original_folder):
        print(f"Original folder not found: {original_folder}")
        return

    if os.path.exists(unique_folder):
        shutil.rmtree(unique_folder)

    shutil.copytree(original_folder, unique_folder)

    normal_folder = os.path.join(unique_folder, 'NORMAL')
    pneumonia_folder = os.path.join(unique_folder, 'PNEUMONIA')

    print(f"Processing {normal_folder} folder...")
    process_normal_folder(normal_folder)
    print(f"Finished processing {normal_folder} folder.\n")

    print(f"Processing {pneumonia_folder} folder...")
    process_pneumonia_folder(pneumonia_folder)
    print(f"Finished processing {pneumonia_folder} folder.\n")

    print("Duplicate removal completed successfully.")

if __name__ == "__main__":
    main()