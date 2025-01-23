import os
import random
import shutil
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def stratified_shuffle_split(data_folder: str, target_folder: str, num_images: int):
    class_folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    
    image_paths = []
    image_labels = []
    
    for idx, class_folder in enumerate(class_folders):
        class_path = os.path.join(data_folder, class_folder)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
        
        for image in image_files:
            image_paths.append(os.path.join(class_path, image))
            image_labels.append(idx) 

    
    image_paths = np.array(image_paths)
    image_labels = np.array(image_labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=num_images, random_state=42)

    for train_index, test_index in sss.split(image_paths, image_labels):
        sampled_image_paths = image_paths[test_index]
    
    for image_path in sampled_image_paths:
        class_folder = os.path.basename(os.path.dirname(image_path))
        target_class_folder = os.path.join(target_folder, class_folder)
        Path(target_class_folder).mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, target_class_folder)
        print(f"Copied {image_path} to {target_class_folder}")

if __name__ == "__main__":
    data_folder = '../data' 
    target_folder = './stratified_data_googlenet' 
    num_images = 2000

    stratified_shuffle_split(data_folder, target_folder, num_images)
