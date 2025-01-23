# DEPENDENCIES
import os
import torch
import numpy as np
from pathlib import Path
import src.config as config
from torchvision import transforms
from src.config import model_preprocessing
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset
from torchvision.datasets import ImageFolder


# DATA LOADER
def get_data_loaders(data_dir:str, model_name:str, batch_size:int=32, train_split:float=0.8, val_split:float=0.1, test_split:float=0.1, num_workers:int=4, subset_size: int = 2000):
    """
    Load datasets from disk and create DataLoader objects for training, validation, and testing
    
    Arguments:
    ----------
        data_dir      { str }   : Path to the directory containing the dataset. It should have subdirectories for each class

        model_name    { str }   : Name of the model for which data needes to be loaded

        batch_size    { int }   : Number of samples per batch

        train_split   { float } : Proportion of data to use for training

        val_split     { float } : Proportion of data to use for validation

        test_split    { float } : Proportion of data to use for testing
        
        num_workers   { int   } : Number of worker threads for data loading
        
    Returns:
    --------
             { dict }           : A dictionary containing DataLoader objects for 'train', 'val', and 'test'.
    """
    # Check if model_name exists in config
    if (model_name not in model_preprocessing):
        raise ValueError(f"Model {model_name} is not available in config.")

    # Retrieve preprocessing settings from model_preprocessing
    preprocess_info = model_preprocessing[model_name]

    assert train_split + val_split + test_split == 1.0, "Splits must sum to 1."

    # Define data augmentation and normalization
    transform_train = transforms.Compose([transforms.Resize(preprocess_info["image_size"]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(10),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = preprocess_info["normalize"]["mean"], std = preprocess_info["normalize"]["std"])
                                        ])

    transform_test  = transforms.Compose([transforms.Resize(preprocess_info["image_size"]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = preprocess_info["normalize"]["mean"], std = preprocess_info["normalize"]["std"])
                                        ])

    # Load dataset from disk
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory {data_dir} does not exist.")

    dataset = ImageFolder(root=data_dir, transform=transform_train)

    # If subset_size is specified, select a random subset of the dataset
    if subset_size is not None:
        indices = np.random.choice(len(dataset), size=subset_size, replace=False)
        dataset = Subset(dataset, indices)

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Apply transform specific to validation and test datasets
    val_dataset.dataset.transform = transform_test
    test_dataset.dataset.transform = transform_test

    # Create DataLoaders
    train_loader = DataLoader(dataset     = train_dataset, 
                              batch_size  = batch_size, 
                              shuffle     = True, 
                              num_workers = num_workers)

    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers)

    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=num_workers)

    data_loaders = {"train": train_loader,
                    "val": val_loader,
                    "test": test_loader
                   }

    return data_loaders