# utils/data_utils.py
import torch
import torchvision.datasets as datasets
import os

def get_calibration_loader(data_path: str,
                           preprocess_transform,
                           num_samples: int = 500,
                           batch_size: int = 32,
                           num_workers: int = 2):
    """
    Creates a DataLoader for calibration using a subset of the CIFAR10 training set.
    """
    print(f"--- Preparing Calibration Data (using CIFAR10 from {data_path}) ---")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created data directory: {data_path}")

    calibration_loader = None
    try:
        # For demonstration, we use CIFAR10. Adapt this for other datasets if needed.
        calibration_dataset_full = datasets.CIFAR10(
            root=data_path,
            train=True,
            download=True,
            transform=preprocess_transform
        )

        # Create a subset for faster calibration
        if num_samples > len(calibration_dataset_full):
            print(f"Warning: Requested {num_samples} calibration samples, but dataset only has {len(calibration_dataset_full)}. Using all.")
            num_samples = len(calibration_dataset_full)
        elif num_samples <= 0:
             print(f"Warning: num_samples ({num_samples}) invalid. Using default 500.")
             num_samples = 500 # Default fallback

        calibration_subset_indices = list(range(num_samples))
        calibration_dataset = torch.utils.data.Subset(calibration_dataset_full, calibration_subset_indices)

        calibration_loader = torch.utils.data.DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle for calibration
            num_workers=num_workers
        )

        print(f"Using {len(calibration_dataset)} images from CIFAR10 for calibration.")
        print(f"Calibration DataLoader created with batch size {batch_size}.")

        # Verify shape
        images, _ = next(iter(calibration_loader))
        print(f"Sample batch tensor shape: {images.shape}, dtype: {images.dtype}")

    except Exception as e:
        print(f"\nError loading or processing calibration data: {e}")
        print("Please ensure network connectivity if downloading, check dataset integrity, or path.")
        calibration_loader = None # Ensure loader is None if it fails

    return calibration_loader