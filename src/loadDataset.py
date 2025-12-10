import torch
import torchvision
from torch.utils.data import Subset, random_split
import os
from . import datasetUtils
from datasets import load_dataset
from huggingface_hub import login


def loadCifar10Subset(root, imagesPerClass, data_transforms):
    # Assuming you have already defined and downloaded your dataset:
    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
    dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=data_transforms) # Use train set for example
    val_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=data_transforms)

    # 1. Initialize storage for selected indices
    selected_indexes = datasetUtils.selectIndexes(dataset, imagesPerClass, 10)
    
    # 4. Create the new subset dataset
    # This new dataset contains exactly 100 non-random images from each class.
    subset_train_dataset = Subset(dataset, selected_indexes)

    print(f"Total images selected: {len(subset_train_dataset)}")

    return subset_train_dataset, dataset, val_dataset
    # Expected output: Total images selected: 1000
    
def loadCifar100Subset(root, imagesPerClass, data_transforms):
    # Assuming you have already defined and downloaded your dataset:
    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
    dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=data_transforms) # Use train set for example
    val_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=data_transforms)

    # 1. Initialize storage for selected indices
    selected_indices = datasetUtils.selectIndexes(dataset, imagesPerClass, 100)
    # 4. Create the new subset dataset
    # This new dataset contains exactly 100 non-random images from each class.
    subset_train_dataset = Subset(dataset, selected_indices)

    print(f"Total images selected: {len(subset_train_dataset)}")

    return subset_train_dataset, dataset, val_dataset
    # Expected output: Total images selected: 1000
    
def loadImagenetA(root, data_transforms, val_split_ratio=0.1):
    # 1. Download and extract the data
    # This calls the function from src/dataset_utils.py to handle the download
    data_dir = datasetUtils.download_imagenet_a(root_dir=root)
    if data_dir is None:
        raise FileNotFoundError("Failed to download or extract ImageNet-A.")

    # 2. Load data using ImageFolder (which expects class subdirectories)
    # This loads all ~7500 images of the ImageNet-A benchmark
    full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=data_transforms)
    
    if not full_dataset.classes:
        raise ValueError(f"Could not find any classes (subdirectories) in {data_dir}. Check the directory structure.")

    # 3. Split the loaded set into a training portion and a validation portion
    dataset_size = len(full_dataset)
    val_size = int(val_split_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    # Use a fixed seed for reproducibility of the split
    g = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=g)

    # Re-attach classes property to the subsets for compatibility
    train_subset.classes = full_dataset.classes
    val_subset.classes = full_dataset.classes
    
    print(f"ImageNet-A loaded: Total images: {dataset_size}")
    print(f"Split into Train/Feature Extraction ({train_size}) and Validation ({val_size}).")
    
    # Return structure matching CIFAR: (subset, full_train_for_compat, val)
    return train_subset, full_dataset, val_subset

def loadHuggingFaceImageNetSubset(root, imagesPerClass, data_transforms, dataset_link="timm/mini-imagenet", num_classes=10, val_split_ratio=0.1):
    print("\n--- Loading ImageNet Subset via Hugging Face ---")
    
    try:
        
        hf_token = datasetUtils.loadToken('token.txt')
        
        login(token=hf_token, add_to_git_credential=False)
        
        # We use 'huggingface/imagenet-100' as a public, manageable proxy for ImageNet-1K
        hf_dataset = load_dataset(dataset_link, split='train', streaming=False)
        
        # Filter down to the desired number of classes for a quick test run
        hf_dataset = hf_dataset.filter(lambda x: x['label'] < num_classes)
        
    except Exception as e:
         raise RuntimeError(f"Failed to load Hugging Face ImageNet subset: {e}")
             
    print(f"Loaded HF Dataset of size: {len(hf_dataset)} (IPC: {len(hf_dataset) // num_classes})")

    # 1. Wrap the Hugging Face Dataset in the custom PyTorch class
    full_dataset = datasetUtils.HuggingFaceImageNetDataset(hf_dataset=hf_dataset, transform=data_transforms)
    
    # 2. Split the wrapped PyTorch dataset
    dataset_size = len(full_dataset)
    val_size = int(val_split_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    g = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=g)

    # 3. Re-attach classes property for compatibility
    train_subset.classes = full_dataset.classes
    val_subset.classes = full_dataset.classes
    
    print(f"ImageNet-HF loaded: Total images: {dataset_size}")
    print(f"Split into Train ({train_size}) and Validation ({val_size}).")
    
    return train_subset, full_dataset, val_subset
    
def getOrCreateDataset(data_dir, imagesPerClass, transform, cache_dir, dataset_name):
    """
    Checks for a cached version of the dataset subset. If found, loads it.
    Otherwise, creates the subset, saves it, and returns it.
    """
    
    # 1. Define the cache file path
    if dataset_name == "cifar10":
        cache_file = os.path.join(cache_dir, f"cifar10_subset_ipc{imagesPerClass}.pt")
    elif dataset_name == "cifar100":
        cache_file = os.path.join(cache_dir, f"cifar100_subset_ipc{imagesPerClass}.pt")
    else:
        cache_file = os.path.join(cache_dir, f"{dataset_name}_ipc{imagesPerClass}.pt")

    if os.path.exists(cache_file):
        print(f"\nLoading cached dataset subset from: {cache_file}")
        try:
            # We save the three parts of the dataset in a dictionary
            cached_data = torch.load(cache_file)
            return cached_data['subset'], cached_data['full_train'], cached_data['val']
        except Exception as e:
            print(f"Error loading cache file: {e}. Rebuilding dataset.")
            os.remove(cache_file) # Delete corrupted cache file

    # 2. If cache doesn't exist or failed to load, create the dataset
    print(f"\nCreating new dataset subset (IPC={imagesPerClass}) and caching it...")
    # Assuming loadDataset is correctly imported from src
    if dataset_name == "cifar10":
        subset, full_train, val = loadCifar10Subset(data_dir, imagesPerClass, transform)
    elif dataset_name == "cifar100":
        subset, full_train, val = loadCifar100Subset(data_dir, imagesPerClass, transform)
    else:
        subset, full_train, val = loadHuggingFaceImageNetSubset(data_dir, imagesPerClass, transform, dataset_name)
    
    # 3. Cache the newly created dataset
    data_to_save = {
        'subset': subset,
        'full_train': full_train,
        'val': val
    }
    torch.save(data_to_save, cache_file)
    print(f"Dataset subset saved successfully to: {cache_file}")
    
    return subset, full_train, val