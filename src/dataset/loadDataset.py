import torch
import torchvision
from torch.utils.data import Subset, random_split
import os
from . import datasetUtils
from datasets import load_dataset
from huggingface_hub import login
from .. import plot

def loadCifar10Subset(root, total_images, data_transforms):
    # Assuming you have already defined and downloaded your dataset:
    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
    dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=data_transforms) # Use train set for example
    val_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=data_transforms)
    
    imagesPerClass = total_images // 10

    # 1. Initialize storage for selected indices
    selected_indices = datasetUtils.selectindices(dataset, imagesPerClass, 10)
    
    # 4. Create the new subset dataset
    # This new dataset contains exactly 100 non-random images from each class.
    subset_train_dataset = Subset(dataset, selected_indices)

    print(f"Total images selected: {len(subset_train_dataset)}")

    return subset_train_dataset, dataset, val_dataset, selected_indices
    # Expected output: Total images selected: 1000
    
def loadCifar100Subset(root, total_images, data_transforms):
    # Assuming you have already defined and downloaded your dataset:
    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
    dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=data_transforms) # Use train set for example
    val_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=data_transforms)

    imagesPerClass = total_images // 100

    # 1. Initialize storage for selected indices
    selected_indices = datasetUtils.selectindices(dataset, imagesPerClass, 100)
    # 4. Create the new subset dataset
    # This new dataset contains exactly 100 non-random images from each class.
    subset_train_dataset = Subset(dataset, selected_indices)

    print(f"Total images selected: {len(subset_train_dataset)}")

    return subset_train_dataset, dataset, val_dataset, selected_indices
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

def loadHuggingFaceDataset(root, total_images, data_transforms, dataset_link="timm/mini-imagenet", num_classes=10, val_split_ratio=0.1):
    print("\n--- Loading ImageNet Subset via Hugging Face ---")
    
    images_per_class = total_images // num_classes
    
    try:
        
        hf_token = datasetUtils.loadToken('token.txt')
        
        login(token=hf_token, add_to_git_credential=False)
        
        hf_train = load_dataset(dataset_link, split='train', streaming=False)
        hf_validation = load_dataset(dataset_link, split='validation', streaming=False)
        
    except Exception as e:
         raise RuntimeError(f"Failed to load Hugging Face ImageNet subset: {e}")
             
    print(f"Loaded HF Dataset of size: {len(hf_train) + len(hf_validation)}")
    
    print(f"\nCreating subset with {images_per_class} images per class for {num_classes} classes.")
        
    train_selected_indices = datasetUtils.getRandomImages(num_classes, images_per_class, hf_train, hf_train.features['label'].num_classes)
    val_selected_indices = datasetUtils.getRandomImages(num_classes, images_per_class, hf_validation, hf_train.features['label'].num_classes)

    hf_train = hf_train.select(train_selected_indices)
    hf_validation = hf_validation.select(val_selected_indices)

    print(f"Loaded {num_classes} classes * {images_per_class} images per class")

    # 5. Wrap in PyTorch Dataset
    train_dataset = datasetUtils.HuggingFaceImageNetDataset(hf_dataset=hf_train, transform=data_transforms)
    val_dataset = datasetUtils.HuggingFaceImageNetDataset(hf_dataset=hf_validation, transform=data_transforms)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    return train_dataset, train_selected_indices, val_dataset, val_selected_indices
    
def getOrCreateDataset(data_dir, total_images, num_classes, transform, cache_dir, dataset_name, subset_num=0, output_dir="./dataStorage"):
    """
    Checks for a cached version of the dataset subset. If found, loads it.
    Otherwise, creates the subset, saves it, and returns it.
    """
    num_classes = 10 if dataset_name == "cifar10" else num_classes
    num_classes = 100 if dataset_name == "cifar100" else num_classes
    
    # 1. Define the cache file path
    dt_name = dataset_name.replace('/', '-') #remove diretório na hora de buscar o arquivo, existe ao ser um link do HuggingFace
    file_name = f"{dt_name}_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    cache_file = os.path.join(cache_dir, file_name)

    if os.path.exists(cache_file):
        print(f"\nLoading cached dataset subset from: {cache_file}")
        try:
            # We save the three parts of the dataset in a dictionary
            cached_data = torch.load(cache_file)
            return cached_data['full_train'], cached_data['val']
        except Exception as e:
            print(f"Error loading cache file: {e}. Rebuilding dataset.")
            os.remove(cache_file) # Delete corrupted cache file

    # 2. If cache doesn't exist or failed to load, create the dataset
    print(f"\nCreating new dataset subset ({total_images}) and caching it...")
    # Assuming loadDataset is correctly imported from src
    if dataset_name == "cifar10":
        subset, train, val, selected_indices = loadCifar10Subset(data_dir, total_images, transform)
    elif dataset_name == "cifar100":
        subset, train, val, selected_indices = loadCifar100Subset(data_dir, total_images, transform)
    else:
        train, train_indices, val, val_indices = loadHuggingFaceDataset(data_dir, total_images, transform, dataset_name, num_classes=num_classes)
        
    plot.writeCsvLine(os.path.join(output_dir, "selectedIndices.csv"), [file_name, train_indices, val_indices])
    
    # 3. Cache the newly created dataset
    data_to_save = {
        'full_train': train,
        'val': val
    }
    torch.save(data_to_save, cache_file)
    print(f"Dataset subset saved successfully to: {cache_file}")
    
    return train, val