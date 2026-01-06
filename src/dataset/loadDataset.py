import torch
import torchvision
from torch.utils.data import Subset, random_split
import os
from . import datasetUtils
from datasets import load_dataset
from huggingface_hub import login
from .. import plot

def loadIndicesFromDataset(dataset, train_indices, val_indices, data_dir, modelc):
    if(dataset == 'imagenet-a' or dataset == 'imagenet-sketch'):
        return loadUrlDownloadedDataset(data_dir, train_indices, val_indices, dataset, modelc)
    if dataset == 'cifar100':
        return loadCifar100Dataset(data_dir, train_indices, val_indices, modelc)
    else:
        return loadHuggingfaceDataset(dataset, train_indices, val_indices, modelc)

def createNewDataset(dataset, total_images, num_classes, output_dir, subset_num, data_dir, modelc):
    
    if (dataset == 'imagenet-a' or dataset == 'imagenet-sketch'):
        return newUrlDownloadedDataset(data_dir, total_images, num_classes, output_dir, subset_num, dataset, modelc)
    if dataset == 'cifar100':
        return newCifar100Dataset(data_dir, total_images, num_classes, output_dir, subset_num, modelc)
    else:
        return newHuggingfaceDataset(dataset, total_images, num_classes, output_dir, subset_num, modelc)
    
def loadCifar100Dataset(data_dir, train_indices, val_indices, modelc):
    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=modelc.data_transforms)
    val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=modelc.data_transforms)
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    print(f"Cifar100 dataset loaded with pre-existing indices")
    
    return train_subset, val_subset
    
def newCifar100Dataset(data_dir, total_images, num_classes, output_dir, subset_num, modelc):

    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=modelc.data_transforms)
    val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=modelc.data_transforms)

    images_per_class = total_images // num_classes

    train_indices = datasetUtils.getRandomImages(num_classes, images_per_class, train_dataset, len(train_dataset.classes))
    val_indices = datasetUtils.getRandomImages(num_classes, images_per_class, val_dataset, len(val_dataset.classes))

    subset_train_dataset = Subset(train_dataset, train_indices)
    subset_val_dataset = Subset(val_dataset, val_indices)

    print(f"Total images selected: {len(subset_train_dataset)}")
    
    file_name = f"cifar100_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    
    plot.writeCsvLine(os.path.join(output_dir, "selectedIndices.csv"), [file_name, train_indices, val_indices])

    return subset_train_dataset, subset_val_dataset

def newUrlDownloadedDataset(root, total_images, num_classes, output_dir, subset_num, dataset, modelc):
    # 1. Download and extract the data
    # This calls the function from src/dataset_utils.py to handle the download
    data_dir = datasetUtils.downloadUrlDataset(root_dir=root)
    if data_dir is None:
        raise FileNotFoundError(f"Failed to download or extract {dataset}.")

    # 2. Load data using ImageFolder (which expects class subdirectories)
    full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=modelc.data_transforms)
    
    if not full_dataset.classes:
        raise ValueError(f"Could not find any classes (subdirectories) in {data_dir}. Check the directory structure.")
    
    images_per_class = total_images // num_classes
    
    val_indices = datasetUtils.getRandomImages(num_classes, images_per_class, full_dataset, len(full_dataset.classes))
    
    train_indices = [ # o resto fica para treinamento da cabeça de validação
        i for i in range(len(full_dataset))
        if i not in val_indices
    ]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, list(val_indices))
    
    val_dataset.classes = full_dataset.classes
    train_dataset.classes = full_dataset.classes
    
    file_name = f"{dataset}_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    
    plot.writeCsvLine(os.path.join(output_dir, "selectedIndices.csv"), [file_name, train_indices, val_indices])
    
    print(f"New {dataset} subset created")
    
    return train_dataset, val_dataset

def loadUrlDownloadedDataset(root, train_indices, val_indices, dataset, modelc):
    print(f"\n--- Loading {dataset} Subset ---")
    
    # 1. Download and extract the data
    # This calls the function from src/dataset_utils.py to handle the download
    data_dir = datasetUtils.downloadUrlDataset(root_dir=root)
    if data_dir is None:
        raise FileNotFoundError(f"Failed to download or extract {dataset}.")

    # 2. Load data using ImageFolder (which expects class subdirectories)
    full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=modelc.data_transforms)
    
    if not full_dataset.classes:
        raise ValueError(f"Could not find any classes (subdirectories) in {data_dir}. Check the directory structure.")
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, list(val_indices))
    
    val_dataset.classes = full_dataset.classes
    train_dataset.classes = full_dataset.classes
    
    print(f"Loaded {dataset} subset")
    
    return train_dataset, val_dataset

def newHuggingfaceDataset(dataset_link, total_images, num_classes, output_dir, subset_num, modelc):
    print("\n--- Loading dataset via Hugging Face ---")
    
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
        
    train_indices = datasetUtils.getRandomImages(num_classes, images_per_class, hf_train, hf_train.features['label'].num_classes)
    val_indices = datasetUtils.getRandomImages(num_classes, images_per_class, hf_validation, hf_train.features['label'].num_classes)

    hf_train = hf_train.select(train_indices)
    hf_validation = hf_validation.select(val_indices)

    print(f"Loaded {num_classes} classes * {images_per_class} images per class")

    # 5. Wrap in PyTorch Dataset
    train_dataset = datasetUtils.HuggingFaceImageNetDataset(hf_dataset=hf_train, transform=modelc.data_transforms)
    val_dataset = datasetUtils.HuggingFaceImageNetDataset(hf_dataset=hf_validation, transform=modelc.data_transforms)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    dt_name = dataset_link.replace('/', '-') #remove diretório na hora de buscar o arquivo, existe ao ser um link do HuggingFace
    file_name = f"{dt_name}_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    
    plot.writeCsvLine(os.path.join(output_dir, "selectedIndices.csv"), [file_name, train_indices, val_indices])

    return train_dataset, val_dataset

def loadHuggingfaceDataset(dataset_link, train_indices, val_indices, modelc):
    print("\n--- Loading Huggingface dataset with pre-selected indices ---")
    
    try:
        
        hf_token = datasetUtils.loadToken('token.txt')
        
        login(token=hf_token, add_to_git_credential=False)
        
        hf_train = load_dataset(dataset_link, split='train', streaming=False)
        hf_validation = load_dataset(dataset_link, split='validation', streaming=False)
        
    except Exception as e:
         raise RuntimeError(f"Failed to load Hugging Face ImageNet subset: {e}")
             
    hf_train = hf_train.select(train_indices)
    hf_validation = hf_validation.select(val_indices)

    # 5. Wrap in PyTorch Dataset
    train_dataset = datasetUtils.HuggingFaceImageNetDataset(hf_dataset=hf_train, transform=modelc.data_transforms)
    val_dataset = datasetUtils.HuggingFaceImageNetDataset(hf_dataset=hf_validation, transform=modelc.data_transforms)

    print(f"\nLoaded dataset with previously selected indices")

    return train_dataset, val_dataset

# ----- funções antigas que estão em desuso ou foram retrabalhadas em outras -----
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