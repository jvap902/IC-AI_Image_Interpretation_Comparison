import torch
import torchvision
from torch.utils.data import Subset
import os


def loadCifar10Subset(root, imagesPerClass, data_transforms):
    # Assuming you have already defined and downloaded your dataset:
    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
    dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=data_transforms) # Use train set for example
    val_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=data_transforms)

    # 1. Initialize storage for selected indices
    selected_indices = []
    count_per_class = {i: 0 for i in range(10)}

    # 2. Iterate through the dataset's indices
    for i in range(len(dataset)):
        # CIFAR10 stores targets as a list/array attribute or returns them with the data.
        # We access the class label (target) for the current index 'i'.
        label = dataset.targets[i] 

        # 3. Check if we need more samples for this class
        if count_per_class[label] < imagesPerClass:
            selected_indices.append(i)
            count_per_class[label] += 1
        
        # Optional: Stop early once all classes have x samples
        if len(selected_indices) == imagesPerClass * 10: # x * 10 classes
            break

    # 4. Create the new subset dataset
    # This new dataset contains exactly 100 non-random images from each class.
    subset_train_dataset = Subset(dataset, selected_indices)

    print(f"Total images selected: {len(subset_train_dataset)}")

    return subset_train_dataset, dataset, val_dataset
    # Expected output: Total images selected: 1000
    
def loadCifar100Subset(root, imagesPerClass, data_transforms):
    # Assuming you have already defined and downloaded your dataset:
    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
    dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=data_transforms) # Use train set for example
    val_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=data_transforms)

    # 1. Initialize storage for selected indices
    selected_indices = []
    count_per_class = {i: 0 for i in range(100)}

    # 2. Iterate through the dataset's indices
    for i in range(len(dataset)):
        # CIFAR10 stores targets as a list/array attribute or returns them with the data.
        # We access the class label (target) for the current index 'i'.
        label = dataset.targets[i] 

        # 3. Check if we need more samples for this class
        if count_per_class[label] < imagesPerClass:
            selected_indices.append(i)
            count_per_class[label] += 1
        
        # Optional: Stop early once all classes have x samples
        if len(selected_indices) == imagesPerClass * 100: # x * 10 classes
            break

    # 4. Create the new subset dataset
    # This new dataset contains exactly 100 non-random images from each class.
    subset_train_dataset = Subset(dataset, selected_indices)

    print(f"Total images selected: {len(subset_train_dataset)}")

    return subset_train_dataset, dataset, val_dataset
    # Expected output: Total images selected: 1000
    
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
    
    # 3. Cache the newly created dataset
    data_to_save = {
        'subset': subset,
        'full_train': full_train,
        'val': val
    }
    torch.save(data_to_save, cache_file)
    print(f"Dataset subset saved successfully to: {cache_file}")
    
    return subset, full_train, val