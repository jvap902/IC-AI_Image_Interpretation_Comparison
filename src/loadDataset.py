import torch
import torchvision
from torch.utils.data import Subset


def loadCifar10Subset(root, imagesPerClass, data_transforms):
    # Assuming you have already defined and downloaded your dataset:
    # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
    dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=data_transforms['train']) # Use train set for example
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])

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