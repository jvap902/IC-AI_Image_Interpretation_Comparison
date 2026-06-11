import random
import torchvision
from pathlib import Path
from huggingface_hub import login
from torch.utils.data import Subset
from datasets import load_dataset, load_from_disk
from kagglehub import dataset_load, KaggleDatasetAdapter
from src.fileManagement import csvUtils
from . import datasetUtils

def loadIndicesFromDataset(dt_info, train_indices, val_indices, data_dir):
    
    dataset = dt_info.name.lower()
    
    print(f"\nLoading previously selected {dataset} indices\n")
    
    if((dataset == 'imagenet-a') or (dataset == 'imagenet-sketch') or ('imagenet-c' in dataset)):
        train_dataset, val_dataset = loadUrlDownloadedDataset(data_dir, train_indices, val_indices, dataset)
    elif dataset == 'cifar100':
        train_dataset, val_dataset = loadCifar100Dataset(data_dir, train_indices, val_indices)
    elif dataset == 'cifar10':
        train_dataset, val_dataset = loadCifar10Dataset(data_dir, train_indices, val_indices)
    else:
        train_dataset, val_dataset = loadHuggingfaceDataset(dt_info, train_indices, val_indices)
    
    datasetUtils.writeDatasetClasses(dt_info)

    return train_dataset, val_dataset

def createNewDataset(dt_info, output_dir, data_dir):
    
    dataset = dt_info.name
    
    print(f"\nCreating new set of indices for {dataset}\n")
    
    if ((dataset == 'imagenet-a') or (dataset == 'imagenet-sketch') or ('imagenet-c' in dataset)):
        train_dataset, val_dataset = newUrlDownloadedDataset(dt_info, data_dir, output_dir)
    elif dataset == 'fgvc-aircraft':
        train_dataset, val_dataset = newKaggleDataset(dt_info, data_dir, output_dir)
    elif dataset == 'cifar100':
        train_dataset, val_dataset = newCifar100Dataset(dt_info, data_dir, output_dir)
    elif dataset == 'cifar10':
        train_dataset, val_dataset = newCifar10Dataset(dt_info, data_dir, output_dir)
    else:
        train_dataset, val_dataset = newHuggingfaceDataset(dt_info, output_dir)
    
    datasetUtils.writeDatasetClasses(dt_info)

    return train_dataset, val_dataset


def newKaggleDataset(dt_info, data_dir, output_dir): #falta validar
    print(f"\n--- Creating {dt_info.name} Subset ---")
    
    url = datasetUtils.getKaggleInfo(dt_info.name)
    
    num_classes = dt_info.num_classes
    
    hf_train = dataset_load(KaggleDatasetAdapter.HUGGING_FACE, url, "train.csv")
    hf_validation = dataset_load(KaggleDatasetAdapter.HUGGING_FACE, url, "val.csv")
    
    print(f"Loaded Kaggle dataset of size: {len(hf_train) + len(hf_validation)}")
    
    print(f"\nCreating subset with {dt_info.images_per_class} images per class for {num_classes} classes.")
        
    train_indices = datasetUtils.imageSelector(dt_info, hf_train, 'train', huggingface=True)
    val_indices = datasetUtils.imageSelector(dt_info, hf_validation, 'validation', huggingface=True)
    
    hf_train = hf_train.select(train_indices)
    hf_validation = hf_validation.select(val_indices)
    
    #salva apenas subset
    hf_train.save_to_disk(f'./data/{dt_info.name_w_subset}/train')
    hf_validation.save_to_disk(f'./data/{dt_info.name_w_subset}/validation')  

    print(f"Loaded {num_classes} classes * {dt_info.images_per_class} images per class")
    
    # 5. Wrap in PyTorch Dataset
    train_dataset = hf_train.with_format("torch")
    val_dataset = hf_validation.with_format("torch")
        
    print(len(val_dataset.classes))
    
    a = []
    for e in val_dataset.classes:
        if e != None:
            a.append(e)
        
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    dt_name = dt_info.name.replace('/', '-') #remove diretório na hora de buscar o arquivo, existe ao ser um link do HuggingFace
    file_name = f"{dt_name}_subset_i{dt_info.num_images}_c{num_classes}({dt_info.subset}).pt"
    
    csvUtils.writeCsvLine(output_dir+"/selectedIndices.csv", [file_name, train_indices, val_indices])

    return train_dataset, val_dataset
    
def loadCifar10Dataset(data_dir, train_indices, val_indices):
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    print(f"Cifar10 dataset loaded with pre-existing indices")
    
    return train_subset, val_subset
    
def newCifar10Dataset(dt_info, data_dir, output_dir):
    
    dataset, subset_num, num_classes, total_images = dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images
    
    if num_classes > 10:
        num_classes = 10
    
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    train_indices = datasetUtils.imageSelector(dt_info, train_dataset, 'train')
    val_indices = datasetUtils.imageSelector(dt_info, val_dataset, 'validation')

    subset_train_dataset = Subset(train_dataset, train_indices)
    subset_val_dataset = Subset(val_dataset, val_indices)

    print(f"Total images selected: {len(subset_train_dataset)}")
    
    file_name = f"cifar10_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    
    csvUtils.writeCsvLine(output_dir+"/selectedIndices.csv", [file_name, train_indices, val_indices])

    return subset_train_dataset, subset_val_dataset
    
def loadCifar100Dataset(data_dir, train_indices, val_indices):
    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
    val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    print(f"Cifar100 dataset loaded with pre-existing indices")
    
    return train_subset, val_subset
    
def newCifar100Dataset(dt_info, data_dir, output_dir):
    
    dataset, subset_num, num_classes, total_images = dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images

    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
    val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)

    train_indices = datasetUtils.imageSelector(dt_info, train_dataset, 'train')
    val_indices = datasetUtils.imageSelector(dt_info, val_dataset, 'validation')

    subset_train_dataset = Subset(train_dataset, train_indices)
    subset_val_dataset = Subset(val_dataset, val_indices)

    print(f"Total images selected: {len(subset_train_dataset)}")
    
    file_name = f"cifar100_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    
    csvUtils.writeCsvLine(output_dir+"/selectedIndices.csv", [file_name, train_indices, val_indices])

    return subset_train_dataset, subset_val_dataset

def newUrlDownloadedDataset(dt_info, data_dir, output_dir):
    dataset, subset_num, num_classes, total_images = dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images
    
    full_dataset = datasetUtils.getUrlDataset(data_dir, dataset)
        
    val_indices = datasetUtils.imageSelector(dt_info, full_dataset, 'validation')
    val_indices_set = set(val_indices)
    
    # o resto dos índices pode ser utilizado para treinar a cabeça de validação

    available_train_indices = [i for i in range(len(full_dataset)) if i not in val_indices_set]
    train_subset_for_selection = Subset(full_dataset, available_train_indices)
    
    train_indices_local = datasetUtils.imageSelector(dt_info, train_subset_for_selection, 'train')
    # Map back to original dataset indices
    train_indices = [available_train_indices[i] for i in train_indices_local]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, list(val_indices))
    
    val_dataset.classes = full_dataset.classes
    train_dataset.classes = full_dataset.classes
    
    file_name = f"{dataset}_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    
    csvUtils.writeCsvLine(output_dir+"/selectedIndices.csv", [file_name, train_indices, val_indices])
    
    print(f"New {dataset} subset created")
    
    return train_dataset, val_dataset

def loadUrlDownloadedDataset(data_dir, train_indices, val_indices, dataset):
    print(f"\n--- Loading {dataset} Subset ---")
    
    full_dataset = datasetUtils.getUrlDataset(data_dir, dataset)
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    val_dataset.classes = full_dataset.classes
    train_dataset.classes = full_dataset.classes
    
    print(f"Loaded {dataset} subset")
    
    return train_dataset, val_dataset

def newHuggingfaceDataset(dt_info, output_dir):
    print("\n--- Loading dataset via Hugging Face ---")
    
    dataset_link, subset_num, num_classes, total_images = dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images
        
    try:
        hf_token = datasetUtils.loadToken('token.txt')
        
        login(token=hf_token, add_to_git_credential=False)
        
        dir_name = dt_info.name.replace('/','-')
        
        # se dataset completo está baixado
        if(Path(f'./data/{dir_name}').is_dir()):
            
            print("Loading already downloaded dataset")
            
            hf_train = load_from_disk(f'./data/{dir_name}/train')
            hf_validation = load_from_disk(f'./data/{dir_name}/validation')
        
        #caso seja necessário baixar
        else:
            print("Downloading huggingface dataset")
            
            hf_train = load_dataset(dataset_link, split='train', streaming=False)
            hf_validation = load_dataset(dataset_link, split='validation', streaming=False)
        
    except Exception as e:
         raise RuntimeError(f"Failed to load Hugging Face Dataset: {e}")
    
    
    print(f"Loaded HF Dataset of size: {len(hf_train) + len(hf_validation)}")
    
    print(f"\nCreating subset with {dt_info.images_per_class} images per class for {num_classes} classes.")
    
        
    train_indices = datasetUtils.imageSelector(dt_info, hf_train, 'train', huggingface=True)
    val_indices = datasetUtils.imageSelector(dt_info, hf_validation, 'validation', huggingface=True)
    
    print(len(train_indices))
    print(len(val_indices))
    
    hf_train = hf_train.select(train_indices)
    hf_validation = hf_validation.select(val_indices)
    
    #salva apenas subset
    hf_train.save_to_disk(f'./data/{dt_info.name_w_subset}/train')
    hf_validation.save_to_disk(f'./data/{dt_info.name_w_subset}/validation')  

    print(f"Loaded {num_classes} classes * {dt_info.images_per_class} images per class")
    
    # 5. Wrap in PyTorch Dataset
    train_dataset = datasetUtils.HuggingFaceDatasetWrapper(hf_train)
    val_dataset = datasetUtils.HuggingFaceDatasetWrapper(hf_validation)
        
    print(len(val_dataset.classes))
        
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    dt_name = dataset_link.replace('/', '-') #remove diretório na hora de buscar o arquivo, existe ao ser um link do HuggingFace
    file_name = f"{dt_name}_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    
    csvUtils.writeCsvLine(output_dir+"/selectedIndices.csv", [file_name, train_indices, val_indices])

    return train_dataset, val_dataset

def loadHuggingfaceDataset(dt_info, train_indices, val_indices):
    print("\n--- Loading Huggingface dataset with pre-selected indices ---")
    
    dataset_link = dt_info.name
    
    try:   
        if(Path(f'./data/{dt_info.name_w_subset}').is_dir()):
            
            print("Loading already selected and downloaded subset")
            
            hf_train = load_from_disk(f'./data/{dt_info.name_w_subset}/train')
            hf_validation = load_from_disk(f'./data/{dt_info.name_w_subset}/validation')
            
        else:
            
            dir_name = dt_info.name.replace('/','-')
        
            # se dataset completo está baixado
            if(Path(f'./data/{dir_name}').is_dir()):
            
                print("Loading already downloaded dataset")
                
                hf_train = load_from_disk(f'./data/{dir_name}/train')
                hf_validation = load_from_disk(f'./data/{dir_name}/validation')
                
            else:
                hf_token = datasetUtils.loadToken('token.txt')
                
                login(token=hf_token, add_to_git_credential=False)
                
                hf_train = load_dataset(dataset_link, split='train', streaming=False)
                hf_validation = load_dataset(dataset_link, split='validation', streaming=False)
                
            hf_train = hf_train.select(train_indices)
            hf_validation = hf_validation.select(val_indices)
            
            
    except Exception as e:
         raise RuntimeError(f"Failed to load Hugging Face ImageNet subset: {e}")

    # 5. Wrap in PyTorch Dataset
    train_dataset = datasetUtils.HuggingFaceDatasetWrapper(hf_train)
    val_dataset = datasetUtils.HuggingFaceDatasetWrapper(hf_validation)

    print(f"\nLoaded dataset with previously selected indices")

    return train_dataset, val_dataset