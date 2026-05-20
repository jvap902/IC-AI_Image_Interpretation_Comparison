import os
import random
import torch
import torchvision
from pathlib import Path
from huggingface_hub import login
from torch.utils.data import Subset, Dataset
from datasets import load_dataset, load_from_disk, Dataset as HFDataset, Features, Image as HFImage, ClassLabel
from kagglehub import dataset_load, KaggleDatasetAdapter
from PIL import Image
from src.fileManagement import csvUtils
from . import datasetUtils

def loadIndicesFromDataset(dt_info, train_indices, val_indices, data_dir):
    
    dataset = dt_info.name
    
    print(f"\nLoading previously selected {dataset} indices\n")
    
    if((dataset == 'imagenet-a') or (dataset == 'imagenet-sketch') or ('imagenet-c' in dataset)):
        train_dataset, val_dataset = loadUrlDownloadedDataset(data_dir, train_indices, val_indices, dataset)
    elif dataset == 'cifar100':
        train_dataset, val_dataset = loadCifar100Dataset(data_dir, train_indices, val_indices, modelc)
    elif dataset == 'cifar10':
        train_dataset, val_dataset = loadCifar10Dataset(data_dir, train_indices, val_indices, modelc)
    else:
        train_dataset, val_dataset = loadHuggingfaceDataset(dt_info, train_indices, val_indices)
    
    datasetUtils.writeDatasetClasses(dt_info)

    return train_dataset, val_dataset

def createNewDataset(dt_info, output_dir, data_dir, modelc):
    
    dataset = dt_info.name
    
    print(f"\nCreating new set of indices for {dataset}\n")
    
    if ((dataset == 'imagenet-a') or (dataset == 'imagenet-sketch') or ('imagenet-c' in dataset)):
        train_dataset, val_dataset = newUrlDownloadedDataset(dt_info, data_dir, output_dir, modelc)
    elif dataset == 'fgvc-aircraft':
        train_dataset, val_dataset = newKaggleHubDataset(dt_info, data_dir, output_dir, modelc)
    elif dataset == 'cifar100':
        train_dataset, val_dataset = newCifar100Dataset(dt_info, data_dir, output_dir, modelc)
    elif dataset == 'cifar10':
        train_dataset, val_dataset = newCifar10Dataset(dt_info, data_dir, output_dir, modelc)
    else:
        train_dataset, val_dataset = newHuggingfaceDataset(dt_info, output_dir, modelc)
    
    datasetUtils.writeDatasetClasses(dt_info)

    return train_dataset, val_dataset

def loadKaggleDataset(data_dir, train_indices, val_indices, dataset, dt_info): #TEMPORÁRIO, DEPOIS SERÁ CARREGADO COMO Datset do huggingface
    print(f"\n--- Loading {dataset} Subset ---")
    
    url, file_name, extract_dir, compression_type = datasetUtils.getDownloadInfo(dataset)
    
    # 1. Download and extract the data
    # This calls the function from src/dataset_utils.py to handle the download
    dataset_dir = datasetUtils.downloadUrlDataset(root_dir=data_dir, url=url, file_name=file_name, extract_dir=extract_dir, compression_type=compression_type)
    if dataset_dir is None:
        raise FileNotFoundError(f"Failed to download or extract {dataset}.")
    
    train_dataset = AircraftDataset(dataset_dir, split='train', variant='variant')
    val_dataset = AircraftDataset(dataset_dir, split='val', variant='variant')
    
    #data_transforms removida, salvar agora como Dataset
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    hf_features = Features({
        "image": HFImage(),  # Tells HF to treat this as an image feature
        "label": ClassLabel(num_classes=len(train_dataset.classes), names=train_dataset.classes)
    })
    
    # 2. Hardened generator logic
    def hf_generator(subset):
        def gen():
            for idx in range(len(subset)):
                try:
                    image, label = subset[idx]
                    
                    # Ensure image is in PIL format for Hugging Face
                    if isinstance(image, torch.Tensor):
                        image = torchvision.transforms.ToPILImage()(image)
                    elif isinstance(image, str):
                        image = Image.open(image).convert('RGB')
                        
                    yield {"image": image, "label": label}
                except Exception as inner_error:
                    # Print out what went wrong BEFORE Hugging Face silences it!
                    print(f"\n[GENERATOR ERROR at index {idx}]: {inner_error}")
                    raise inner_error
        return gen

    # 3. Supply the features schema to from_generator
    hf_train = HFDataset.from_generator(hf_generator(train_subset), features=hf_features)
    hf_validation = HFDataset.from_generator(hf_generator(val_subset), features=hf_features)
    
    dir_name = dt_info.name_w_subset
        
    hf_train.save_to_disk(f'./data/{dir_name}/train')
    hf_validation.save_to_disk(f'./data/{dir_name}/validation')
    
    print(f"{dataset} dataset loaded with pre-existing indices")
    
    return hf_train.with_format("torch"), hf_validation.with_format("torch")
    
def newKaggleDataset(dt_info, data_dir, output_dir, modelc):
    print(f"\n--- Creating {dt_info.name} Subset ---")
    
    url = datasetUtils.getKaggleInfo(dt_info.name)
    
    hf_train = dataset_load(KaggleDatasetAdapter.HUGGING_FACE, url, "train.csv")
    hf_validation = dataset_load(KaggleDatasetAdapter.HUGGING_FACE, url, "val.csv")
    
    print(f"Loaded Kaggle dataset of size: {len(hf_train) + len(hf_validation)}")
    
    print(f"\nCreating subset with {dt_info.images_per_class} images per class for {num_classes} classes.")
        
    train_indices = datasetUtils.imageSelector(dt_info, hf_train, hf_train.features['label'].num_classes, 'train', huggingface=True)
    val_indices = datasetUtils.imageSelector(dt_info, hf_validation, hf_train.features['label'].num_classes, 'validation', huggingface=True)
    
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
    
    dt_name = dataset_link.replace('/', '-') #remove diretório na hora de buscar o arquivo, existe ao ser um link do HuggingFace
    file_name = f"{dt_name}_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    
    csvUtils.writeCsvLine(os.path.join(output_dir, "selectedIndices.csv"), [file_name, train_indices, val_indices])

    return train_dataset, val_dataset
    
    train_indices = datasetUtils.imageSelector(dt_info, train_dataset, len(train_dataset.classes), 'train')
    val_indices = datasetUtils.imageSelector(dt_info, val_dataset, len(val_dataset.classes), 'validation')
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    print(f"Total images selected: {len(train_subset)}")
    
    file_name = f"{dt_info.name}_subset_i{dt_info.num_images}_c{dt_info.num_classes}({dt_info.subset}).pt"
    
    csvUtils.writeCsvLine(os.path.join(output_dir, "selectedIndices.csv"), [file_name, train_indices, val_indices])
    
    return train_subset, val_subset
    
def loadCifar10Dataset(data_dir, train_indices, val_indices, modelc):
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=modelc.data_transforms)
    val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=modelc.data_transforms)
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    print(f"Cifar10 dataset loaded with pre-existing indices")
    
    return train_subset, val_subset
    
def newCifar10Dataset(dt_info, data_dir, output_dir, modelc):
    
    dataset, subset_num, num_classes, total_images = dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images
    
    if num_classes > 10:
        num_classes = 10
    
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=modelc.data_transforms)
    val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=modelc.data_transforms)
    
    train_indices = datasetUtils.imageSelector(dt_info, train_dataset, len(train_dataset.classes), 'train')
    val_indices = datasetUtils.imageSelector(dt_info, val_dataset, len(val_dataset.classes), 'validation')

    subset_train_dataset = Subset(train_dataset, train_indices)
    subset_val_dataset = Subset(val_dataset, val_indices)

    print(f"Total images selected: {len(subset_train_dataset)}")
    
    file_name = f"cifar10_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    
    csvUtils.writeCsvLine(os.path.join(output_dir, "selectedIndices.csv"), [file_name, train_indices, val_indices])

    return subset_train_dataset, subset_val_dataset
    
def loadCifar100Dataset(data_dir, train_indices, val_indices, modelc):
    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=modelc.data_transforms)
    val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=modelc.data_transforms)
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    print(f"Cifar100 dataset loaded with pre-existing indices")
    
    return train_subset, val_subset
    
def newCifar100Dataset(dt_info, data_dir, output_dir, modelc,):
    
    dataset, subset_num, num_classes, total_images = dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images

    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=modelc.data_transforms)
    val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=modelc.data_transforms)

    train_indices = datasetUtils.imageSelector(dt_info, train_dataset, len(train_dataset.classes), 'train')
    val_indices = datasetUtils.imageSelector(dt_info, val_dataset, len(val_dataset.classes), 'validation')

    subset_train_dataset = Subset(train_dataset, train_indices)
    subset_val_dataset = Subset(val_dataset, val_indices)

    print(f"Total images selected: {len(subset_train_dataset)}")
    
    file_name = f"cifar100_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    
    csvUtils.writeCsvLine(os.path.join(output_dir, "selectedIndices.csv"), [file_name, train_indices, val_indices])

    return subset_train_dataset, subset_val_dataset

def newUrlDownloadedDataset(dt_info, data_dir, output_dir, modelc):
    dataset, subset_num, num_classes, total_images = dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images
    
    full_dataset = datasetUtils.getUrlDataset(data_dir, dataset, modelc)
        
    val_indices = datasetUtils.imageSelector(dt_info, full_dataset, len(full_dataset.classes), 'validation')
    
    # o resto dos índices pode ser utilizado para treinar a cabeça de validação

    possible_train_indices = list(range(len(full_dataset)))
    possible_train_indices = [item for item in possible_train_indices if item not in set(val_indices)]
    
    train_indices = random.sample(possible_train_indices, k=2000)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, list(val_indices))
    
    val_dataset.classes = full_dataset.classes
    train_dataset.classes = full_dataset.classes
    
    file_name = f"{dataset}_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    
    csvUtils.writeCsvLine(os.path.join(output_dir, "selectedIndices.csv"), [file_name, train_indices, val_indices])
    
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

def newHuggingfaceDataset(dt_info, output_dir, modelc):
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
    
        
    train_indices = datasetUtils.imageSelector(dt_info, hf_train, hf_train.features['label'].num_classes, 'train', huggingface=True)
    val_indices = datasetUtils.imageSelector(dt_info, hf_validation, hf_train.features['label'].num_classes, 'validation', huggingface=True)
    
    print(len(train_indices))
    print(len(val_indices))
    
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
    
    print(len(set(a)))
        
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    dt_name = dataset_link.replace('/', '-') #remove diretório na hora de buscar o arquivo, existe ao ser um link do HuggingFace
    file_name = f"{dt_name}_subset_i{total_images}_c{num_classes}({subset_num}).pt"
    
    csvUtils.writeCsvLine(os.path.join(output_dir, "selectedIndices.csv"), [file_name, train_indices, val_indices])
    
    # Debugging check
    img, label = train_dataset[0]
    print(f"DEBUG: First image label index: {label}")

    return train_dataset, val_dataset

def loadHuggingfaceDataset(dt_info, train_indices, val_indices):
    print("\n--- Loading Huggingface dataset with pre-selected indices ---")
    
    dataset_link = dt_info.name
    
    try:   
        hf_token = datasetUtils.loadToken('token.txt')
        
        login(token=hf_token, add_to_git_credential=False)
        
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
                
                hf_train = load_dataset(dataset_link, split='train', streaming=False)
                hf_validation = load_dataset(dataset_link, split='validation', streaming=False)
                
            hf_train = hf_train.select(train_indices)
            hf_validation = hf_validation.select(val_indices)
            
            
    except Exception as e:
         raise RuntimeError(f"Failed to load Hugging Face ImageNet subset: {e}")

    # 5. Wrap in PyTorch Dataset
    train_dataset = hf_train.with_format("torch")
    val_dataset = hf_validation.with_format("torch")

    print(f"\nLoaded dataset with previously selected indices")

    return train_dataset, val_dataset


class AircraftDataset(Dataset):
    """
    Custom Dataset for FGVC-Aircraft. 
    Expects data at: root/fgvc-aircraft-2013b/data/
    """
    def __init__(self, root, split='trainval', transform=None, variant='family'):
        self.root = root
        self.transform = transform
        self.split = split
        
        # Aircraft has 3 levels of labels: 'manufacturer', 'family', 'variant'
        # 'variant' is the most fine-grained (100 classes)
        metadata_file = os.path.join(root, 'data', f'images_{variant}_{split}.txt')
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}. Ensure dataset is extracted correctly.")

        self.image_labels = []
        self.image_names = []
        
        # Read the mapping file
        with open(metadata_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                self.image_names.append(parts[0])
                self.image_labels.append(parts[1])

        # Create class-to-index mapping
        self.classes = sorted(list(set(self.image_labels)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.targets = [self.class_to_idx[lbl] for lbl in self.image_labels]
        self.images_dir = os.path.join(root, 'data', 'images')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx] + ".jpg"
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        return image, label