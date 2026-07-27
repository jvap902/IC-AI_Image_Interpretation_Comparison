import json
import torch
import random
import torchvision
from tqdm.auto import tqdm
from torch.utils.data import Subset
from collections import defaultdict
import torchvision.transforms.functional as F
from src.fileManagement.csvUtils import findInCsv, writeCsvLine
    
def loadToken(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_labels(dataset):
    # Directly access the 'label' column of the HF dataset
    if isinstance(dataset, Subset):
        return [dataset.dataset.targets[i] for i in dataset.indices]
    if hasattr(dataset, 'targets'):
        return dataset.targets
    else:
        return dataset['label']

def getRandomImages(dt_info, dataset):
    
    print("Selecting indices\n")
    
    num_classes = dt_info.num_classes
    images_per_class = dt_info.images_per_class
    labels = extract_labels(dataset)
    
    # 1. Pick num_classes classes
    available_classes = list(set(labels))
    selected_classes = set(random.sample(available_classes, num_classes))

    # 2. Collect indices per class
    class_indices = defaultdict(list)
    
    for idx, label in enumerate(tqdm(labels, desc="Scanning dataset labels")):
        if label in selected_classes:
            class_indices[label].append(idx)

    # 3. Check availability - verifica se existe o número de imagens por classe nesta classe
    for c in selected_classes:
        if len(class_indices[c]) < images_per_class:
            raise ValueError(
                f"Class {c} only has {len(class_indices[c])} images, "
                f"requested {images_per_class}."
            )

    # 4. Select balanced subset
    selected_indices = []
    for c in selected_classes:
        indices = class_indices[c]
        rand_img_start = random.randint(0, len(indices)-images_per_class)
        selected_indices.extend(indices[rand_img_start : rand_img_start + images_per_class])
    
    print("Indices selected")
                
    return selected_indices

def getRandomImagesFromClasses(dt_info, dataset, train_or_validation, huggingface=False):
    
    print("Getting specific classes")
    
    available_class_wnids = dt_info.available_classes[train_or_validation]
    
    class_names = getClasses(dataset)
            
    available_class_ids = []
    
    
    if huggingface:
        name_to_wnid = nameToWnid(dt_info.name)
        
        for idx, name in enumerate(class_names):
            clean_name = name.lower().replace('_', ' ').split(',')[0].strip()
            wnid_of_this_class = name_to_wnid.get(clean_name)
            
            if wnid_of_this_class in available_class_wnids:
                available_class_ids.append(idx)
    else:
        for idx, wnid in enumerate(class_names):
            
            if wnid in available_class_wnids:
                available_class_ids.append(idx)
    
                
    print("Number of requested classes:", len(available_class_wnids))
    print("Number of mapped class IDs:", len(available_class_ids))
                
    print(f"Selecting indices from {len(available_class_wnids)} specific classes\n")

    labels = extract_labels(dataset)

    # Collect indices per class
    class_indices = defaultdict(list)

    for idx, label in enumerate(labels):
        if label in available_class_ids:
            class_indices[label].append(idx) 
            
    # Check availability - verifica se existe o número de imagens por classe nesta classe
    for c in available_class_ids:
        if len(class_indices[c]) < dt_info.images_per_class:
            raise ValueError(
                f"Class {c} only has {len(class_indices[c])} images, "
                f"requested {dt_info.images_per_class}."
            )

    # Select balanced subset
    selected_indices = []
    for c in available_class_ids:
        indices = class_indices[c]
        rand_img_start = random.randint(0, len(indices)-dt_info.images_per_class)
        selected_indices.extend(indices[rand_img_start : rand_img_start + dt_info.images_per_class]) #sequência aleatória de índices    
    
    print(f"Indices selected - {len(selected_indices)}")
                
    return selected_indices

def imageSelector(dt_info, dataset, train_or_validation, huggingface=False):
    if dt_info.available_classes[train_or_validation][0] == 'all':
        return getRandomImages(dt_info, dataset)
    else:
        return getRandomImagesFromClasses(dt_info, dataset, train_or_validation, huggingface=huggingface)


def getClasses(dataset):
    if hasattr(dataset, 'features') and 'label' in dataset.features:
        # This returns the list of WNIDs for ImageNet: ['n01443537', 'n01443538', ...]
        return dataset.features['label'].names
    # Fallback for torchvision/other datasets
    elif isinstance(dataset, Subset):
        return dataset.dataset.classes
    return dataset.classes

def writeDatasetClasses(dt_info):
    classes_file = './dataStorage/datasetClasses.csv'
    ans = findInCsv(classes_file, ['dataset', 'subset', 'num_classes', 'num_images'], [dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images])
    if len(ans) == 0:
        writeCsvLine(classes_file, [dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images, dt_info.available_classes['train'], dt_info.available_classes['validation']])
        
def nameToWnid(dataset_name):
    if 'imagenet' in dataset_name.lower():
        with open("./data/imagenet_class_index.json") as f:
            class_index = json.load(f)
    
        # Create Name -> WNID map: {"cloak": "n03033013"}
        name_to_wnid = {
            v[1].lower().replace('_', ' ').split(',')[0].strip(): v[0] 
            for v in class_index.values()
        }

        return name_to_wnid
    
def getStdMapping():
    with open("./data/imagenet_class_index.json") as f:
        class_index = json.load(f)
    # class_index looks like {"0": ["n01443537", "goldfish"]}
    # We want {"n01443537": 0}
    return {v[0]: int(k) for k, v in class_index.items()}

def imagenetCDistortionMap(dist):
    distortions = {
        "noise": {'gaussian_noise', 'shot_noise', 'impulse_noise'},
        "blur": {'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'},
        "weather": {'frost', 'snow', 'fog', 'brightness'},
        "digital": {'constrast', 'elastic_transform', 'pixelate', 'jpeg_compression'},
        "extra": {'speckle_noise', 'spatter', 'gaussian_blur', 'saturate'},
        "types": {'noise', 'blur', 'weather', 'digital', 'extra'}
    }
    
    if dist in distortions['types']:
        return distortions[dist] #retorna distorções disponíveis para o tipo de entrada
    
    else:
        for type, dists in distortions.items():
            if dist in dists:
                return type
            
    raise ValueError("Unavailable distortion")

def pathConcat(dataset):
    if 'imagenet-c' in dataset:
        split_dt = dataset.split('-')
        return f'/{split_dt[-2]}/{split_dt[-1]}' #[-2] = distorção, [-1] = intensidade
    else:
        return '' #se não precisa concatenar nada só retorna str vazia

class HuggingFaceDatasetWrapper(torch.utils.data.Dataset):
    """Unpacks HuggingFace dict items into (image, label) tuples to match PyTorch Dataset interface."""

    def __init__(self, hf_dataset, image_key="image", label_key="label"):
        self.dataset = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.classes = hf_dataset.features[label_key].names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample[self.image_key]
        label = sample[self.label_key]

        if isinstance(image, torch.Tensor):
            image = image.clone()
        else:
            image = F.pil_to_tensor(image)  # keeps [0, 255] uint8, no rescaling
        
        if isinstance(label, torch.Tensor):
            label = label.clone()
        else:
            label = torch.tensor(label)

        return image, label