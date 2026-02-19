import os
import tarfile
import zipfile
from tqdm.auto import tqdm
import requests
from torch.utils.data import Dataset, Subset
import random
from collections import defaultdict
from ..plot import findInCsv, writeCsvLine
import json


# ImageNet-A download details
IMAGENET_A_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
IMAGENET_A_FILENAME = "imagenet-a.tar"
IMAGENET_A_EXTRACT_DIR = "imagenet-a"

# Imagenet-sketch
IMAGENET_SKETCH_URL = "https://www.kaggle.com/api/v1/datasets/download/wanghaohan/imagenetsketch"
IMAGENET_SKETCH_FILENAME = "archive.zip"
IMAGENET_SKETCH_EXTRACT_DIR = "imagenet-sketch"

#FGVC-Aircraft
AIRCRAFT_URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
AIRCRAFT_FILENAME = "fgvc-aircraft-2013b.tar.gz"
AIRCRAFT_EXTRACT_DIR = "fgvc-aircraft-2013b"

def getDownloadInfo(dataset):
    if dataset == 'imagenet-a':
        return IMAGENET_A_URL, IMAGENET_A_FILENAME, IMAGENET_A_EXTRACT_DIR, 'tar'
    elif dataset == 'imagenet-sketch':
        return IMAGENET_SKETCH_URL, IMAGENET_SKETCH_FILENAME, IMAGENET_SKETCH_EXTRACT_DIR, 'zip'
    elif dataset == 'fgvc-aircraft':
        return AIRCRAFT_URL, AIRCRAFT_FILENAME, AIRCRAFT_EXTRACT_DIR, 'tar'
    else:
        raise ValueError("Unsupported dataset")

def downloadUrlDataset(root_dir, url, file_name, extract_dir, compression_type):
    """
    Downloads and extracts the ImageNet-A dataset from the corrected Berkeley URL.

    Args:
        root_dir (str): The base directory ('data/') where the dataset will be stored.

    Returns:
        str: The path to the extracted ImageNet-A or imagenet-sketch directory, or None if failed.
    """
    # Ensure the root data directory exists
    os.makedirs(root_dir, exist_ok=True)
    
    # Define paths
    extract_path = os.path.join(root_dir, extract_dir)
    file_path = os.path.join(root_dir, file_name)
    
    is_extracted_path = os.path.join(extract_path, 'sketch') if extract_dir == 'imagenet-sketch' else extract_path
    
    # 1. Check if the dataset is already extracted
    # We check for a common file structure to avoid re-downloading large files
    # The extracted directory should contain subdirectories (classes).
    if os.path.exists(is_extracted_path) and len(os.listdir(is_extracted_path)) > 1:
        print(f"{file_name} already found and extracted at: {is_extracted_path}")
        return is_extracted_path

    # 2. Download the file
    print(f"Downloading dataset from: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        with open(file_path, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), 
                             total=total_size//block_size, 
                             unit='KB', 
                             desc=file_name):
                f.write(data)
        
        print(f"\nDownload complete. File saved to: {file_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error during download from {url}: {e}")
        return None

    # 3. Extract the file
    if compression_type=='tar':
        return extractTarFile(root_dir, file_name, extract_path, file_path)
    elif compression_type=='zip':
        return extractZipFile(root_dir, file_name, extract_path, file_path)
    else:
        raise ValueError(f"compression type {compression_type} not supported for extraction")
    
def extractZipFile(root, file_name, extract_path, zip_filepath):
    print(f"Extracting {file_name} to {extract_path}...")
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(path=root)

        os.remove(zip_filepath)
        print("Extraction complete and ZIP file removed.")

        return extract_path

    except zipfile.BadZipFile as e:
        print(f"Invalid ZIP file: {e}")
        return None
    except OSError as e:
        print(f"OS error during cleanup: {e}")
        return extract_path

def extractTarFile(root, file_name, extract_path, tar_filepath):
    print(f"Extracting {file_name} to {extract_path}...")
    try:
        with tarfile.open(tar_filepath, 'r') as tar:
            # The tarball contains a single folder named 'imagenet-a'
            # We extract it directly into the root_dir
            tar.extractall(path=root) 
        
        # Clean up the tar file after successful extraction
        os.remove(tar_filepath)
        print("Extraction complete and tar file removed.")
        
        return extract_path

    except tarfile.TarError as e:
        print(f"Error during extraction: {e}")
        return None
    except OSError as e:
        print(f"OS error during file cleanup: {e}")
        return extract_path # Return path even if cleanup failed
    
def loadToken(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
def selectindices(dataset, imagesPerClass, num_classes):
    # 1. Initialize storage for selected indices
    selected_indices = []
    count_per_class = {i: 0 for i in range(num_classes)}

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
        if len(selected_indices) == imagesPerClass * num_classes: # x * 10 classes
            break
        
        
    return selected_indices

def extract_labels(hf_dataset):
    # Directly access the 'label' column of the HF dataset
    return hf_dataset['label']

def getRandomImages(dt_info, dataset, dataset_classes : int):
    
    print("Selecting indices\n")
    
    num_classes = dt_info.num_classes
    images_per_class = dt_info.images_per_class
    
    # 1. Pick num_classes classes
    selected_classes = random.sample(range(dataset_classes), num_classes)

    # 2. Collect indices per class
    class_indices = defaultdict(list)
    
    labels = extract_labels(dataset)

    for idx, label in enumerate(labels):
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
    
    available_class_wnids = dt_info.classes[train_or_validation]
    
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

def imageSelector(dt_info, dataset, dataset_classes : int, train_or_validation, huggingface=False):
    if dt_info.classes[train_or_validation][0] == 'all':
        return getRandomImages(dt_info, dataset, dataset_classes)
    else:
        return getRandomImagesFromClasses(dt_info, dataset, train_or_validation, huggingface=huggingface)


def getClasses(dataset):
    if hasattr(dataset, 'features') and 'label' in dataset.features:
        # This returns the list of WNIDs for ImageNet: ['n01443537', 'n01443538', ...]
        return dataset.features['label'].names
    # Fallback for torchvision/other datasets
    return dataset.classes

def writeDatasetClasses(dt_info):
    classes_file = './dataStorage/datasetClasses.csv'
    ans = findInCsv(classes_file, ['dataset', 'subset', 'num_classes', 'num_images'], [dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images])
    if len(ans) == 0:
        writeCsvLine(classes_file, [dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images, dt_info.classes['train'], dt_info.classes['validation']])
        
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