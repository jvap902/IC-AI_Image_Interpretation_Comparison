import os
import tarfile
import zipfile
from tqdm.auto import tqdm
import requests
from torch.utils.data import Dataset, Subset
import random
from collections import defaultdict
from ..plot import findInCsv, writeCsvLine


# ImageNet-A download details
IMAGENET_A_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
IMAGENET_A_FILENAME = "imagenet-a.tar"
IMAGENET_A_EXTRACT_DIR = "imagenet-a"

# Imagenet-sketch
IMAGENET_SKETCH_URL = "https://www.kaggle.com/api/v1/datasets/download/wanghaohan/imagenetsketch"
IMAGENET_SKETCH_FILENAME = "archive.zip"
IMAGENET_SKETCH_EXTRACT_DIR = "imagenet-sketch"

def getDownloadInfo(dataset):
    if dataset == 'imagenet-a':
        return IMAGENET_A_URL, IMAGENET_A_FILENAME, IMAGENET_A_EXTRACT_DIR, 'tar'
    elif dataset == 'imagenet-sketch':
        return IMAGENET_SKETCH_URL, IMAGENET_SKETCH_FILENAME, IMAGENET_SKETCH_EXTRACT_DIR, 'zip'
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
    
# --- Helper Class to Convert HuggingFace Dataset to PyTorch Dataset ---
class HuggingFaceImageNetDataset(Dataset):
    
    # Assuming the Hugging Face dataset is passed in hf_dataset
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform
        # Assume classes are set up here
        if hasattr(hf_dataset, 'features') and 'label' in hf_dataset.features:
            # Attempt to infer classes from HF dataset
            self.classes = hf_dataset.features['label'].names
        else:
            self.classes = [f"Class {i}" for i in range(100)] # Placeholder
            
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, index):
        # 1. Retrieve the item from the Hugging Face dataset
        item = self.hf_dataset[index]
        image = item['image'] # Assuming the key is 'image'
        label = item['label']
        
        # 2. >>> CRITICAL FIX: Explicitly enforce 3 channels (RGB) at the source
        # This converts grayscale 'L' or RGBA to 3-channel 'RGB' directly on the PIL image.
        # This bypasses any potential worker serialization issues in transforms.Compose.
        try:
            image = image.convert('RGB')
        except Exception as e:
            # Log an error if the conversion fails for a specific image, but continue
            print(f"Warning: Failed to convert image at index {index} to RGB. Error: {e}")
            # If conversion fails, we let the transform run, hoping it recovers (unlikely)
            
        # 3. Apply the rest of the transformation pipeline
        # This includes Resize, CenterCrop, ToTensor, and Normalize.
        if self.transform:
            # This is your original failing line, now called on a guaranteed RGB image
            image = self.transform(image) 
            
        return image, label
    
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

def getRandomImages(dt_info, dataset, dataset_classes : int):
    
    print("Selecting indices\n")
    
    num_classes = dt_info.num_classes
    images_per_class = dt_info.images_per_class
    
    # 1. Pick num_classes classes
    selected_classes = random.sample(range(dataset_classes), num_classes)

    # 2. Collect indices per class
    class_indices = defaultdict(list)

    for idx, item in enumerate(dataset):
        _, label = item
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

def getRandomImagesFromClasses(dt_info, dataset, train_or_validation):
    
    print("Getting classes from another dataset")
    
    available_class_names = dt_info.classes[train_or_validation]
        
    available_classes = []
    
    for idx, name in enumerate(getClasses(dataset)):
        if name in available_class_names:
            available_classes.append(idx)
                
    print(f"Selecting indices from {len(available_classes)} specific classes\n")

    # Collect indices per class
    class_indices = defaultdict(list)

    for idx, item in enumerate(dataset):
        _, label = item
        if label in available_classes:
            class_indices[label].append(idx)   
            
    # Check availability - verifica se existe o número de imagens por classe nesta classe
    for c in available_classes:
        if len(class_indices[c]) < dt_info.images_per_class:
            raise ValueError(
                f"Class {c} only has {len(class_indices[c])} images, "
                f"requested {dt_info.images_per_class}."
            )

    # Select balanced subset
    selected_indices = []
    for c in available_classes:
        indices = class_indices[c]
        rand_img_start = random.randint(0, len(indices)-dt_info.images_per_class)
        selected_indices.extend(indices[rand_img_start : rand_img_start + dt_info.images_per_class]) #sequência aleatória de índices    
    
    print("Indices selected")
                
    return selected_indices

def imageSelector(dt_info, dataset, dataset_classes : int, train_or_validation):
    if dt_info.classes['train'][0] == 'all':
        return getRandomImages(dt_info, dataset, dataset_classes)
    else:
        return getRandomImagesFromClasses(dt_info, dataset, train_or_validation)

def getClasses(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset.classes
    return dataset.classes

def writeDatasetClasses(dt_info):
    classes_file = './dataStorage/datasetClasses.csv'
    ans = findInCsv(classes_file, ['dataset', 'subset', 'num_classes', 'num_images'], [dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images])
    if len(ans) == 0:
        writeCsvLine(classes_file, [dt_info.name, dt_info.subset, dt_info.num_classes, dt_info.num_images, dt_info.classes['train'], dt_info.classes['validation']])