import os
import tarfile
from urllib.request import urlretrieve
from tqdm.auto import tqdm
import requests
from torch.utils.data import Dataset
import random
from collections import defaultdict


# ImageNet-A download details
IMAGENET_A_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
IMAGENET_A_FILENAME = "imagenet-a.tar"
IMAGENET_A_EXTRACT_DIR = "imagenet-a"

def download_imagenet_a(root_dir):
    """
    Downloads and extracts the ImageNet-A dataset from the corrected Berkeley URL.

    Args:
        root_dir (str): The base directory ('data/') where the dataset will be stored.

    Returns:
        str: The path to the extracted ImageNet-A directory, or None if failed.
    """
    # Ensure the root data directory exists
    os.makedirs(root_dir, exist_ok=True)
    
    # Define paths
    extract_path = os.path.join(root_dir, IMAGENET_A_EXTRACT_DIR)
    tar_filepath = os.path.join(root_dir, IMAGENET_A_FILENAME)
    
    # 1. Check if the dataset is already extracted
    # We check for a common file structure to avoid re-downloading large files
    # The extracted directory should contain subdirectories (classes).
    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 1:
        print(f"ImageNet-A already found and extracted at: {extract_path}")
        return extract_path

    # 2. Download the file
    print(f"Downloading ImageNet-A from: {IMAGENET_A_URL}")
    try:
        response = requests.get(IMAGENET_A_URL, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        with open(tar_filepath, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), 
                             total=total_size//block_size, 
                             unit='KB', 
                             desc=IMAGENET_A_FILENAME):
                f.write(data)
        
        print(f"\nDownload complete. File saved to: {tar_filepath}")

    except requests.exceptions.RequestException as e:
        print(f"Error during download from {IMAGENET_A_URL}: {e}")
        return None

    # 3. Extract the tar file
    print(f"Extracting {IMAGENET_A_FILENAME} to {extract_path}...")
    try:
        with tarfile.open(tar_filepath, 'r') as tar:
            # The tarball contains a single folder named 'imagenet-a'
            # We extract it directly into the root_dir
            tar.extractall(path=root_dir) 
        
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
    
def selectIndexes(dataset, imagesPerClass, num_classes):
    # 1. Initialize storage for selected indices
    selected_indexes = []
    count_per_class = {i: 0 for i in range(num_classes)}

    # 2. Iterate through the dataset's indices
    for i in range(len(dataset)):
        # CIFAR10 stores targets as a list/array attribute or returns them with the data.
        # We access the class label (target) for the current index 'i'.
        label = dataset.targets[i] 

        # 3. Check if we need more samples for this class
        if count_per_class[label] < imagesPerClass:
            selected_indexes.append(i)
            count_per_class[label] += 1
        
        # Optional: Stop early once all classes have x samples
        if len(selected_indexes) == imagesPerClass * num_classes: # x * 10 classes
            break
        
        
    return selected_indexes

def getRandomImages(num_classes, images_per_class, hf_dataset, dataset_classes):
    
    class_interval_start = random.randint(0, dataset_classes - num_classes)
    
    # 1. Pick num_classes classes
    selected_classes = list(range(class_interval_start, class_interval_start+num_classes))

    # 2. Collect indices per class
    class_indices = defaultdict(list)

    for idx, item in enumerate(hf_dataset):
        label = item["label"]
        if label in selected_classes:
            class_indices[label].append(idx)

    # 3. Check availability
    for c in selected_classes:
        if len(class_indices[c]) < images_per_class:
            raise ValueError(
                f"Class {c} only has {len(class_indices[c])} images, "
                f"requested {images_per_class}."
            )

    # 4. Select balanced subset
    selected_indices = []
    for c in selected_classes:
        selected_indices.extend(class_indices[c][:images_per_class])
        
    return selected_indices