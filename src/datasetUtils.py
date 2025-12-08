import os
import tarfile
from urllib.request import urlretrieve
from tqdm.auto import tqdm
import requests
from torch.utils.data import Dataset

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
    """
    Wraps a Hugging Face Dataset object (which holds PIL images) 
    to be compatible with PyTorch's DataLoader.
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        
        # We need classes for compatibility, infer them from the HF dataset if possible
        # or use a placeholder list of 1000 classes for ImageNet-1k compatibility.
        if 'label' in self.hf_dataset.features:
            self.classes = self.hf_dataset.features['label'].names
        else:
            self.classes = [f"Class {i}" for i in range(1000)] # Placeholder
        
        print(f"Dataset has {len(self.classes)} classes.")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # The Hugging Face dataset returns a dictionary with 'image' (as PIL) and 'label'
        item = self.hf_dataset[idx]
        image = item['image']
        label = item['label']

        if self.transform:
            # Apply the PyTorch transforms to the PIL Image
            image = self.transform(image)
            
        return image, label