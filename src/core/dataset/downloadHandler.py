import os
import tarfile
import zipfile
import requests
import torchvision

from utils.image_sorter import sort_imagenet1k_images

url_downloaded = ['imagenet-a', 'imagenet-sketch', 'imagenet-c', 'imagenet-1k-2012']

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

#ImageNet-1K
IMAGENET_1K_URL = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
IMAGENET_1K_FILENAME = "ILSVRC2012_img_val.tar"
IMAGENET_1K_EXTRACT_DIR = "imagenet-1k/validation"

#Imagenet-C
def imagenetCDownloadInfo(distortion):
    
    tp = imagenetCDistortionMap(distortion)
    
    imagenet_c_url = f"https://zenodo.org/records/2235448/files/{tp}.tar?download=1"
    imagenet_c_filename = f"{tp}.tar"
    imagenet_c_extract_dir = tp
    
    return imagenet_c_url, imagenet_c_filename, imagenet_c_extract_dir

def getDownloadInfo(dataset):
    match dataset:
        case 'imagenet-a':
            return IMAGENET_A_URL, IMAGENET_A_FILENAME, IMAGENET_A_EXTRACT_DIR, 'tar'
        case 'imagenet-sketch':
            return IMAGENET_SKETCH_URL, IMAGENET_SKETCH_FILENAME, IMAGENET_SKETCH_EXTRACT_DIR, 'zip'
        case 'fgvc-aircraft':
            return AIRCRAFT_URL, AIRCRAFT_FILENAME, AIRCRAFT_EXTRACT_DIR, 'tar'
        case s if 'imagenet-c' in s:
            url, filename, extract_dir = imagenetCDownloadInfo(dataset.split('-')[-2])
            return url, filename, extract_dir, 'tar'
        case _:
            raise ValueError("Unsupported dataset")

def downloadUrlDataset(root_dir, url, file_name, extract_dir, compression_type):
    """
    Downloads and extracts the dataset

    Args:
        root_dir (str): The base directory ('data/') where the dataset will be stored.

    Returns:
        str: The path to the extracted dataset, or None if failed.
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
    
def getUrlDataset(data_dir, dataset):
    url, file_name, extract_dir, compression_type = getDownloadInfo(dataset)
    
    # 1. Download and extract the data
    # This calls the function from src/datasetUtils.py to handle the download
    folder_path = downloadUrlDataset(root_dir=data_dir, url=url, file_name=file_name, extract_dir=extract_dir, compression_type=compression_type)
    if folder_path is None:
        raise FileNotFoundError(f"Failed to download or extract {dataset}.")

    folder_path = folder_path+pathConcat(dataset)
    
    if dataset == 'imagenet-1k-2012':
        sort_imagenet1k_images()

    # 2. Load data using ImageFolder (which expects class subdirectories)
    full_dataset = torchvision.datasets.ImageFolder(root=folder_path)
    
    if not full_dataset.classes:
        raise ValueError(f"Could not find any classes (subdirectories) in {data_dir}. Check the directory structure.")
    
    return full_dataset