import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, random_split
from datasets import load_dataset 
from huggingface_hub import login 
import os
import numpy as np
from PIL import Image

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

# --- FGVC-Aircraft Custom Dataset Class ---
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