import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, random_split
from datasets import load_dataset 
from huggingface_hub import login 
import os
import numpy as np
from PIL import Image
import json

# --- Helper Class to Convert HuggingFace Dataset to PyTorch Dataset ---

class HuggingFaceImageNetDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform
        
        # Load mappings
        self._build_mappings()
        
        # Get the names from the HF dataset features
        if hasattr(hf_dataset, 'features') and 'label' in hf_dataset.features:
            self.hf_classes = hf_dataset.features['label'].names
        else:
            self.hf_classes = []

    def _build_mappings(self):
        # 1. Load the master JSON
        with open("./data/imagenet_class_index.json", "r") as f:
            json_data = json.load(f)
        
        self.name_to_wnid = {}
        self.wnid_to_idx = {}
        
        self.classes = [None] * 1000
        for idx_str, (wnid, name) in json_data.items():
            self.classes[int(idx_str)] = wnid
        
        for idx_str, (wnid, raw_name) in json_data.items():
            official_idx = int(idx_str)
            self.wnid_to_idx[wnid] = official_idx
            
            # Map the full name (e.g., "mailbag, postbag")
            clean_full = raw_name.lower().replace('_', ' ').strip()
            self.name_to_wnid[clean_full] = wnid
            
            # Map every part (e.g., "mailbag" and "postbag" individually)
            for part in raw_name.split(','):
                clean_part = part.lower().replace('_', ' ').strip()
                self.name_to_wnid[clean_part] = wnid

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]
        
        # 1. Identify the WNID of the image
        hf_internal_idx = item['label']
        hf_name = self.hf_classes[hf_internal_idx]
        
        # Clean the name from HF
        clean_hf_name = hf_name.lower().replace('_', ' ').strip()
        
        # Try finding the WNID (Bridge)
        wnid = self.name_to_wnid.get(clean_hf_name)
        if not wnid:
            # Fallback: if HF has "mailbag, postbag" and we only matched "mailbag"
            wnid = self.name_to_wnid.get(clean_hf_name.split(',')[0].strip())
            
        if not wnid:
            raise KeyError(f"Could not map dataset name '{hf_name}' to a WNID.")

        # 2. Get the official index for the model (0-999)
        real_label = self.wnid_to_idx[wnid]
        
        image = item['image'].convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, real_label

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