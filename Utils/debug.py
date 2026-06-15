#from ..src import *
import os
import re
import torch
from pathlib import Path
from datasets import load_dataset
from src.core.dataset.loadDataset import loadUrlDownloadedDataset
import matplotlib.pyplot as plt



#model = regnet_y_32gf()
#train_nodes, eval_nodes = get_graph_node_names(model)
##print(eval_nodes[-30:])
#
#print(model.named_children)
#
#model, preprocess = clip.load("ViT-B/32", device=device)#
#print(model.named_children)


#pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
#processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
#model = AutoModel.from_pretrained(
#    pretrained_model_name, 
#    device_map="auto", 
#)
#print(model.named_children)

def get_hf_equivalent_index(local_subset, subset_idx):
    """
    Maps an index from a PyTorch local Subset to its exact 
    equivalent index in the Hugging Face ImageNet-1k validation split.
    """
    # 1. Get the global index of the underlying dataset
    global_idx = local_subset.indices[subset_idx]
    
    # 2. Get the original file path from the underlying ImageFolder dataset
    # (ImageFolder stores tuples of (path, class_int) in .samples)
    file_path, local_label = local_subset.dataset.samples[global_idx]
    filename = os.path.basename(file_path)  # e.g., "ILSVRC2012_val_00000031.JPEG"
    
    # 3. Extract the original sequence number using regex
    match = re.search(r"val_(\d+)\.", filename)
    if not match:
        raise ValueError(f"Could not parse the standard ImageNet validation number from: {filename}")
        
    file_number = int(match.group(1))  # e.g., 31
    
    # 4. Map to Hugging Face 0-based index
    hf_idx = file_number - 1
    
    return hf_idx, file_path

if __name__ == "__main__":
    
    path = Path('dataStorage/model_output/embedding')
    
    for f in path.iterdir():
        if 'imagenet-c' in f.name:
            f.unlink()