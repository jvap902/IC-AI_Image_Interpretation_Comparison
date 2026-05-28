#from ..src import *
import torch
import open_clip
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import maxvit_t, resnet18, regnet_y_32gf, vit_b_16, efficientnet_b0, swin_t, convnext_tiny
from transformers import AutoModel, AutoImageProcessor
import clip
from src.fileManagement.jsonUtils import getJsonInfo


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

if __name__ == "__main__":
    
    folder_path = Path('dataStorage/model_output/embedding')

    for file in folder_path.iterdir():

        if "imagenet-c-gaussian_noise-1(0)" in file.name:
            print(file.name)
            file.unlink()
