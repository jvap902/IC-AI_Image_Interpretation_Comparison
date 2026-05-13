#from ..src import *
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from pathlib import Path
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import maxvit_t, resnet18, regnet_y_32gf, vit_b_16, efficientnet_b0, swin_t, convnext_tiny
from transformers import AutoModel, AutoImageProcessor
import clip


#model = regnet_y_32gf()
#train_nodes, eval_nodes = get_graph_node_names(model)
##print(eval_nodes[-30:])
#
#print(model.named_children)
#
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load("ViT-B/32", device=device)#
#print(model.named_children)


#pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
#processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
#model = AutoModel.from_pretrained(
#    pretrained_model_name, 
#    device_map="auto", 
#)
#print(model.named_children)

def embeddingSavePath(m_s, m_name, m_w, dt_w_s):
    
    return f'./dataStorage/model_output/embedding/{m_name.replace('/', '-')}_{m_w}_{m_s}_{dt_w_s.replace('/', '-')}.pt'

def notTuple(inst, dt):
    path = embeddingSavePath(inst[0], inst[1], inst[2], f"{dt[0]}({dt[1]})")
    
    a = torch.load(path, weights_only=True)
    
    if type(a) is tuple:
        return False
    return True

if __name__ == "__main__":
    
    names = ["cifar10Data.csv", "fgvc-aircraftData.csv", "ILSVRC-imagenet-1kData.csv", "imagenet-sketchData.csv"]
    
    for n in names:
        df1 = pd.read_csv(f"dataStorage/rsaData/{n}")
        df2 = pd.read_csv(f"oldResults/{n}")
        
        arr1 = df1["pearson"].to_numpy()
        arr2 = df2["pearson"].to_numpy()
        
        dif = arr1-arr2
        #print(f"{n} difference is {dif}\n")
        
        thresh = 0.0001
        
        if(dif > thresh).any():
            print(f"Existe erro maior que {thresh}, o máximo é {dif.max()}")