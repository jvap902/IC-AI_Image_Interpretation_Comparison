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

def embeddingSavePath(m_s, m_name, m_w, dt_w_s):
    
    return f'./dataStorage/model_output/embedding/{m_name.replace('/', '-')}_{m_w}_{m_s}_{dt_w_s.replace('/', '-')}.pt'

def notTuple(inst, dt):
    path = embeddingSavePath(inst[0], inst[1], inst[2], f"{dt[0]}({dt[1]})")
    
    a = torch.load(path, weights_only=True)
    
    if type(a) is tuple:
        return False
    return True

if __name__ == "__main__":
    
    lib = getJsonInfo("tempData/a.json")
    mine = getJsonInfo("dataStorage/ckaData/ILSVRC-imagenet-1k(0)/results.json")
    
    if len(lib) < 27:
        print("erro1")
    if len(mine) < 27:
        print("erro2")
    
    for key in lib:
        dif = abs(lib[key] - mine[key])
        
        if dif > 0.000001:
            print(f"diferença maior, {key} -> {dif:.10f}")
            
    print("concluido")