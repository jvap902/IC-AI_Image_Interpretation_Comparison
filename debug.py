from src import *
import numpy as np
import torch
import torch.nn as nn

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import maxvit_t, resnet18, regnet_y_32gf, vit_b_16, efficientnet_b0, swin_t, convnext_tiny
from transformers import AutoModel, AutoImageProcessor


#model = regnet_y_32gf()
#train_nodes, eval_nodes = get_graph_node_names(model)
#print(eval_nodes[-30:])
#
#print(model.named_children)


pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)
print(model.named_modules)
