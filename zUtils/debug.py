#from ..src import *
import numpy as np
import torch
import torch.nn as nn

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

if __name__ == "__main__":
    import pandas as pd

    # Create a sample DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print("Original DataFrame:")
    print(df)

    # Add a new column 'C' with a list of values
    # The length of the list must match the length of the DataFrame
    df['C'] = [7, 8, 9]
    print("\nDataFrame after direct assignment of list:")
    print(df)

    # Add a new column 'D' with a single scalar value (broadcasts to all rows)
    df['D'] = 10
    print("\nDataFrame after direct assignment of scalar value:")
    print(df)
