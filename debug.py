from src import *
import numpy as np
import torch
import torch.nn as nn

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import maxvit_t, resnet18, vit_b_16, convnext_tiny, efficientnet_b0

model = maxvit_t()
train_nodes, eval_nodes = get_graph_node_names(model)
print(eval_nodes[-30:])

for i in range(1, 4):
    model.classifier[-i] = nn.Identity()



dataset = 'imagenet-sketch'

#plot.heatMap(f'./dataStorage/results/{dataset}Data.csv', 'pearson')
#plot.heatMap(f'./dataStorage/results/{dataset}Data.csv', 'spearman')