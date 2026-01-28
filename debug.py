from src import plot, fileSystem
import numpy as np
import torch

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import maxvit_t, resnet18, vit_b_16, convnext_tiny, efficientnet_b0

#model = maxvit_t()
#train_nodes, eval_nodes = get_graph_node_names(model)
#print(eval_nodes[-100:]) # View all node names

dataset = 'imagenet-sketch'

#plot.heatMap(f'./dataStorage/results/{dataset}Data.csv', 'pearson')
#plot.heatMap(f'./dataStorage/results/{dataset}Data.csv', 'spearman')

fileSystem.createFile('AAA.csv', 'was,ich,liebe\n')