from src import plot
import numpy as np

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import maxvit_t, resnet18, vit_b_16, convnext_tiny, efficientnet_b0

model = maxvit_t()
train_nodes, eval_nodes = get_graph_node_names(model)
print(eval_nodes[-100:]) # View all node names

dataset = 'imagenet-sketch'

#plot.heatMap(f'./dataStorage/{dataset}Data.csv', 'pearson')
#plot.heatMap(f'./dataStorage/{dataset}Data.csv', 'spearman')

#import torch
#from transformers import AutoImageProcessor, AutoModel
#from transformers.image_utils import load_image
#
#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = load_image(url)
#
#pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
#processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
#model = AutoModel.from_pretrained(
#    pretrained_model_name, 
#    device_map="auto", 
#)
#
#inputs = processor(images=image, return_tensors="pt").to(model.device)
#with torch.inference_mode():
#    outputs = model(**inputs)
#
#pooled_output = outputs.pooler_output
#print("Pooled output shape:", pooled_output.shape)
#print(type(inputs))