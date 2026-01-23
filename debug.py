from src import plot
import numpy as np
import torch

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import maxvit_t, resnet18, vit_b_16, convnext_tiny, efficientnet_b0

#model = maxvit_t()
#train_nodes, eval_nodes = get_graph_node_names(model)
#print(eval_nodes[-100:]) # View all node names

import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

dataset = 'imagenet-sketch'

#plot.heatMap(f'./dataStorage/{dataset}Data.csv', 'pearson')
#plot.heatMap(f'./dataStorage/{dataset}Data.csv', 'spearman')