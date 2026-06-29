import torch

if __name__ == "__main__":
    print(torch.hub.list("facebookresearch/dino:main", force_reload=True))
    #dino_vits = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    #print(dino_vits.named_children)