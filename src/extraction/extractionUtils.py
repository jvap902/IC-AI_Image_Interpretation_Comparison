import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def clipExtractor(modelc, inputs):
    data = modelc.model.encode_image(inputs)
    
    if data.dim() > 2:
        data = torch.flatten(data, start_dim=1)
    
    return data.float()

def generalExtractor(modelc, inputs):
    data = modelc.model(inputs)
    
    if data.dim() > 2:
        data = torch.flatten(data, start_dim=1)
    
    return data.float()

def huggingfaceExtractor(modelc, inputs):
    
    inp = modelc.data_transforms(inputs, return_tensors="pt").to(modelc.model.device)
    
    with torch.inference_mode():    
        data = modelc.model(**inp)
    
    return data.float()