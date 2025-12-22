import timm
from src.plot import writeCsvLine

def getScale(n_params):
    if n_params < 10e6:
        return "tiny"
    elif n_params < 30e6:
        return "small"
    elif n_params < 100e6:
        return "base"
    elif n_params < 300e6:
        return "large"
    else:
        return "giant"

def inferDataset(model_name):
    if "in1k" in model_name:
        return "ImageNet-1K"
    return "DATASET"
    
if __name__ == "__main__":

    pretrained_models = timm.list_models(pretrained=True)

    for model_name in pretrained_models:
        model = timm.create_model(model_name, pretrained=False)
        
        cfg = model.default_cfg
        
        url = cfg.get("url")
        hf_id = cfg.get("hf_hub_id")
        arch = cfg.get("architecture")
        num_params = sum(p.numel() for p in model.parameters())
        scale = getScale(num_params)
        dataset = inferDataset(model_name)
        
        data = [model_name, url, hf_id, arch, str(num_params), scale, "TRAINING_TYPE", dataset]
        
        writeCsvLine('modelInfo.csv', data)
