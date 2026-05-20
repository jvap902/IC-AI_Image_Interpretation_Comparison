from torchvision.models import get_model, get_model_weights

def loadTorchvisionModel(name, weights):
    model = get_model(name, weights=weights)
    
    weights_enum = get_model_weights(name)
    
    weights_obj = getattr(weights_enum, weights)
    
    preprocess = weights_obj.transforms()
    
    return model, preprocess

if __name__ == "__main__":
    loadTorchvisionModel("resnet18", "IMAGENET1K_V1")