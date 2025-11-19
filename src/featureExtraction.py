import timm 
import torch
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def globalEmbedding():
    feature_extractor_pooled = timm.create_model('resnet50', pretrained=True, num_classes=0) #zero classes para passar embedding adiante

    feature_extractor_pooled.to(device)
    feature_extractor_pooled.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        pooled_features = feature_extractor_pooled(dummy_input)

    print(f"Pooled feature shape: {pooled_features.shape}")

def lastFeatureMap():
    feature_extractor_unpooled = timm.create_model('resnet50', pretrained=True)
        
    feature_extractor_unpooled.to(device)
    feature_extractor_unpooled.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        unpooled_features = feature_extractor_unpooled.forward_features(dummy_input)

    # The output is a spatial tensor, e.g., [batch_size, channels, height, width]
    print(f"Unpooled feature shape: {unpooled_features.shape}") # Example: torch.Size()


def multiScaleFeatures():
    # Create a feature pyramid extractor
    feature_pyramid_extractor = timm.create_model(
        'resnet50', 
        pretrained=True, 
        features_only=True, 
        out_indices=(1, 2, 3, 4) # Corresponds to different stages of the ResNet
    )

    feature_pyramid_extractor.to(device)
    feature_pyramid_extractor.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Get the feature pyramid
    with torch.no_grad():
        feature_pyramid = feature_pyramid_extractor(dummy_input)

    # The output is a list of tensors, one for each requested index
    print("Feature pyramid shapes:")
    for i, features in enumerate(feature_pyramid):
        print(f"  Index {i}: {features.shape}")
    # Example output:
    # Index 0: torch.Size()
    # Index 1: torch.Size()
    # Index 2: torch.Size()
    # Index 3: torch.Size()

def extract_features_to_tensors(dataloader, model):
    """Extracts features from a dataloader using a feature extractor model and returns them as tensors."""
    features_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting Features"):
            inputs = inputs.to(device)
            features = model(inputs)
            
            features_list.append(features.cpu())
            labels_list.append(labels.cpu())
            
    # Concatenate all batches
    all_features = torch.cat(features_list)
    all_labels = torch.cat(labels_list)
    
    return all_features, all_labels

if __name__ == "__main__":
    multiScaleFeatures()

