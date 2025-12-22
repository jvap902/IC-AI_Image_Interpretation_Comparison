from .modelCreation import getModel
from .classUtils import getExtractor, get_attention_layers, get_conv_layers, is_vit_patch_embedding

class Model:
    
    def __init__(self, model_name, model_source):
        self.name = model_name
        self.source = model_source
        
        self.model, self.data_transforms = getModel(self.source, self.name)
        
        self.featureExtractor = getExtractor(self.source)
        
        self.archType = self.getArchitectureType()
        
    def extract(self, inputs):
        return self.featureExtractor(self, inputs)
    
    def getArchitectureType(self): #ideia bem inicial, provalvemente será necessário um método mais complicado para isso
        
        if self.source == "timm":
            return self.model.default_cfg.get("architecture", "Unknown")
        
        conv_layers = get_conv_layers(self.model)
        att_layers  = get_attention_layers(self.model)

        has_conv = len(conv_layers) > 0
        has_att  = len(att_layers) > 0

        # Count ViT-style patch embeddings
        vit_patch_convs = [c for c in conv_layers if is_vit_patch_embedding(c)]

        # CNN only
        if has_conv and not has_att:
            return "CNN"

        # Transformer only
        if has_att:
            # Pure ViT: attention + exactly one patch conv
            if len(vit_patch_convs) == 1 and len(conv_layers) == 1:
                return "Vision Transformer"

            # Hybrid: CNN backbone + attention
            if len(conv_layers) > 1:
                return "Hybrid (CNN + Transformer)"

            return "Vision Transformer"

        return "Unknown"
        
        