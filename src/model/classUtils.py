from ..extraction.extractionUtils import clipExtractor, generalExtractor, huggingfaceExtractor
import torch.nn as nn

def getExtractor(model_type):
    if (model_type == 'clip'):
        return clipExtractor
    elif (model_type == 'huggingface'):
        return huggingfaceExtractor
    else:
        return generalExtractor
    

def get_conv_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]

def get_attention_layers(model):
    attention_keywords = (
        "Attention",
        "MultiheadAttention",
        "Transformer",
        "SelfAttention",
        "MHSA",
    )

    return [m for m in model.modules() if any(k in m.__class__.__name__ for k in attention_keywords)]


def is_vit_patch_embedding(conv: nn.Conv2d):
    return (
        conv.in_channels == 3
        and conv.kernel_size == conv.stride
        and conv.kernel_size[0] >= 8
    )
