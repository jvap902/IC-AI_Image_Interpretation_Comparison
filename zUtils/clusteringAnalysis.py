from ..src.plot import *
import numpy as np

datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)] #apenas datasets utilizados no artigo

if __name__ == "__main__":
    
    for (dataset, subset) in datasets:
        