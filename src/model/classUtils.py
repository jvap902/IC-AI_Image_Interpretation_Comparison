from ..extraction.extractionUtils import clipExtractor, generalExtractor, huggingfaceExtractor

def getExtractor(model_type):
    if (model_type == 'clip'):
        return clipExtractor
    elif (model_type == 'huggingface'):
        return huggingfaceExtractor
    else:
        return generalExtractor