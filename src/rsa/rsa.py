from ..extraction.featureExtraction import extractFeatures
from .similarity import similarityAnalysis
from ..fileManagement.jsonUtils import *
from .dataCollection import gatherAdditionalData
from ..fileManagement import csvUtils, defaultPaths

def getRsaPaths(json_path=defaultPaths.jsonInfoPath()):
    fields = ["dissimilarity_folder", "dissimilarity_csv_path", "fst_embedding_path", "snd_embedding_path"]
    
    return getJsonInfo(fields, json_path)

def rsa(dt_info, fst_modelc, snd_modelc, total_images, num_classes, args):
    dissimilarity_folder, dissimilarity_csv_path, fst_embedding_path, snd_embedding_path = getRsaPaths()

    get_fst_embedding = len(similarityAnalysis.isDissimilarityCalculated(dt_info.name_w_subset, dissimilarity_csv_path, fst_modelc)) == 0 or args.existing_dissimilarity == False
    get_snd_embedding = len(similarityAnalysis.isDissimilarityCalculated(dt_info.name_w_subset, dissimilarity_csv_path, snd_modelc)) == 0 or args.existing_dissimilarity == False
    
    extractFeatures(get_fst_embedding, get_snd_embedding, fst_embedding_path, snd_embedding_path, fst_modelc, snd_modelc)

    # --- Montando matriz ---

    fst_dissimilarity_path = similarityAnalysis.getCosineDissimilarity(fst_embedding_path, dissimilarity_csv=dissimilarity_csv_path, dissimilarity_folder=dissimilarity_folder, modelc=fst_modelc, dt_info=dt_info, existing_dissimilarity=args.existing_dissimilarity)
    
    snd_dissimilarity_path = similarityAnalysis.getCosineDissimilarity(snd_embedding_path, dissimilarity_csv=dissimilarity_csv_path, dissimilarity_folder=dissimilarity_folder, modelc=snd_modelc, dt_info=dt_info, existing_dissimilarity=args.existing_dissimilarity)

    print("\nCalculating Pearson's correlation\n")
    pearson, p_value = similarityAnalysis.calculateCorrelations(fst_dissimilarity_path, snd_dissimilarity_path, correlation_type='pearson')

    print(f"Pearson's Rank Correlation Coefficient (ρ): {pearson:.4f}")

    spearman, p_value = similarityAnalysis.calculateCorrelations(fst_dissimilarity_path, snd_dissimilarity_path, correlation_type='spearman', chunked=args.chunked)

    print(f"Spearman's Rank Correlation Coefficient (ρ): {spearman:.4f}")

    runData = [str(total_images), str(num_classes), fst_modelc.source, fst_modelc.name, args.m1_weights, snd_modelc.source, snd_modelc.name, args.m2_weights, str(fst_modelc.acc), str(snd_modelc.acc), str(spearman), str(pearson), dt_info.name_w_subset]

    csvUtils.writeCsvLine(args.output_file, runData)
    
    gatherAdditionalData(fst_modelc, dt_info, has_embedding=get_fst_embedding)
    gatherAdditionalData(snd_modelc, dt_info, has_embedding=get_snd_embedding)