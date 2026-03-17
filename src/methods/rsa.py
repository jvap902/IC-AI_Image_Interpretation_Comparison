import torch
from ..similarity import similarityAnalysis

def rsa(dt_info, dissimilarity_csv_path):
    get_fst_embedding = len(similarityAnalysis.isDissimilarityCalculated(dt_info.name_w_subset, dissimilarity_csv_path, fst_modelc)) == 0 or args.existing_dissimilarity == False
    get_snd_embedding = len(similarityAnalysis.isDissimilarityCalculated(dt_info.name_w_subset, dissimilarity_csv_path, snd_modelc)) == 0 or args.existing_dissimilarity == False    
    
    fst_embedding_path = dataCollection.getSavePath(fst_modelc, dt_info, True)
    snd_embedding_path = dataCollection.getSavePath(snd_modelc, dt_info, True)
    
    # --- Feature Extraction ---

    with torch.no_grad():
        
        if get_fst_embedding:            
            print(f"\n--- Extracting Features for {first_model_name} ---")
            # This function iterates over all batches in val_loader and returns ONE large tensor
            first_features, _ = featureExtraction.getFeatureTensors(fst_modelc.val_loader, fst_modelc)
            torch.save(first_features, fst_embedding_path)
            print(f"\nSaved first embedding tensor (Shape: {first_features.shape}) to: {fst_embedding_path}")
        
        if get_snd_embedding:
            print(f"\n--- Extracting Features for {second_model_name} ---")
            second_features, _ = featureExtraction.getFeatureTensors(snd_modelc.val_loader, snd_modelc)
            torch.save(second_features, snd_embedding_path)
            print(f"Saved second embedding tensor (Shape: {second_features.shape}) to: {snd_embedding_path}")

    # --- Montando matriz ---

    fst_dissimilarity_path = similarityAnalysis.getCosineDissimilarity(fst_embedding_path, dissimilarity_csv=dissimilarity_csv_path, dissimilarity_folder=dissimilarity_folder, modelc=fst_modelc, dt_info=dt_info, existing_dissimilarity=args.existing_dissimilarity)
    
    snd_dissimilarity_path = similarityAnalysis.getCosineDissimilarity(snd_embedding_path, dissimilarity_csv=dissimilarity_csv_path, dissimilarity_folder=dissimilarity_folder, modelc=snd_modelc, dt_info=dt_info, existing_dissimilarity=args.existing_dissimilarity)

    print("\nCalculating Pearson's correlation\n")
    pearson, p_value = similarityAnalysis.calculateCorrelations(fst_dissimilarity_path, snd_dissimilarity_path, correlation_type='pearson')

    print(f"Pearson's Rank Correlation Coefficient (ρ): {pearson:.4f}")

    spearman, p_value = similarityAnalysis.calculateCorrelations(fst_dissimilarity_path, snd_dissimilarity_path, correlation_type='spearman', chunked=args.chunked)

    print(f"Spearman's Rank Correlation Coefficient (ρ): {spearman:.4f}")