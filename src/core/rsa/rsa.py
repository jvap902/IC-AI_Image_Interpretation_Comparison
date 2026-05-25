import numpy as np
from src import config
from src.fileManagement.jsonUtils import getJsonInfo
from src.fileManagement.csvUtils import findInCsv, writeCsvLine
from . import similarity
from .dataCollection import gatherAdditionalData

def rsaMethod(dt_info, fst_modelc, fst_emb, snd_modelc, snd_emb, args):
    dissimilarity_csv_path = getJsonInfo(config.json_info_path, ["cosineDissimilarity"])[0]

    # --- Montando matriz ---
    if args.existing_dissimilarity:
        fst_rdm_info = similarity.savedRdm(dt_info.name_w_subset, dissimilarity_csv_path, fst_modelc)
        snd_rdm_info = similarity.savedRdm(dt_info.name_w_subset, dissimilarity_csv_path, snd_modelc)
        fst_rdm = np.load(fst_rdm_info[0]['path']) if len(fst_rdm_info)>0 else similarity.rdm(fst_emb, fst_modelc, dt_info, dissimilarity_csv_path)
        snd_rdm = np.load(snd_rdm_info[0]['path']) if len(snd_rdm_info)>0 else similarity.rdm(snd_emb, snd_modelc, dt_info, dissimilarity_csv_path)
    else:
        fst_rdm = similarity.rdm(fst_emb, fst_modelc, dt_info, dissimilarity_csv_path)
        snd_rdm = similarity.rdm(snd_emb, snd_modelc, dt_info, dissimilarity_csv_path)
        

    print("\nCalculating RSMs\n")
    
    (pearson, p_p), (spearman, p_s) = similarity.rsm(fst_rdm, snd_rdm)

    print(f"Pearson's Rank Correlation Coefficient (ρ): {pearson:.4f}")

    print(f"Spearman's Rank Correlation Coefficient (ρ): {spearman:.4f}")

    runData = [str(dt_info.num_images), str(dt_info.num_classes), fst_modelc.source, fst_modelc.name, args.m1_weights, snd_modelc.source, snd_modelc.name, args.m2_weights, str(fst_modelc.acc), str(snd_modelc.acc), str(spearman), str(pearson), dt_info.name_w_subset]

    writeCsvLine(args.output_file, runData)
    
    gatherAdditionalData(fst_modelc, dt_info)
    gatherAdditionalData(snd_modelc, dt_info)