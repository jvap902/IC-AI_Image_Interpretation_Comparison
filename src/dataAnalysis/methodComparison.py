import numpy as np
from scipy.stats import pearsonr, spearmanr
from src import config

info_path = config.json_info_path

def rsaCka(df_rsa, df_cka, metric='pearson'):
    
    if metric == 'pearson':
        corr = pearsonr
    elif metric == 'spearman':
        corr = spearmanr

    rsa_array = df_rsa.to_numpy()[np.triu_indices(27, 1)]
    cka_array = df_cka.to_numpy()[np.triu_indices(27, 1)]

    p, _ = corr(rsa_array, cka_array)

    model_wise = df_rsa.corrwith(df_cka)
    
    return model_wise, p