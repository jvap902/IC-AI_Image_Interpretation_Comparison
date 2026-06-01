import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from src import config
from src.fileManagement import csvUtils, jsonUtils

def savedRdm(dt_name_w_subset, dissimilarity_csv, instance):

    (src, name, weights) = instance

    params = ['model', 'model_source', 'model_weights', 'dataset']
    values = [name, src, weights, dt_name_w_subset]
    
    ans = csvUtils.findInCsv(dissimilarity_csv, params, values)
    
    return ans

def rdm(embedding):
    
    print(f"Calculating RDM")
    
    embedding_np = embedding.flatten(1).detach().cpu().numpy()
    rdm = 1 - cosine_similarity(embedding_np)
    
    return rdm

def rsm(fst_rdm, snd_rdm):
    
    #algumas rdms foram salvas já no formato 1d, outras no 2d
    fst_rdm_flat = fst_rdm if fst_rdm.ndim == 1 else fst_rdm[np.triu_indices(fst_rdm.shape[0], k=1)]
    snd_rdm_flat = snd_rdm if snd_rdm.ndim == 1 else snd_rdm[np.triu_indices(snd_rdm.shape[0], k=1)]
    
    pearson_values = pearsonr(fst_rdm_flat, snd_rdm_flat)
    spearman_values = spearmanr(fst_rdm_flat, snd_rdm_flat)
    
    return pearson_values, spearman_values

def main(distortion):
    
    datasets = [config.datasets[3]] #imagenet-1k
    datasets.extend([dt for dt in config.datasets if distortion in dt[0].lower()])

    for model in config.instances:

        (src, name, weights) = model

        rdms = []

        for dt in datasets:
            dt_name_w_subset = f"{dt[0].replace('/', '-')}({dt[1]})"

            dissimilarity_csv_path = jsonUtils.getJsonInfo(config.json_info_path, ["cosineDissimilarity"])[0]

            emb_path = f'./dataStorage/model_output/embedding/{name.replace('/', '-')}_{weights}_{src}_{dt_name_w_subset}.pt'
            emb = torch.load(emb_path, weights_only=True)

            rdm_info = savedRdm(dt_name_w_subset, dissimilarity_csv_path, model)
            rdms.append(np.load(rdm_info[0]['path']) if len(rdm_info)>0 else rdm(emb, model, dt_name_w_subset, dissimilarity_csv_path))

        for i in range(len(rdms)):
            for j in range(i+1, len(rdms)):

                dt_name_w_subset1 = f"{datasets[i][0].replace('/', '-')}({datasets[i][1]})"
                dt_name_w_subset2 = f"{datasets[j][0].replace('/', '-')}({datasets[j][1]})"

                (pearson, p_p), (spearman, p_s) = rsm(rdms[i], rdms[j])

                runData = [src, name, weights, dt_name_w_subset1, dt_name_w_subset2, pearson, spearman]

                csvUtils.writeCsvLine(f'dataStorage/distortionData/{distortion}.csv', runData)
        

def modelDistortion(dataset_indices, distortion, model, metric='pearson'):

    datasets = [config.datasets[i] for i in dataset_indices]
    datasets = [f"{dt[0].replace('/', '-')}({dt[1]})" for dt in datasets]

    file_path = f'dataStorage/distortionData/{distortion}.csv'

    csv_data = csvUtils.findInCsv(file_path, params=["model_src", "model_name", "model_weights"], values=[model[0], model[1], model[2]])

    df = pd.DataFrame(columns=datasets, index=datasets)
    for row in csv_data:

        val = float(row[metric])

        df.at[row['dt(subset)1'], row['dt(subset)2']] = val
        df.at[row['dt(subset)2'], row['dt(subset)1']] = val

    np.fill_diagonal(df.values, 1.0)

    return df.astype(float)

if __name__ == "__main__":
    main(distortion='gaussian_noise')