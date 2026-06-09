import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from src import config
from . plot import heatmap
from src.codifications import modelCod
from src.fileManagement import csvUtils, jsonUtils, fileSystem

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

    for model in tqdm(config.instances):

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

                file_path = f'dataStorage/distortionData/{distortion}.csv'
                fileSystem.createFile(file_path, "model_src,model_name,model_weights,dt(subset)1,dt(subset)2,pearson,spearman\n")
                csvUtils.writeCsvLine(file_path, runData)
        

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

def distortionDataFrame(datasets_indices, distortion_type, individual_heatmap=True):
    
    name_map = {'ILSVRC-imagenet-1k(0)': 'base'} #esse é fixo

    for i in datasets_indices:
        dt = config.datasets[i]

        if 'imagenet-1k' in dt[0]:
            continue

        dt_name_w_subset = f"{dt[0].replace('/', '-')}({dt[1]})"
        name_map[dt_name_w_subset] = dt[0][-1]
    
    heat_df = pd.DataFrame(columns=["1", "3", "5"])
    
    for model in config.instances:
        df = modelDistortion(dataset_indices=datasets_indices, distortion=distortion_type, model=model)
        df = df.rename(index=name_map, columns=name_map)
        print(f"\nDistortion comparison {model[1]}: \n{df}")
        
        if individual_heatmap: heatmap(df, title=f"{model[0]}, {model[1]}, {model[2]}", linewidths=0.3, save_path=f"dataStorage/processedResults/distortion/{distortion_type}/{modelCod(model[0], model[1], model[2])}.png", show=False, annot=True)
        
        heat_df.loc[modelCod(model[0], model[1], model[2])] = [df.at['base', '1'], df.at['base', '3'], df.at['base', '5']]

    return heat_df

if __name__ == "__main__":
    main(distortion='motion_blur')