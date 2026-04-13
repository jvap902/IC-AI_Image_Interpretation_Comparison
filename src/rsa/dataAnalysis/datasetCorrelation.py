from ...csvUtils import *
from typing import List,Tuple
from src.codifications import *
from seaborn import heatmap, barplot, color_palette
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from ...fileSystem.fileSystem import getJsonInfo
import json

results_folder = getJsonInfo()["rsaData"]
datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)]
metrics = ['pearson', 'spearman']
output_folder = "dataStorage/processedResults/datasetCorrelations"

instances = getInstances()

def getDataFrames(metric, results_folder=results_folder, datasets: List[Tuple[str, int]] = datasets):

    df_dict = {}

    for dt in datasets:
        dt_name_w_subset = dtNameSubset(dt)

        csv_path = f"{results_folder}/{dt[0].replace('/', '-')}Data.csv"

        data = findInCsv(csv_path, ['dataset'], [dt_name_w_subset])

        df_dict[dt_name_w_subset] = dataFrameFromData(data, metric)

    return df_dict


def valueToColor(value):
    if value > 0.8:
        return "white"
    if value > 0.6:
        return "light_orange"
    if value > 0.4:
        return "orange"
    if value > 0.2:
        return "red"
    if value > 0.0:
        return "pink"
    if value > -0.2:
        return "purple"
    
    return "black"

def dtNameToAcr(n):
    match n:
        case 'imagenet-sketch':
            return 'sk'
        case 'cifar10':
            return 'cf'
        case 'fgvc-aircraft':
            return 'air'
        case 'ILSVRC/imagenet-1k':
            return 'ik'
        case _:
            raise


def savePkl(df, pkl_path, field_name, metric='pearson', json_path=f"{output_folder}/pklPaths.json"):
    
    df.to_pickle(pkl_path)    
    
    with open(json_path, "r+") as f:
        json_data = json.load(f)
        
    json_data[metric][field_name] = pkl_path
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
        

def loadPkl(model_str="main", metric='pearson', json_path=f"{output_folder}/pklPaths.json"):
    with open(json_path, 'r+') as f:
        json_data = json.load(f)
    
    pkl_path = json_data[metric][model_str]
    
    df = pd.read_pickle(pkl_path)
    
    return df    


def dtCorrelationHeatmaps(dic, metric):
    for e in instances:
        df = pd.DataFrame(dic[e])

        names = dtNameSubset(datasets)

        df.columns = names
        df.index = names

        #print(df)
        plt.figure(figsize=(4, 3))
            
        heatmap(df, vmin=-0.5, vmax=1.0)
        
        model_str = getModelTrainStr(e[0], e[1], e[2])
        
        pkl_path = f"{output_folder}/pkls/{metric}-{model_str}.pkl"

        savePkl(df, pkl_path, model_str, metric)
        
        plt.title(str(e))
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{metric}/{model_str}.png")
        #plt.show()


def dtCorrelationGrouping(dic, metric):
    
    #sk = sketch, cf = cifar, ik = imagenet-1k, air = aircraft
    
    similar_bhvs = dict() #dicionario de dicionários de listas de modelos com comportamentos semelhantes
    similar_bhvs["abstract"]=[]
    similar_bhvs["sk_cf"]=dict()
    similar_bhvs["sk_air"]=dict()
    similar_bhvs["sk_ik"]=dict()
    similar_bhvs["cf_air"]=dict()
    similar_bhvs["cf_ik"]=dict()
    similar_bhvs["air_ik"]=dict()
    
    for key, value in similar_bhvs.items():
        if key=="abstract":
            continue
        value["black"] = []
        value["purple"] = []
        value["pink"] = []
        value["red"] = []
        value["orange"] = []
        value["light_orange"] = []
        value["white"] = []
    
    acronyms = [dtNameToAcr(dt[0]) for dt in datasets]
    
    for e in instances:
        for i, d1 in enumerate(acronyms):
            for j, d2 in enumerate(acronyms[i+1:]):
                val = dic[e][i][j+i+1]
                color = valueToColor(val)
                similar_bhvs[f"{d1}_{d2}"][color].append((getModelTrainStr(e[0], e[1], e[2]), val))
                
    for key, value in similar_bhvs.items():
        if key=="abstract":
            continue
        
        abst = ', '.join([f"{color}: {len(v)}" for color, v in value.items()])
        abst = f"{key} -> " + abst
        
        similar_bhvs['abstract'].append(abst)
    
    with open(f'{output_folder}/{metric}Data.json', 'w') as f:
        json.dump(similar_bhvs, f, indent=4)
        
    return similar_bhvs


def MtoMDatasetCorrelation(metric):
    df_dict = getDataFrames(metric)

    dic = {}
    for e in instances:
        dic[e] = np.zeros((len(datasets), len(datasets)))
    #data = np.zeros((len(datasets), len(datasets)))

    for i in range(len(datasets)):
        for j in range(i):
            dt1_name = dtNameSubset(datasets[i])
            dt2_name = dtNameSubset(datasets[j])

            df1 = df_dict[dt1_name]
            df2 = df_dict[dt2_name]

            correlations = df1.corrwith(df2, method=metric)

            for idx, e in enumerate(instances):
                val = correlations[instances[idx]]

                dic[e][i][j] = val
                dic[e][j][i] = val

    for e in instances:
        _ = np.fill_diagonal(dic[e], 1.0)
        
    dtCorrelationHeatmaps(dic, metric)
    
    similar_bhv = dtCorrelationGrouping(dic, metric)
        
        
def generalDatasetCorrelation(metric):
    df_dict = getDataFrames(metric)

    data = np.zeros((len(datasets), len(datasets)))

    for i in range(len(datasets)):
        dt1_name = dtNameSubset(datasets[i])
        df1 = df_dict[dt1_name]
        
        for j in range(i):
            dt2_name = dtNameSubset(datasets[j])
            df2 = df_dict[dt2_name]
            
            dt1 = df1.to_numpy()[np.triu_indices(n=len(instances), k=1, m=len(instances))]
            dt2 = df2.to_numpy()[np.triu_indices(n=len(instances), k=1, m=len(instances))]

            if metric == 'pearson':
                val, _ = pearsonr(dt1, dt2)
            elif metric == 'spearman':
                val, _ = spearmanr(dt1, dt2)
            else:
                raise
            
            data[i][j] = val
            data[j][i] = val

    _ = np.fill_diagonal(data, 1.0)


    df = pd.DataFrame(data)

    names = dtPaperName(datasets)

    df.columns = names
    df.index = names

    #print(df)
    
    pkl_path = f"{output_folder}/pkls/{metric}.pkl"

    savePkl(df, pkl_path, "main", metric)
    
    plt.figure(figsize=(10, 8))
    
    ax = heatmap(df, vmin=-0.5, vmax=1.0)
    
    plt.title(f"Correlação de {metric} entre datasets")
    plt.tight_layout()
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(f"{output_folder}/{metric}.eps", format='eps', dpi=100)
    #plt.show()
    
    return df


def modelAccAvg(model_val_df):
    
    accs = []
    
    for dt_name in dtNameSubset(datasets):
        
        dt_info = model_val_df[model_val_df["dataset"] == dt_name]
        
        accs.append(dt_info["accuracy"].values[0])
            
    return np.array(accs).mean()


def MRSSBarChart(df, bar_param, extension='png', dpi=100):
    
    drop_metric = [m for m in metrics if m != bar_param]
    drop_metric = drop_metric[0] #funciona pois consideramos apenas 2 métricas
    
    df['Model'] = [getModelTrainStr(row["model_source"], row["model_name"], row["model_weights"]) for index, row in df.iterrows()]
    
    plt.figure(figsize=(10, 5))
    ax = barplot(df, x='Model', y=f'{bar_param}_avg', hue=f'{bar_param}_avg', legend=False, palette=color_palette("flare_r", as_cmap=True), saturation=1.0)
    plt.yticks(np.arange(0.0, 1.0, 0.1))
    ax.set(ylabel="Model's results average dataset correlation")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/model-wise.{extension}", format=extension, dpi=dpi)
    plt.show()
    

def MRSS(output_csv_path=f"{output_folder}/mrss.csv", validation_csv_path=f"dataStorage/validation_results.csv", bar_param='pearson', extension='png'):
    
    df = pd.DataFrame(columns=["model_source", "model_name", "model_weights", "pearson_avg", "pearson_median", "spearman_avg", "spearman_median", "acc_avg"])
    
    validation_df = pd.read_csv(validation_csv_path, index_col=["model","model_source","model_weights"])
    
    for model in instances:
    
        model_str = getModelTrainStr(model[0], model[1], model[2])
        
        #new_data expects specific order ["model_source", "model_name", "model_weights", "pearson_avg", "pearson_median", "spearman_avg", "spearman_median", "acc_avg"]
        new_data = [model[0], model[1], model[2]] 
        
        m_val_df = validation_df.loc[(model[1], model[0], model[2])]
        
        for metric in metrics:
        
            m_df = loadPkl(model_str, metric)
            
            matrix = m_df.to_numpy()
            
            upper_triang = matrix[np.triu_indices_from(matrix, k=1)]
            
            metric_avg = upper_triang.mean()
            metric_med = np.median(upper_triang)
            
            new_data.extend([metric_avg, metric_med])
        
        new_data.append(modelAccAvg(m_val_df))
        
        df.loc[len(df)] = new_data
        
    df.set_index(['model_source', 'model_name', 'model_weights'])
    
    print(df)
    
    #df.to_csv(output_csv_path, mode='w', header=True, index=False, index_label="modelo")
    
    MRSSBarChart(df, bar_param, extension=extension)
    

def corrAccMSRR(mrss_csv=f"{output_folder}/mrss.csv"):
    df = pd.read_csv(mrss_csv)
    
    accs = df['acc_avg'].to_numpy()
    p_avg = df['pearson_avg'].to_numpy()
    p_med = df['pearson_median'].to_numpy()
    s_avg = df['spearman_avg'].to_numpy()
    s_med = df['spearman_median'].to_numpy()
    
    data = np.array([p_avg, p_med, s_avg, s_med])
    
    for arr in data:
        p, _ = pearsonr(arr, accs)
        s, _ = spearmanr(arr, accs)
        
        print(f"Pearson correlation: {p}\nSpearman correlation: {s}\n")


if __name__ == "__main__":
    metric=metrics[0]
    #MtoMDatasetCorrelation(metric)
    #generalDatasetCorrelation(metric)
    MRSS(extension='eps')
    #corrAccMSRR()