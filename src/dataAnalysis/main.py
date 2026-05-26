import argparse
from src import config
from src.ckaHandler import getCkaData
from src.rsaHandler import getRsaData
from src.codifications import dtNameSubset
from . import clustering, plot, methodComparison, datasetCorrelation

parser = argparse.ArgumentParser()

parser.add_argument("-a", "--analysis", required=False, default='basic', type=str, help="Analysis to be made")
parser.add_argument("-met", "--method", required=False, default='rsa', type=str, help="Method the data was extracted from (rsa or cka), if comparing both, this won't be used")
parser.add_argument("-d", "--dataset", nargs='+', required=False, default=[-1], type=int, help="Dataset index or -1 for all datasets, only the first will be used for analysis which do not compare datasets")
parser.add_argument("-heat", "--heatmap", required=False, default=False, action='store_true', help="Whether or not a heatmap will be generated at the end")

args = parser.parse_args()

def datasetsDataFrames(datasets, method):
    
    df_dict = {}
    
    if method == 'rsa':
        getInfo = getRsaData
    elif method == 'cka':
        getInfo = getCkaData
    
    for (dataset, subset) in datasets:
        df_dict[dtNameSubset((dataset, subset))] = getInfo(dataset, subset)
    
    return df_dict

def getAnalysisData(analysis, method, dataset):
    mult_dt_analyses = ["mrss", "drc"]
    
    if analysis in mult_dt_analyses:
        return datasetsDataFrames(dataset, method)
    
    dataset = dataset[0]
    
    if analysis == 'method comparison':
        return {"rsa": getRsaData(dataset[0], dataset[1]), "cka": getCkaData(dataset[0], dataset[1])}
    
    return getRsaData(dataset[0], dataset[1]) if method == 'rsa' else getCkaData(dataset[0], dataset[1])

if __name__ == "__main__":
    
    datasets = config.datasets if args.dataset[0] == -1 else [config.datasets[d] for d in args.dataset]
    
    data = getAnalysisData(args.analysis, args.method, datasets)
    
    match args.analysis:
        case 'basic':
            print(f"Basic: {data}")
            heat_df = data
        
        case 'clustering':
            dist_matrix = clustering.distMatrix(data)
            cluster_dict = clustering.linkData(dist_matrix, method='average')
            plot.dendrogram(cluster_dict['linkage'], config.cods)

        case 'method comparison':
            df_rsa = data["rsa"]
            df_cka = data["cka"]
            comparison = methodComparison.rsaCka(df_rsa, df_cka)
            print(f"Method comparison: \n{comparison}")
        
        case 'mrss':
            print(f"\nMRSS: \n{datasetCorrelation.MRSS(data, datasets, metric='pearson')}")
        
        case 'drc':
            heat_df = datasetCorrelation.DRC(data, datasets, metric='pearson')
            print(f"\nDRC: \n{heat_df}")
            
    if args.heatmap:
        plot.heatmap(heat_df)