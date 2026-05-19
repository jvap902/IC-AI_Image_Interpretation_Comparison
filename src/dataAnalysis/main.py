import argparse
from src import config
from src.ckaHandler import getCkaData
from src.rsaHandler import getRsaData
from src.codifications import dtNameSubset
from . import clustering, plot, methodComparison, datasetCorrelation

parser = argparse.ArgumentParser()

parser.add_argument("-met", "--method", nargs='+', type=str, help="Insert list of methods the that was extracted from (rsa or cka)")
parser.add_argument("-d", "--dataset", nargs='+', default=[-1], type=int, help="Insert list of datasets indices")
parser.add_argument("-a", "--analyses", nargs='+', type=str, help="Insert list of analyses to be made")

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

if __name__ == "__main__":
    
    if args.dataset[0] == -1:
        datasets = config.datasets
    else:
        datasets = [config.datasets[i] for i in args.dataset]

    dataset, subset = datasets[0]
    
    if 'clustering' in args.analyses:
        df = getCkaData(dataset, subset) if args.method=='cka' else getRsaData(dataset, subset, param='pearson', codify=True)
        dist_matrix = clustering.distMatrix(df)
        cluster_dict = clustering.linkData(dist_matrix, method='average')
        plot.dendrogram(cluster_dict['linkage'], config.cods)
    
    if 'method comparison' in args.analyses:
        df_rsa = getRsaData(dataset, subset, param='pearson', codify=True)
        df_cka = getCkaData(dataset, subset)
        comp = methodComparison.rsaCka(df_rsa, df_cka)
        print(f"Method comparison: \n{comp}")
        
    if 'mrss' in args.analyses:
        df_dict = datasetsDataFrames(datasets, 'rsa')
        mrss = datasetCorrelation.MRSS(df_dict, 'pearson')
        print(f"\nMRSS: \n{mrss}")
        
    if 'drc' in args.analyses:
        df_dict = datasetsDataFrames(datasets, 'rsa')
        drc = datasetCorrelation.DRC(df_dict, 'pearson')
        print(f"\nDRC: \n{drc}")