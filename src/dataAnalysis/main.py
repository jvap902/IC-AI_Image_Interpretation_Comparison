import argparse
from src import config
from ckaHandler import getCkaData
from rsaHandler import getRsaData
import clustering, heatmaps, methodComparison

parser = argparse.ArgumentParser()

parser.add_argument("-met", "--method", nargs='+', type=str, help="Insert list of methods the that was extracted from (rsa or cka)")
parser.add_argument("-d", "--dataset", nargs='+', type=int, help="Insert list of datasets indices")
parser.add_argument("-a", "--analyses", nargs='+', type='str', help="Insert list of analyses to be made")

args = parser.parse_args()

if __name__ == "__main__":
    
    dfs = dict()

    datasets = [config.datasets[i] for i in args.dataset]

    if 'rsa' in args.method:
        dfs['rsa'] = getRsaData(args.dataset)
    if 'cka' in args.method:
        dfs['cka'] = getCkaData(args.dataset)

    if 'clustering' in args.analyses:
        dist_matrix = clustering.distMatrix(dfs[next(iter(dfs))])
        cluster_dict = clustering.linkData(dist_matrix, method='average')
        clustering.pltDendrogram(cluster_dict['linkage'], config.cods)

    raise NotImplementedError