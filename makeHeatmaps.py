from src.plot import heatMap

datasets = ['timm/mini-imagenet', 'imagenet-sketch', 'cifar10', 'cifar100', 'fgvc-aircraft']
    
    
for dt in datasets:
    
    dataset = dt.replace('/', '-')
    
    heatMap(f'./dataStorage/results/{dataset}Data.csv', 'pearson')
    heatMap(f'./dataStorage/results/{dataset}Data.csv', 'spearman')