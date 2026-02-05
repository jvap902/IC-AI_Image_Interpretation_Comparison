from src.plot import heatMap

datasets = ['timm/mini-imagenet', 'imagenet-sketch', 'cifar10', 'cifar100', 'fgvc-aircraft']
    
    
for dt in datasets[4:]:
    
    dataset = dt.replace('/', '-')
    
    print(f"\n--- {dt} ---")
    
    print(f"{dataset} - pearson")
    heatMap(f'./dataStorage/results/{dataset}Data.csv', 'pearson')
    
    print(f"{dataset} - spearman")
    heatMap(f'./dataStorage/results/{dataset}Data.csv', 'spearman')