from src.plot import heatMap

if __name__ == "__main__":

    datasets = ['timm/mini-imagenet', 'imagenet-sketch', 'cifar10', 'cifar100', 'fgvc-aircraft', 'ILSVRC/imagenet-1k']
    datasets = [datasets[1], datasets[5]]
        
    for dt in datasets:
        
        dataset = dt.replace('/', '-')
        
        print(f"\n--- {dt} ---")
        
        print(f"{dataset} - pearson")
        heatMap(f'./ztempData/{dataset}Data.csv', 'pearson')
        
        print(f"{dataset} - spearman")
        heatMap(f'./ztempData/{dataset}Data.csv', 'spearman')