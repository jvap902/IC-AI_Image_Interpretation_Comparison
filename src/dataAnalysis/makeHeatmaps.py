from src.plot import heatMap

if __name__ == "__main__":

    datasets = [('imagenet-sketch', 1), ('cifar10', 0), ('fgvc-aircraft', 0), ('ILSVRC/imagenet-1k', 0)]
        
    save_folder=f'./dataStorage/processedResults/model-model'
    csv_folder = f'./dataStorage/results/'

    for (dataset, subset) in datasets:

        extension = 'eps'
        
        dt = dataset.replace('/', '-')
        
        print(f"\n--- {dt} ---")
        
        print(f"{dataset} - pearson")
        heatMap(f'{csv_folder}/{dt}Data.csv', 'pearson', specific_value=(['dataset'], [f'{dt}({subset})']), save_path=f'{save_folder}/pearson/{dt}.{extension}', show=False, codification=True, extension=extension)
        
        #print(f"{dataset} - spearman")
        #heatMap(f'{csv_folder}/{dt}Data.csv', 'spearman', specific_value=(['dataset'], [f'{dt}({subset})']), save_path=f'{save_folder}/spearman/{dt}.{extension}', show=False, codification=True, extension=extension)