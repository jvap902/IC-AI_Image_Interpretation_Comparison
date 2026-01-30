from src.plot import heatMap

dataset = 'fgvc-aircraft'

heatMap(f'./dataStorage/results/{dataset}Data.csv', 'pearson')
heatMap(f'./dataStorage/results/{dataset}Data.csv', 'spearman')