from src import plot

plot.correlationGraphic("dataStorage/runData.csv", "images_per_class", "correlation", ["spearman", "pearson"])