from src import plot

plot.heatMap('./dataStorage/cifar100Data.csv', 'spearman')
plot.heatMap('./dataStorage/cifar100Data.csv', 'pearson')