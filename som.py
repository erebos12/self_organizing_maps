import pandas as pd
import numpy as np

### Customer segmentation based on credit data
### Goal is to detect customers who cheated (finding outliers) / fraud detection

dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values  # all input data of credit application i.e. amount etc
y = dataset.iloc[:, -1].values  # last column 'Application approved' - credit was approved or not

# Feature scaling / Normalization
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))  # scale all parameters between 0 and 1
X = sc.fit_transform(X)  # normalize X according to scaler sc

# Training SOM
# -> finding clusters in dataset X
from minisom import MiniSom

#   -> input_len=number of columns in X to consider
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualization
from pylab import bone, pcolor, colorbar, plot, show

# print(som.distance_map().T)
bone()
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']
# for i, x in enumerate(X):
#    w = som.winner(x)  # winning node of customer x on index i
#    plot(w[0] + 0.5,
#         w[1] + 0.5,
#         markers[y[i]],
#         markeredcolor=colors[y[i]],
#         markerfacecolor='None',
#         markersize=10,
#         markeredgewidth=2)
# show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis=0)
frauds = sc.inverse_transform(frauds) # de-normalize back to original values
print(frauds)

