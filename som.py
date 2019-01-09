import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Customer segmentation based on credit data
### Goal is to detect customers who cheated (finding outliers)

dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values # all input data of credit application i.e. amount etc
y = dataset.iloc[:, -1].values # last column 'Application approved' - credit was approved or not

# Feature scaling / Normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1)) # scale all parameters between 0 and 1
X = sc.fit_transform(X) # normalize X according to scaler sc

# Training SOM
from MiniSom

print('all fine')