import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.metrics as sm

import pandas as pd
import numpy as np

iris = datasets.load_iris()

x = pd.DataFrame(iris.data, columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'])

y = pd.DataFrame(iris.target, columns = ['Targets'])

from sklearn.preprocessing import StandardScaler

# Create scaler: scaler
scaler = StandardScaler()

# Fit_transform scaler to 'X'
X_norm = scaler.fit_transform(x)

from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
