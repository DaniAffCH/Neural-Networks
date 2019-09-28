import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PRATIO", "B", "LSTAT", "MEDV"]

boston = pd.read_csv(dataset_url, sep="\s+", names = features)

x = boston.drop("MEDV", axis=1).values
y = boston["MEDV"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#porto su scala comune
ss = StandardScaler()
X_train = ss.fit_transform(x_train)
X_test = ss.fit(x_test)

model = Sequential()
model.add(Dense(1, input_dim = X_train.shape[1]))

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100)

model.evaluate(X_test, y_test)
