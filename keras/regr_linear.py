#ATTN! KERAS UTILIZZA TROPPA POTENZA DI CALCOLO PER UNA SEMPLICE REGRESSIONE LINEARE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PRATIO", "B", "LSTAT", "MEDV"]

boston = pd.read_csv(dataset_url, sep="\s+", names = features)
#RM = numero di stanze

x = boston["RM"].values
y = boston["MEDV"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#Modello sequenziale per creare piu strati in questo caso 1
model = Sequential()
#Dense(dimensione output, dimensione input)
layer = Dense(1, input_dim = 1)

model.add(layer)
model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=100)

model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

import matplotlib.pyplot as plt
plt.scatter(x_test, y_test, edgecolors="white")
plt.plot(x_test, y_pred, color="red")
plt.xlabel("Numero di stanze [RM]")
plt.ylabel("Prezzo in $1000 [MEDV]")

#Visualizzazione pesi
model.get_weights()
