import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU

breast_cancer = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                           names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"])

x = breast_cancer.drop(['diagnosis','id'],axis=1).values
y = breast_cancer['diagnosis'].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#creazione rete neurale
#Modello vuoto senza strati o nodi

def principale():
    model = Sequential()
    #1 hidden layer con 12 nodi
    model.add(Dense(12, input_dim=x_train.shape[1], activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="sgd", loss = "binary_crossentropy", metrics=["accuracy"])

    model.summary()
    model.fit(x_train, y_train, epochs=100)

    model.evaluate(x_test, y_test)


model = Sequential()
model.add(Dense(12, input_dim=x_train.shape[1]))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(4))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="sgd", loss = "binary_crossentropy", metrics=["accuracy"])

model.summary()
model.fit(x_train, y_train, epochs=100)

model.evaluate(x_test, y_test)
