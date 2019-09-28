import pandas as pd 
import numpy as np 
def datiMancanti():
    irisDataset = pd.read_csv("data/iris_nan.csv")

    y = irisDataset["class"].values
    x = irisDataset.drop("class", axis = 1).values 
    print(x)
    #SOLUZIONI

    #eliminare i NaN (righe)
    dropR = irisDataset.dropna()

    #eliminare i NaN (colonne)
    dropC = irisDataset.dropna(axis = 1)

    #sostituire i NaN con la media o mediana (.median()) o moda (.mode())
    replace = irisDataset.mean()
    irisDataset_raplace = irisDataset.fillna(replace)
    print(irisDataset_raplace)

def Norm():
    #Se i dati non sono sulla stessa scala, l'algoritmo potrebbe attribuire piu importanza ai valori piu alti 

    wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", usecols=[0,1,7], names=["classe", "alcol", "flavonoidi"])
    y = wine["classe"].values
    x = wine.drop("classe", axis=1)
    print(wine.describe())

    wine_cop = wine.copy()
    #normalizzazione tra 0 e 1 (xi-x_min)/(x_max-x_min)
    #MinMaxScaler per numpy array
    daNormalizzare = ["alcol","flavonoidi"]
    norm = wine[daNormalizzare]
    wine_cop[daNormalizzare] = (norm-norm.min())/(norm.max()-norm.min())
    print(wine_cop)
    #standardizzazione pone la media su 0 con deviazione di 1 xi-x_mean/x_deviazionestandard
    #StandardScaler per numpy array
    wine_cop = wine.copy()
    daStandardizzare= ["alcol","flavonoidi"]
    stand = wine[daStandardizzare]
    wine_cop[daStandardizzare] = (stand-stand.mean())/(stand.std())
    print(wine_cop)
Norm()