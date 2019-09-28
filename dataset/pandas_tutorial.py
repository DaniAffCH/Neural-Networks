import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

irisDataset = pd.read_csv("data/iris.csv")
x = irisDataset.drop("variety", axis=1)
y = irisDataset["variety"]
#irisDataset.head() visualizza le prime righe

copia = irisDataset.copy()
copia = irisDataset.sample(frac=1)

#posizione
print("Indicizzazione per posizione \n%s" % (copia.iloc[5]))

print("\n\n\n")
#indice
print("Indicizzazione per indice \n%s" % (copia.loc[22]))
print("\n\n\n")
#shape dimensione

#applico una maschera al dataset (maggiore della media)
petaloLungoMASK = irisDataset["petal.length"] > irisDataset["petal.length"].mean()
petaloLungo = irisDataset[petaloLungoMASK]
print("Maschera \n%s" % (petaloLungo))
print("\n\n\n")
#ordinamento
print("Ordinamento \n%s" % (irisDataset.sort_values("petal.length").head()))
print("\n\n\n")
#arrotondare i valori
arrotondati = x.applymap(lambda val:int(round(val, 0)))
print("Arrotondati \n%s" % (arrotondati))

#fillna -> riempire valori non validi
irisDataset.plot(x = "sepal.length", y = "sepal.width", kind="scatter").show()
