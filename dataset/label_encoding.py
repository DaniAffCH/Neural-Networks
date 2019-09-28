import pandas as pd 
import numpy as np 
#label con numero univoco
from sklearn.preprocessing import LabelEncoder
#variabili di comodo
from sklearn.preprocessing import OneHotEncoder

shirt = pd.read_csv("data/shirt.csv", index_col=0)
x = shirt.values

size_mapping = {
    "S":0,
    "M":1,
    "L":2,
    "XL":3
}

shirt["taglia"] = shirt["taglia"].map(size_mapping)


#divido i colori
shirt = pd.get_dummies(shirt, columns=["colore"])

print(shirt.head())