import pandas as pd 
import numpy as np 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

#per pandas
boston_df = pd.DataFrame(data = np.c_[boston["data"], boston["target"]], columns = np.append(boston["feature_names"], "TARGET"))
print(boston_df)

boston_test = boston_df.sample(frac=0.3) 
boston_train = boston_df.drop(boston_test.index)