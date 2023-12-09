from RegressionTree import RegressionTree
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sys

def fitModel(x , y):
    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    for depth in range(30):
        # Create model and fit
        tree = RegressionTree(pd.DataFrame(x_train), y_train, max_depth = depth)
        tree.fit()
        # evaluate predictions
        predict = tree.predictFrame(pd.DataFrame(x_test))
        error = mean_squared_error(y_test, predict)
        print("Test  - depth : {:2},  error : {}".format(depth, error))

        predict = tree.predictFrame(pd.DataFrame(x_train))
        error = mean_squared_error(y_train, predict)
        print("Train - depth : {:2},  error : {}".format(depth, error))


file = "DecFeb.csv"

rawdata = np.loadtxt(file, delimiter=",", dtype=str)
rawdata = rawdata[1:]

# x = 5 -> -1
x = list(map(lambda x : x[5:-1], rawdata))
# Ensure numeric datatype
x = pd.DataFrame(list(map(lambda innerList : list(map(float, innerList)), x)))
# y_electrical = 1
y_electrical = list(map(lambda x : float(x[1]), rawdata))
# y_thermal = 2
y_thermal = list(map(lambda x : float(x[2]), rawdata))
# y_cooling = 3
y_cooling = list(map(lambda x : float(x[3]), rawdata))
# y_heating = 4
y_heating = list(map(lambda x : float(x[4]), rawdata))

with open('output.txt', 'w') as sys.stdout:
    print("electrical")
    fitModel(x, y_electrical)
    print("thermal")
    fitModel(x, y_thermal)
    print("cooling")
    fitModel(x, y_cooling)
    print("heating")
    fitModel(x, y_heating)