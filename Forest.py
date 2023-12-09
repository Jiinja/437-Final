# import all necessary libraries
import matplotlib.ticker
import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.linear_model import Ridge
from sklearn import linear_model
from RegressionTree import RegressionTree
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import sys


# This has the same code as the google colab.


def fitModel(x, y):
    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    for depth in range(30):
        # Create model and fit
        tree = RegressionTree(pd.DataFrame(x_train), y_train, max_depth=depth)
        tree.fit()
        # evaluate predictions
        predict = tree.predictFrame(pd.DataFrame(x_test))
        error = mean_squared_error(y_test, predict)
        print("Test  - depth : {:2},  error : {}".format(depth,
                                                         error))  # TODO: we can independently determine which depth to use later

        predict = tree.predictFrame(pd.DataFrame(x_train))
        error = mean_squared_error(y_train, predict)
        print("Train - depth : {:2},  error : {}".format(depth, error))

    calculate_metrics(y_test, predict, f"Test - depth: {depth}")

    predict_train = tree.predictFrame(pd.DataFrame(x_train))
    calculate_metrics(y_train, predict_train, f"Train - depth: {depth}")


def ForestTrain(n_trees, x_train, y_train, max_depth, sklearn_flag, max_features):
    Forest = []  # list of trees
    samples = []  # list of bootstrap keys

    for i in range(0, n_trees):
        ## 1. Create Bootstrapped Dataset ###
        ## Randomly select samples from the original dataset (xtrain)
        ## And have this dataset be the same size as the original set.
        n = x_train.shape[0]
        sample = np.random.randint(n,size=n)  # sample (with replacement) a set of indicies of n test samples # this is our bootstrap key
        samples.append(sample)
        if sklearn_flag:  # use sklearn regressor
            tree = DecisionTreeRegressor(max_depth=max_depth, max_features=max_features)
            tree.fit(x_train.loc[sample], pd.DataFrame(y_train).loc[sample].T.values[0])
        else:  # use ours
            tree = RegressionTree((x_train.loc[sample]), pd.DataFrame(y_train).loc[sample].T.values[0],
                                  max_depth=max_depth)
            tree.fit()
        Forest.append(tree)  # add bootstrapped fit tree to forest.
    return Forest, samples


def ForestPredict(n_trees, sklearn_flag, Forest, x_test, y_test):
    predictions = []  # list of output of all trees
    errors = []  # list of how off each prediction is (for evaluating model later)
    for i in range(0, n_trees):
        if sklearn_flag:
            predict = Forest[i].predict(pd.DataFrame(x_test))
        else:
            predict = Forest[i].predictFrame(pd.DataFrame(x_test))
        predictions.append(predict)
        error = mean_squared_error(y_test, predict)  # maybe we can use other metrics to compare individual trees later.
        errors.append(error)
    return predictions, errors


def calculate_metrics(actual, predicted, label):
    # Calculate R^2 value
    r2 = r2_score(actual, predicted)
    print(f"{label} R^2 Score:", r2)

    # Calculate mean squared error
    mse = mean_squared_error(actual, predicted)
    print(f"{label} Mean Squared Error:", mse)

    # Calculate root mean squared error
    rmse = np.sqrt(mse)
    print(f"{label} Root Mean Squared Error:", rmse)

    # Calculate mean absolute error
    mae = mean_absolute_error(actual, predicted)
    print(f"{label} Mean Absolute Error:", mae)


def obtain_data(file, label):
    rawdata = np.loadtxt(file, delimiter=",", dtype=str)
    rawdata = rawdata[1:]

    # x = 5 -> -1
    x = list(map(lambda x: x[5:-1], rawdata))
    # Ensure numeric datatype
    x = pd.DataFrame(list(map(lambda innerList: list(map(float, innerList)), x)))
    # y_electrical = 1
    y_electrical = list(map(lambda x: float(x[1]), rawdata))
    # y_thermal = 2
    y_thermal = list(map(lambda x: float(x[2]), rawdata))
    # y_cooling = 3
    y_cooling = list(map(lambda x: float(x[3]), rawdata))
    # y_heating = 4
    y_heating = list(map(lambda x: float(x[4]), rawdata))

    if label == 'electrical':
        y_label = y_electrical
    elif label == 'thermal':
        y_label = y_thermal
    elif label == 'cooling':
        y_label = y_cooling
    elif label == 'heating':
        y_label = y_heating
    else:
        y_label = y_electrical

    return x, y_label

def sweeping_forecast(datasets, label):
    for dataset in datasets:
        x, y = obtain_data(dataset, label)
        forecasting_outputs = [round(len(y) / 24 * 0.9), round(len(y) / 24 * 0.75), round(len(y) / 24 * 0.5), round(len(y) / 24 * 0.25), round(len(y) / 24 * 0.10)]
        for forecasting_output in forecasting_outputs: #looping for showing different forecast output lenghts
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24*forecasting_output / x.shape[0], shuffle=False) #Shuffle is false due to temporal constraints.
            max_features = None
            n_trees = 100
            max_depth = 10
            Forest, samples = ForestTrain(n_trees, x_train, y_train, max_depth, True, max_features)
            predictions, errors = ForestPredict(n_trees, True, Forest, x_test, y_test)
            True_Prediction = pd.Series(np.median(pd.DataFrame(predictions), axis=0))

            # Linear comparison ##
            # reg = linear_model.LinearRegression()
            # reg.fit(x_train, y_train)
            # predict_linear = reg.predict(x_test)
            #
            ## Plotting ##
            plt.plot(range(0, len(y)), y, label='Real Measured Data')
            plt.plot(range(len(y)-len(True_Prediction.index), len(y)), True_Prediction.values, label="Forest Predicted Output")
            #plt.plot(range(len(y) - len(True_Prediction.index), len(y)), predict_linear,label="Linear Predicted Output")
            plt.axvspan(len(y)-len(True_Prediction.index), len(y), facecolor='g', alpha=0.5, label='Testing Region')
            plt.axvspan(0,len(y) - len(True_Prediction.index), facecolor='lightblue', alpha=0.5, label='Training Region')
            plt.xlabel('Number of Days')
            plt.ylabel('Building Load ('+str(label)+')')
            plt.title('Predictions on '+dataset+" Dataset")
            plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 24)))) # convert hours to days
            plt.margins(x=0) #removes whitespace
            plt.legend()
            plt.show()
    return
def train_vs_test_acc(datasets, label):
    for dataset in datasets:
        x, y = obtain_data(dataset, label)
        forecasting_output = round(len(y) / 24 * 0.5)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24 * forecasting_output / x.shape[0],shuffle=False)  # Shuffle is false due to temporal constraints.
        max_features = None
        n_trees = 100
        max_depth = 5
        Forest, samples = ForestTrain(n_trees, x_train, y_train, max_depth, True, max_features)
        predictions, errors = ForestPredict(n_trees, True, Forest, x_test, y_test)
        True_Prediction = pd.Series(np.median(pd.DataFrame(predictions), axis=0))  # np.mean(pd.DataFrame(predictions))
        MSE = mean_squared_error(y_test, True_Prediction.tolist())
        reg = linear_model.LinearRegression()
        reg.fit(x_train, y_train)
        predict_linear = reg.predict(x_test)
        mse_lin = mean_squared_error(y_test, predict_linear)
        predict_linear_train = reg.predict(x_train)
        mse_lin_train = mean_squared_error(y_train, predict_linear_train)
        ## Plotting ##
        plt.plot(range(0, len(y)), y, label='Real Data Output')
        plt.plot(range(len(y) - len(True_Prediction.index), len(y)), True_Prediction.values,
                 label="Predicted Output on Testing data: MSE=" + str(round(MSE)))  # ")  # +str(round(MSE)))
        plt.plot(range(len(y) - len(True_Prediction.index), len(y)), predict_linear, label="Linear Output on testing data: MSE = "+str(round(mse_lin)))
        plt.axvspan(len(y) - len(True_Prediction.index), len(y), facecolor='g', alpha=0.5, label='Testing Region')
        plt.axvspan(0, len(y) - len(True_Prediction.index), facecolor='lightblue', alpha=0.5,
                    label='Training Region')

        predictions, errors = ForestPredict(n_trees, True, Forest, x_train, y_train)
        True_Prediction = pd.Series(np.median(pd.DataFrame(predictions), axis=0))  # np.mean(pd.DataFrame(predictions))
        MSE = mean_squared_error(y_train, True_Prediction.tolist())

        plt.plot(range(0, len(y_train)), True_Prediction.values,
                 label="Predicted output on Training data: MSE = " + str(round(MSE)))
        plt.plot(range(0, len(y_train)), predict_linear_train, label="linear output on Training data: MSE = "+str(round(mse_lin_train)))
        plt.xlabel('Number of Days')
        plt.ylabel('Building Load (' + str(label) + ')')
        plt.title('Predictions on ' + dataset + " Dataset")
        plt.gca().get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 24))))  # convert hours to days
        plt.margins(x=0)  # removes whitespace
        plt.legend()
        plt.show()
    return
def n_tress_vs_max_feature_split(datasets,label):
    for dataset in datasets:
        x, y = obtain_data(dataset, label)
        max_feature_list = [19,29,39,49]
        for max_features in max_feature_list:
            forecasting_output = round(len(y) / 24 * 0.25)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24 * forecasting_output / x.shape[0],shuffle=False)
            n_trees_list = [50, 100, 200, 500, 1000]
            max_depth = 10
            plt.plot(range(len(y) - forecasting_output * 24, len(y)), y_test, label='Real Data Output')
            for n_trees in n_trees_list:
                Forest, samples = ForestTrain(n_trees, x_train, y_train, max_depth, True, max_features)
                predictions, errors = ForestPredict(n_trees, True, Forest, x_test, y_test)
                True_Prediction = pd.Series(
                    np.median(pd.DataFrame(predictions), axis=0))  # np.mean(pd.DataFrame(predictions))
                MSE = mean_squared_error(y_test, True_Prediction.tolist())
                plt.plot(range(len(y) - len(True_Prediction.index), len(y)), True_Prediction.values,
                         label="" + str(n_trees) + " Trees: MSE= " + str(round(MSE)))  # )
            plt.axvspan(len(y) - len(True_Prediction.index), len(y), facecolor='g', alpha=0.5)
            plt.xlabel('Number of Days')
            plt.ylabel('Building Load (' + str(label) + ')')
            plt.title('Predictions on ' + dataset + " Dataset, Splitting on " + str(max_features) + " max features")
            plt.gca().get_xaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 24))))  # convert hours to days
            plt.margins(x=0)  # removes whitespace
            plt.legend()
            plt.show()
    return
def depth_variance(datasets, label):
    for dataset in datasets:
        x, y = obtain_data(dataset, label)
        forecasting_output = round(len(y) / 24 * 0.5)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24 * forecasting_output / x.shape[0],shuffle=False)  # Shuffle is false due to temporal constraints.
        max_features = None
        n_trees = 100
        max_depths = [1, 5, 10, 15]
        #plt.rcParams["figure.figsize"] = (20, 10)
        plt.plot(range(0, len(y)), y, label='Real Data Output')
        for max_depth in max_depths:
            Forest, samples = ForestTrain(n_trees, x_train, y_train, max_depth, True, max_features)
            predictions, errors = ForestPredict(n_trees, True, Forest, x_test, y_test)
            True_Prediction = pd.Series(np.median(pd.DataFrame(predictions), axis=0))  # np.mean(pd.DataFrame(predictions))
            MSE = mean_squared_error(y_test, True_Prediction.tolist())
            ## Plotting ##
            plt.plot(range(len(y) - len(True_Prediction.index), len(y)), True_Prediction.values,label="" + str(max_depth) + " Depth: MSE= " + str(round(MSE))+" testing")
            predictions, errors = ForestPredict(n_trees, True, Forest, x_train, y_train)
            True_Prediction = pd.Series(
                np.median(pd.DataFrame(predictions), axis=0))  # np.mean(pd.DataFrame(predictions))
            MSE = mean_squared_error(y_train, True_Prediction.tolist())

            plt.plot(range(0, len(y_train)), True_Prediction.values,
                     label="" + str(max_depth) + " Depth: MSE= " + str(round(MSE))+" training")
            # )
        reg = linear_model.LinearRegression()
        reg.fit(x_train, y_train)
        predict_linear = reg.predict(x_test)
        mse_lin = mean_squared_error(y_test, predict_linear)
        predict_linear_train = reg.predict(x_train)
        mse_lin_train = mean_squared_error(y_train, predict_linear_train)
        ## Plotting ##

        plt.plot(range(len(y) - len(y_test), len(y)), predict_linear, label="Linear Output on testing data: MSE = "+str(round(mse_lin)))
        plt.axvspan(len(y) - len(True_Prediction.index), len(y), facecolor='g', alpha=0.5, label='Testing Region')
        plt.axvspan(0, len(y) - len(True_Prediction.index), facecolor='lightblue', alpha=0.5,
                    label='Training Region')

        plt.plot(range(0, len(y_train)), predict_linear_train, label="linear output on Training data: MSE = "+str(round(mse_lin_train)))
        plt.xlabel('Number of Days')
        plt.ylabel('Building Load (' + str(label) + ')')
        plt.title('Predictions on ' + dataset + " Dataset")
        plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 24))))  # convert hours to days
        plt.margins(x=0)  # removes whitespace
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
    return

def Final_Best_Forest(datasets, label):
  for dataset in datasets:
        x, y = obtain_data(dataset, label)
        forecasting_output = round(len(y) / 24 * 0.5)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24 * forecasting_output / x.shape[0],shuffle=False)
        max_features = None
        n_trees = 50
        max_depths = 5
        Forest, samples = ForestTrain(n_trees, x_train, y_train, max_depths, False, max_features)
        predictions, errors = ForestPredict(n_trees, False, Forest, x_test, y_test) # False indicates our own tree code is used.
        True_Prediction = pd.Series(np.median(pd.DataFrame(predictions), axis=0))
        MSE = mean_squared_error(y_test, True_Prediction.tolist())
        #
        reg = linear_model.LinearRegression()
        reg.fit(x_train, y_train)
        predict_linear = reg.predict(x_test)
        mse_lin = mean_squared_error(y_test, predict_linear)
        #
        plt.plot(range(len(y) - forecasting_output * 24, len(y)), y_test, label='Real Data Output')
        plt.plot(range(len(y) - len(True_Prediction.index), len(y)), True_Prediction.values,
                         label="" + str(n_trees) + " Trees: MSE= " + str(round(MSE)))  # )
        plt.plot(range(len(y) - len(y_test), len(y)), predict_linear, label="Linear Output: MSE = "+str(round(mse_lin)))
        plt.axvspan(len(y) - len(True_Prediction.index), len(y), facecolor='g', alpha=0.5)
        plt.xlabel('Number of Days')
        plt.ylabel('Building Load (' + str(label) + ')')
        plt.title('Predictions on ' + dataset + " Dataset, Splitting on " + str(max_features) + " max features")
        plt.gca().get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 24))))  # convert hours to days
        plt.margins(x=0)  # removes whitespace
        plt.legend()
        plt.show()
  return

# if __name__ == '__main__':
#     !gdown 1VK4YXqsJX24wciOTyT6M_P-la6WBhjdJ # shared drive link to datasets
#     !gdown 13N9Iwed5ar7F4tv1EeTiKQ2bpc942FHB
#     !gdown 1GOGmMsuF7I3hZ_CBTOuwOqqJGGjHcM2X
#     !gdown 1VyjK4Ce3DC4RucQ5QaqsXoQZoC--gg0P
#     datasets = ["MarMay.csv", "JunAug.csv", "SepNov.csv", "DecFeb.csv"]
#     #### label options: ['electrical', 'thermal', 'cooling', 'heating']
#     label = 'electrical'



