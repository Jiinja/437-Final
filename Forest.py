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
        samples.append(sample)  # idk if we need to save this yet.
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


def ForestReg(x, y, sklearn_flag,days, max_depth, max_features, n_trees):
    # Split dataset
    DAYs = 30
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24*DAYs / x.shape[0],shuffle=False)  # This is the original full sized data set
    # for test_size: I want the test group to be 24 intervals to act like a Day-Ahead predictor (one row==1hour) . x*x.size[] = 24 => 24/z.size[]
    ## HOWEVER: we need 24 CONSECUTIVE rows for testing. IT CANT be random for the output to make sense. Its bound temporally. => shuffle = False.
    # Since we bootstrap many trees anyway for variety, I dont think its necessary to have a random train/test split (right??)
    ## BUTALSO: now we are just testing on the last day? Shouldnt we pick a random day within the set?
    ## ALSO ALSO: I feel like having the extra binary columns indicating which hour it is may reduce the temporal noise bit..

    max_depth = 10  # determine best depth in report
    n_trees = 100 # determine size of forest in report
    max_features = 35  # max features <= num features == len(x)(48)
    ##### Forest Train ####
    Forest, samples = ForestTrain(n_trees, x_train, y_train, max_depth, sklearn_flag, max_features)
    ######### Forest Prediction #############:
    # Tally each tree's output and take an average of the answer (like average jelly bean guess idea)
    predictions, errors = ForestPredict(n_trees, sklearn_flag, Forest, x_test, y_test)
    ####
    ## reportable improvement:
    #True_Prediction = (pd.DataFrame(predictions)).sum(axis=0) / n_trees  # average output for each hour in 1 day interval.
    True_Prediction = np.mean(pd.DataFrame(predictions))

    True_Prediction2 = pd.Series(np.median(pd.DataFrame(predictions), axis=0))
    True_error = mean_squared_error(y_test, True_Prediction.tolist())

    ### ###################data analysis ramblings/ideas:
    # hopefully this is better than a linear regression.
    ##TODO: add code to regression tree to allow for random variable subsetting at each step.
    # this idea is to add variety to the trees created from the data set.
    # instead of considering all the x variables for the root node, only select a subset of them(?).
    ##TODO: EVALUATION METRIC FROM VIDEO: OUT OF BAG SAMPLE ACCURACY. WE CAN USE THIS TO ALSO SEE THE VAIRABLE SUBSETTING ACCURACY
    ##TODO: This is why I think we need to cluster the binary variable types. Day/hour/ weekday/ day of week. Since the accuracy between a monday and tuesday will be different than a weekend.(saves computation time)
    ##SAMPLES INVERSE. x_train[not_sampled], y_train[not_sampled] _> so we can see accuracy of the trees that have not been trained on the unused bootstrapped values.

    # quick linear regression test #
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    predict_linear = reg.predict(x_test)

    calculate_metrics(y_test, True_Prediction.tolist(), "Forest")
    calculate_metrics(y_test, predict_linear, "LINEAR")
    test_params = [max_depth, max_features, n_trees]

    return True_Prediction, y_test, predict_linear, test_params


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





"""
Code Demo:
When determining the best characteristics for this forest, we are keeping these values constant between tests for consistency: 
Max_depth = 6, Max_features = None, n_trees = 100. 
"""
def Find_Best_Depth(forecasting_output, x, y):
    """
    Vary the Max_depth and compare MSE, choose minimal error choice.
    Compare MSE vs. median aggregation
    Compare averaged values.
    See if there is convergence.
    #note: capped max depth at 10 because >10 we dont see any grandularity improvement + takes a long time to finish.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24*forecasting_output / x.shape[0], shuffle=False) #Shuffle is false due to temporal constraints.
    max_features = None
    n_trees = 100
    error_per_depth = []
    for max_depth in range(1, 10): # get range of max depth values and create and test forest and report accuracy wrt mse.
        Forest, samples = ForestTrain(n_trees, x_train, y_train, 10, True, None)
        predictions, errors = ForestPredict(n_trees, True, Forest, x_test, y_test)
        True_Prediction = pd.Series(np.median(pd.DataFrame(predictions), axis=0))#np.mean(pd.DataFrame(predictions))
        error_per_depth.append(mean_squared_error(y_test, True_Prediction.tolist()))
        plt.plot(range(len(y)-len(True_Prediction.index), len(y)), True_Prediction.values, label="Predicted")
        plt.plot(range(0, len(y)), y)
        plt.legend()
        plt.show()
        print("TEST")
    plt.plot(range(1,10), error_per_depth, label='MSE')
    plt.scatter(range(1,10), error_per_depth)
    plt.title("MSE WRT Max_Depth")
    plt.legend()
    plt.show()

    best_depth = np.argmin(error_per_depth)
    return best_depth








def Find_Best_Max_Feature_Split(forecasting_output):
    """
    can use MSE to compare max Features, but also lets try and implement an out-of-bag accuracy here.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24*forecasting_output / x.shape[0], shuffle=False) #Shuffle is false due to temporal constraints.
    n_trees = 100
    max_depth = 6
    error_per_depth = []
    for max_features in range(1,48):  # get range of max depth values and create and test forest and report accuracy wrt mse.
        Forest, samples = ForestTrain(n_trees, x_train, y_train, max_depth, True, max_features)
        predictions, errors = ForestPredict(n_trees, True, Forest, x_test, y_test)
        True_Prediction = np.mean(pd.DataFrame(predictions))
        error_per_depth.append(mean_squared_error(y_test, True_Prediction.tolist()))
    plt.plot(range(1, 48), error_per_depth, label='MSE')
    plt.scatter(range(1, 48), error_per_depth)
    plt.title("MSE WRT Max_features")
    plt.legend()
    plt.show()
    best_max_features = np.argmin(error_per_depth)
    return best_max_features
def Find_Best_Number_Trees_In_Forest(forecasting_output):
    """
    This should be the easiest to find
    just compare MSE to number trees in forest.
    See where diminishing returns exist.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24*forecasting_output / x.shape[0], shuffle=False) #Shuffle is false due to temporal constraints.
    max_features = None
    max_depth = 6
    error_per_depth = []
    for n_trees in range(1,200,25):  # get range of max depth values and create and test forest and report accuracy wrt mse.
        Forest, samples = ForestTrain(n_trees, x_train, y_train, max_depth, True, max_features)
        predictions, errors = ForestPredict(n_trees, True, Forest, x_test, y_test)
        True_Prediction = np.mean(pd.DataFrame(predictions))
        error_per_depth.append(mean_squared_error(y_test, True_Prediction.tolist()))
    plt.plot( range(1,200,25), error_per_depth, label='MSE')
    plt.scatter( range(1,200,25), error_per_depth)
    plt.title("MSE WRT Max_features")
    plt.legend()
    plt.show()
    best_max_features = np.argmin(error_per_depth)

    return best_number_trees
def Compare_Leaf_Predictions(forecasting_output):
    """
    We have dataframe of predictions for each tree. we need to figures out how to aggregate/average the data in the best way
    Maybe we can try and linearly forecast the intervals magically as well somehow later.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24*forecasting_output / x.shape[0], shuffle=False) #Shuffle is false due to temporal constraints.
    max_features = None
    n_trees = 100
    max_depth = 6

    Forest, samples = ForestTrain(n_trees, x_train, y_train, max_depth, sklearn_flag, max_features)
    return winning_predictor

def Get_Best_Forest(forecasting_output, best_number_trees,winning_predictor, best_max_features, best_max_depth):
    """
    creates a forest using the best values found from previous functions.
    This forest will be used to compare against the linear regression.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24*forecasting_output / x.shape[0], shuffle=False) #Shuffle is false due to temporal constraints.

    Forest, samples = ForestTrain(best_number_trees, x_train, y_train, best_max_depth, sklearn_flag, best_max_features)
    return
def Final_Evaluation():
    """
    compares best-tuned forest to basic linear regression
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24*forecasting_output / x.shape[0], shuffle=False) #Shuffle is false due to temporal constraints.

    return

def obtain_data(file):
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
    return x, y_electrical

if __name__ == '__main__':
    # best_depth = Find_Best_Depth(forecasting_output, x, y)
    # best_max_features = Find_Best_Max_Feature_Split(forecasting_output)
    # best_number_of_trees = Find_Best_Number_Trees_In_Forest(forecasting_output)
    datasets = ["MarMay.csv", "JunAug.csv", "SepNov.csv", "DecFeb.csv"]
    for dataset in datasets:
        x,y = obtain_data(dataset)
        #forecasting_outputs = [45,30,25,20,15] #15, 20, 25 # Number of days forecast (testing size) # fun demo output
        forecasting_outputs = [round(len(y)/24*0.9),round(len(y)/24*0.75),round(len(y)/24*0.5),round(len(y)/24*0.25),round(len(y)/24*0.10)]
        for forecasting_output in forecasting_outputs:
            print('hi')
            # ## I am not doing cross validation here.
            # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24*forecasting_output / x.shape[0], shuffle=False) #Shuffle is false due to temporal constraints.
            # max_features = None
            # n_trees = 100
            # max_depth = 10
            # Forest, samples = ForestTrain(n_trees, x_train, y_train, 10, True, None)
            # predictions, errors = ForestPredict(n_trees, True, Forest, x_test, y_test)
            # True_Prediction = pd.Series(np.median(pd.DataFrame(predictions), axis=0))#np.mean(pd.DataFrame(predictions))
            # #NOTE on MSE: When changing the forecasting ouputs, mse loses ites meaning because the
            # # length of y_test changes. That means there are less opportunities/chances for there
            # # to exist an error, so of  course a smaller prediction interval will give a smaller erros
            # # so we cant use mse comparison when changing the number of output days.
            # MSE = mean_squared_error(y_test, True_Prediction.tolist())
            # ## Plotting ##
            # plt.plot(range(0, len(y)), y, label='Real Data Output')
            # plt.plot(range(len(y)-len(True_Prediction.index), len(y)), True_Prediction.values, label="Predicted Output")#+str(round(MSE)))
            # plt.axvspan(len(y)-len(True_Prediction.index), len(y), facecolor='g', alpha=0.5, label='Testing Region')
            # plt.axvspan(0,len(y) - len(True_Prediction.index), facecolor='lightblue', alpha=0.5, label='Training Region')
            # plt.ylim([0,400])
            # plt.xlabel('Number of Days')
            # plt.ylabel('Electrical Load (kW)') # or Thermal Load (kBTU)
            # plt.title('Predictions on '+dataset+" Dataset")
            # plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 24)))) # convert hours to days
            # plt.margins(x=0) #removes whitespace
            # plt.legend()
            # plt.show()
            ## I am not doing cross validation here.
        forecasting_output = round(len(y) / 24 * 0.5)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24 * forecasting_output / x.shape[0],
                                                            shuffle=False)  # Shuffle is false due to temporal constraints.
        max_features = None
        n_trees = 100
        max_depth = 10
        Forest, samples = ForestTrain(n_trees, x_train, y_train, 10, True, None)
        predictions, errors = ForestPredict(n_trees, True, Forest, x_test, y_test)
        True_Prediction = pd.Series(np.median(pd.DataFrame(predictions), axis=0))  # np.mean(pd.DataFrame(predictions))
        MSE = mean_squared_error(y_test, True_Prediction.tolist())

        reg = linear_model.LinearRegression()
        reg.fit(x_train, y_train)
        predict_linear = reg.predict(x_test)
        predict_linear_train = reg.predict(x_train)
        ## Plotting ##
        plt.plot(range(0, len(y)), y, label='Real Data Output')
        plt.plot(range(len(y) - len(True_Prediction.index), len(y)), True_Prediction.values,label="Predicted Output"+str(round(MSE)))#")  # +str(round(MSE)))
        plt.plot(range(len(y) - len(True_Prediction.index), len(y)), predict_linear, label="Linear Output")
        plt.axvspan(len(y) - len(True_Prediction.index), len(y), facecolor='g', alpha=0.5, label='Testing Region')
        plt.axvspan(0, len(y) - len(True_Prediction.index), facecolor='lightblue', alpha=0.5,
                    label='Training Region')

        predictions, errors = ForestPredict(n_trees, True, Forest, x_train, y_train)
        True_Prediction = pd.Series(np.median(pd.DataFrame(predictions), axis=0))  # np.mean(pd.DataFrame(predictions))
        MSE = mean_squared_error(y_train, True_Prediction.tolist())

        plt.plot(range(0, len(y_train)), True_Prediction.values,label="Predicted output on Training data" + str(round(MSE)))
        plt.plot(range(0, len(y_train)), predict_linear_train,label="linear output on Training data")

        plt.ylim([0, 400])
        plt.xlabel('Number of Days')
        plt.ylabel('Electrical Load (kW)')  # or Thermal Load (kBTU)
        plt.title('Predictions on ' + dataset + " Dataset")
        plt.gca().get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 24))))  # convert hours to days
        plt.margins(x=0)  # removes whitespace
        plt.legend()
        plt.show()
    ### Varying Number of trees to MSE ###
        # forecasting_output = round(len(y)/24*0.25)
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24 * forecasting_output / x.shape[0],shuffle=False)
        # max_features = None
        # n_trees_list = [25, 50, 100, 200, 500, 1000]
        # max_depth = 10
        # plt.plot(range(len(y) - forecasting_output*24, len(y)), y_test, label='Real Data Output')
        # for n_trees in n_trees_list:
        #     Forest, samples = ForestTrain(n_trees, x_train, y_train, 10, True, None)
        #     predictions, errors = ForestPredict(n_trees, True, Forest, x_test, y_test)
        #     True_Prediction = pd.Series(np.median(pd.DataFrame(predictions), axis=0))  # np.mean(pd.DataFrame(predictions))
        #     MSE = mean_squared_error(y_test, True_Prediction.tolist())
        #     ## Plotting ##
        #     plt.plot(range(len(y)-len(True_Prediction.index), len(y)), True_Prediction.values, label=""+str(n_trees)+" Trees: MSE= "+str(round(MSE)))#)
        # plt.axvspan(len(y)-len(True_Prediction.index), len(y), facecolor='g', alpha=0.5, label='Testing Region')
        # #plt.axvspan(0,len(y) - len(True_Prediction.index), facecolor='lightblue', alpha=0.5, label='Training Region')
        # #plt.ylim([0,400])
        # plt.xlabel('Number of Days')
        # plt.ylabel('Electrical Load (kW)') # or Thermal Load (kBTU)
        # plt.title('Predictions on '+dataset+" Dataset")
        # plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 24)))) # convert hours to days
        # plt.margins(x=0) #removes whitespace
        # plt.legend()
        # plt.show()
    # # Varying Depth ##
    #     forecasting_output = round(len(y)/24*0.25)
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=24 * forecasting_output / x.shape[0],shuffle=False)
    #     max_features = None
    #     n_trees = 100
    #     max_depths = [1, 3, 5, 7 , 9, 10, 15]
    #     plt.plot(range(len(y) - forecasting_output*24, len(y)), y_test, label='Real Data Output')
    #     for max_depth in max_depths:
    #         Forest, samples = ForestTrain(n_trees, x_train, y_train, max_depth, True, None)
    #         predictions, errors = ForestPredict(n_trees, True, Forest, x_test, y_test)
    #         True_Prediction = pd.Series(np.median(pd.DataFrame(predictions), axis=0))  # np.mean(pd.DataFrame(predictions))
    #         MSE = mean_squared_error(y_test, True_Prediction.tolist())
    #         ## Plotting ##
    #         plt.plot(range(len(y)-len(True_Prediction.index), len(y)), True_Prediction.values, label=""+str(max_depth)+" Depth: MSE= "+str(round(MSE)))#)
    #     plt.axvspan(len(y)-len(True_Prediction.index), len(y), facecolor='g', alpha=0.5, label='Testing Region')
    #     #plt.axvspan(0,len(y) - len(True_Prediction.index), facecolor='lightblue', alpha=0.5, label='Training Region')
    #     #plt.ylim([0,400])
    #     plt.xlabel('Number of Days')
    #     plt.ylabel('Electrical Load (kW)') # or Thermal Load (kBTU)
    #     plt.title('Predictions on '+dataset+" Dataset")
    #     plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x / 24)))) # convert hours to days
    #     plt.margins(x=0) #removes whitespace
    #     plt.legend()
    #     plt.show()
    #         # Varying Depth ##

    print("TEST")

#####################################################################################################################

    # prediction, actual, linear_predict, test_params = ForestReg(x, y_electrical, True)
    # plt.plot(prediction.index, prediction.values, label='Prediction - forest reg')
    # plt.plot(prediction.index, linear_predict, label='Prediction - linear reg')
    # plt.plot(prediction.index, actual, label='Actual')
    # plt.title("Num Trees: " + str(test_params[2]) + " Max Depth: " + str(test_params[0]) + " Max_features: " + str(
    #     test_params[1]))
    # plt.legend()
    # plt.show()


    ####
    # have functions that obtain optimal tuning values ##
    # use those values for a final 'tuned' forest ##
    # compare that against linear regression #