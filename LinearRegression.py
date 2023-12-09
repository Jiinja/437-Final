import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def readData(file):
    # Read file & ignore headers
    rawdata = np.loadtxt(file, delimiter=",", dtype=str)
    rawdata = rawdata[1:]

    # x = 5 -> -1
    x = list(map(lambda x : x[5:-1], rawdata))
    # Ensure numeric datatype
    x = list(map(lambda innerList : list(map(float, innerList)), x))
    # y_electrical = 1
    y_electrical = list(map(lambda x : float(x[1]), rawdata))
    # y_thermal = 2
    y_thermal = list(map(lambda x : float(x[2]), rawdata))
    # y_cooling = 3
    y_cooling = list(map(lambda x : float(x[3]), rawdata))
    # y_heating = 4
    y_heating = list(map(lambda x : float(x[4]), rawdata))\
    
    # Create model for each output
    print("y_electrical:")
    fitModel(x, y_electrical)

    print("y_thermal:")
    fitModel(x, y_thermal)

    print("y_cooling:")
    fitModel(x, y_cooling)

    print("y_heating:")
    fitModel(x, y_heating)


def fitModel(x , y):
    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create model and fit
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)

    # evaluate predictions
    predict = reg.predict(x_test)
    error = mean_squared_error(y_test, predict)
    print("Mean Squared Error:", error)

    # calculate R^2 value
    r2 = r2_score(y_test, predict)
    print("R^2 Score:", r2)

    # plot the actual vs. predicted values
    plt.scatter(range(len(y_test)), y_test, color='black', label='Actual')
    plt.scatter(range(len(predict)), predict, color='blue', label='Predicted')
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

    # print root mean squared error
    rmse = np.sqrt(error)
    print("Root Mean Squared Error:", rmse)

    # print mean absolute error
    residuals = y_test - predict
    mae = np.mean(np.abs(residuals))
    print("Mean Absolute Error:", mae)



if __name__ == "__main__":
    print("DecFeb")
    readData("DecFeb.csv")
    
    print("JunAug")
    readData("JunAug.csv")

    print("MarMay")
    readData("MarMay.csv")

    print("SepNov")
    readData("SepNov.csv")

