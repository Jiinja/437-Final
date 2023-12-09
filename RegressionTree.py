import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time


class RegressionTree():
    """
    Decision Tree Regression Class 
    This class is both a tree and a node in the tree
    """
    def __init__(
        self,
        x_values : pd.DataFrame, # Training input
        y_values        = None,  # Training output, if None, assume output is last column in x_values
        current_depth   = 0,     # Depth of node with respesct to the root
        max_depth       = 5,     # maximum depth of tree 
        usable_features = None   # list of feature (column) indexes that are usable when training
    ):
        # setting up which features can be used/iterated on later 
        self.usable_features = usable_features if usable_features else list(range(x_values.shape[1])) 

        # saving x and y values for fitting
        if y_values == None:
            # if no y_values given, assume they are last column of x
            self.x_values = x_values.copy()
            self.y_values = np.array(self.x_values.iloc[:, -1])
            self.features = len(self.x_values.columns) - 1
        else:
            # y values given, treat x as all features
            self.x_values = x_values.copy()
            self.y_values = np.array(y_values)

            self.features = len(self.x_values.columns)
            self.x_values.insert(self.features, self.features, self.y_values)

        # setting up depth
        self.current_depth = current_depth
        self.max_depth     = max_depth
        
        # Initializing variables
        self.decision_feature  = None
        self.decision_value    = None
        self.left_value        = None
        self.right_value       = None

        # Initializing children
        self.left  = None
        self.right = None

    
    def fit(self) -> None:
        # if self.current_depth == 0:
        #     start_time = time.time()
        # error to infinity
        min_error = float('inf')

        # for each feature being used (column of x)
        for feature in self.usable_features:

            # Process only unique values
            for value in np.unique(self.x_values[feature]):

                # Get sum of squared residuals based on splitting data on that feature & value
                current_error, left_average, right_average = self.split_error(feature, value)

                # If new error is smaller than previous
                if current_error < min_error:
                    # update min error & node decision boundary
                    min_error = current_error
                    self.decision_feature = feature
                    self.decision_value = value
                    self.left_value = left_average
                    self.right_value = right_average

        # Create children and fit them as well
        if(self.current_depth < self.max_depth) and len(self.x_values) > 1:
            
            # Filtering left and right sides
            left_mask = self.x_values[self.decision_feature] <= self.decision_value
            right_mask = ~left_mask

            left_x = self.x_values[left_mask]
            right_x = self.x_values[right_mask]

            # Ensure child node will have at least 2 values to split
            if np.sum(left_mask) > 1:
                # Make child and fit
                self.left = RegressionTree(left_x, current_depth=self.current_depth+1, max_depth=self.max_depth, usable_features=self.usable_features)     
                self.left.fit()        
            
            # Ensure child node will have at least 2 values to split
            if np.sum(right_mask) > 1:
                # Make child and fit
                self.right = RegressionTree(right_x, current_depth=self.current_depth+1, max_depth=self.max_depth, usable_features=self.usable_features)            
                self.right.fit()

        # if self.current_depth == 0:
        #     print("Fitting took {} seconds".format(time.time() - start_time))

        # calculate and print metrics for the current node
        self.calculate_metrics(self.y_values, self.predictFrame(self.x_values), "Node")

        # plot actual vs. predicted values for the current node
        self.plot_actual_vs_predicted(range(len(self.y_values)), self.y_values, "Node")


    def split_error(self, feature, value) -> (int, int, int):
        # Get feature as a NumPy array
        feature_values = self.x_values[feature].to_numpy()

        # Use boolean indexing to separate y_values based on feature
        left_mask = feature_values <= value
        right_mask = ~left_mask

        left = self.y_values[left_mask]
        right = self.y_values[right_mask]

        # Get mean of each size, ensuring the lists aren't empty
        left_average = np.mean(left) if not len(left) == 0 else 0
        right_average = np.mean(right) if not len(right) == 0 else 0

        # Get sum of squared residuals using NumPy
        result = 0
        if not left_average == 0:
            result += np.sum((left - left_average) ** 2)
        if not right_average == 0:
            result += np.sum((right - right_average) ** 2)

        # Return total
        return result, left_average, right_average
        

    """
    This method takes a list that represents a single row of a dataframe
    returns a prediction for the row
    """
    def predictRow(self, input):
        if input[self.decision_feature] <= self.decision_value:
            # go left
            return self.left.predictRow(input) if self.left else self.left_value
        else:
            # go right
            return self.right.predictRow(input) if self.right else self.right_value

    """
    This method takes a dataframe and attempts to predict an output for each value
    returns a list of predictions in the order of the dataframe
    """
    def predictFrame(self,input):
        # start_time = time.time()
        input_columns = len(input.columns)
        if input_columns != self.features:
            raise Exception("Dimensions of input must match dimensions of training data {} != {}".format(input_columns, self.features))

        # Convert to numpy because SPEED
        input_array = input.to_numpy()

        # Predict values for each row
        result = [self.predictRow(row) for row in input_array]

        # Calculate and print metrics for the overall predictions
        self.calculate_metrics(self.y_values, result, "Overall")

        # Plot actual vs. predicted values for the overall predictions
        self.plot_actual_vs_predicted(range(len(self.y_values)), self.y_values, "Overall")

        # print("Predict frame took {} seconds".format(time.time() - start_time))
        
        return result

    
    def calculate_metrics(self, actual, predicted, label):
        r2 = r2_score(actual, predicted)
        print(f"{label} R^2 Score:", r2)

        error = mean_squared_error(actual, predicted)
        print(f"{label} Mean Squared Error:", error)

        rmse = np.sqrt(mse)
        print(f"{label} Root Mean Squared Error:", rmse)

        mae = np.mean(np.abs(np.array(actual) - np.array(predicted)))
        print(f"{label} Mean Absolute Error:", mae)

    def plot_actual_vs_predicted(self, x_values, actual, label):
        plt.scatter(x_values, actual, color='black', label='Actual')
        plt.scatter(x_values, self.predictFrame(self.x_values), color='blue', label='Predicted')
        plt.xlabel('Data Points')
        plt.ylabel('Values')
        plt.title(f'Actual vs. Predicted Values ({label})')
        plt.legend()
        plt.show()