import numpy as np


class Calculations():
    def RMSE(self, y_hat, y):
        return np.sqrt(self.mse(y, y_hat))

    def mse(actual, predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)
        differences = np.subtract(actual, predicted)
        squared_differences = np.square(differences)
        return squared_differences.mean()

    def MAPE(y_hat, y):
        abs = np.abs((np.subtract(y, y_hat)/y))
        return np.mean(abs)

    def manhattan_distance(point1, point2):
        distanceOut = 0
        for x1, x2 in zip(point1, point2):
            difference = x2 - x1
            absolute_difference = abs(difference)
            distanceOut += absolute_difference
        return distanceOut

    # true_values = Y
    # predictions = Y_

    # N = true_values.shape[1]
    # accuracy = (true_values == predictions).sum() / N
    # TP = ((predictions == 1) & (true_values == 1)).sum()
    # FP = ((predictions == 1) & (true_values == 0)).sum()
    # precision = TP / (TP+FP)
    
    def true_positives(self, Y, Y_):
        return ((Y_ == 1) & (Y == 1)).sum()
    
    def false_positives(self, Y, Y_):
        return ((Y_ == 1) & (Y == 0)).sum()
    
    def precision(self, Y, Y_):
        TP = true_positives(Y, Y_)
        return (TP)/(TP+false_positives(Y, Y_))

    def fmeasure(self):
        return 0

    def accuracy(self, Y, Y_h):
        return 100-((Y == Y_h).sum()/(Y.shape[0]))