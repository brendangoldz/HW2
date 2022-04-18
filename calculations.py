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

    def precision(self):
        return 0

    def fmeasure(self):
        return 0

    def accuracy(self, Y, Y_h):
        # return (100-np.mean(np.abs(Y_h-Y))*100)
        return (Y == Y_h).sum()/(Y.shape[0])
