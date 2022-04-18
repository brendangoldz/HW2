import numpy as np


class LinearRegression():

    def theta(self, X, Y):
        m = X.shape[0]

        X = np.append(X, np.ones((m, 1)), axis=1)
        # Make Y size same as X so matrices align
        # for operations
        Y = Y.reshape(m, 1)
        # Make Y a Matrix
        Y = np.mat(Y)

        # w = (1/(X.T * X)) * (X.T * Y)

        f1 = np.dot(X.T, X)  # f1 = X.T * X
        f2 = np.dot(X.T, Y)  # f2 = X.T * Y

        # w = (f1)^-1 * f2
        return np.linalg.pinv(f1)*f2

    def y_hat(self, X, theta):
        m = X.shape[0]  # Size of X samples
        X = np.append(X, np.ones((m, 1)), axis=1)
        return np.dot(X, theta).flatten().astype(int)
    
    def format(self, weights):
        print(weights)
        weights = np.array(weights).flatten()
        output = " y = " + str(weights[0]) + " + "
        weight_size = np.shape(weights)[0]
        parts_of_model = list()
        for i in range(1, weight_size):
            new_str = str(weights[i]) + "x" + str(i)
            if True == (i+1 < weight_size):
                new_str += " + "
            output += new_str
        return output

