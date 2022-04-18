import numpy as np
from data_handler import DataHandler


class LogisticalRegression():
    tr_log_loss = np.array(0)
    val_log_loss = np.array(0)

    TERMINATION_VALUE = 2^-32
    ITERATIONS = 2000
    LEARNING_RATE = 0.0001

    def __init__(self, data_handler, training, validation):
        

    def probability(self, wx_b):
        # Probability F(X) for where Y = 1
        return 1/(1 + np.exp(-(wx_b)))

    def weights(self, X, Y, Y_h):
        m = X.shape[1]
        return (1/m*(np.dot(X.T, (np.subtract(Y_h, Y)))))

    def cost(self, Y, P):
        p1 = Y*np.log(P)
        p2 = 1 - Y
        p3 = np.log(1-P)
        cost = -(np.mean(p1+np.multiply(p2, p3)))
        return cost

    def compute_weights_bias(self, w, X, Y, b):
        Y = Y.reshape(self.training_data_x.shape[0], 1)
        wx_b = np.dot(w.T, X) + b
        P = self.probability(wx_b)
        cost = self.cost(Y, P)
        weight = self.weights(X, Y, P)
        beta = np.mean(np.subtract(P, Y))
        return cost, weight, beta

    def gradient_descent(self, w, b, X):
        P = self.probability(np.dot(w.T, X) + b)
        m = P.size[1]
        Y_preds = np.zeros(1, m)
        for i in range(m):
            if P[0, i] > 0.5:
                Y_preds[0, i] = 1
            else:
                Y_preds[0, i] = 0
        return Y_preds

    def calculate(self, w, b, X, Y):
        costs = []
        for i in range(self.ITERATIONS):
            cost, weight, beta = self.compute_weights_bias(w, X, Y, b)
            print(cost, weight, beta)
            print(w, b)
            w = w - self.LEARNING_RATE * weight
            b = b - self.LEARNING_RATE * beta
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
                print("Mean Cost after iteration %i: %f" % (i, costs.mean()))
            costs.append(cost)

        return w, b, weight, beta

    def regression(self):
        weights = np.zeros(shape=(self.training_data_x.shape[0], 1))
        bias = 0
        w, b, deriv_weight, deriv_beta = self.calculate(
            weights, bias, self.training_data_x, self.training_data_y)
        train_y_preds = self.gradient_descent(w, b, self.training_data_x)

        return train_y_preds
