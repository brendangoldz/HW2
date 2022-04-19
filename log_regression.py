import numpy as np
from data_handler import DataHandler


class LogisticalRegression():
    tr_log_loss = np.array(0)
    val_log_loss = np.array(0)

    TERMINATION_VALUE = 2^-32
    ITERATIONS = 1000
    LEARNING_RATE = 0.005

    def linear_mod(self, w, X, b):
        return np.dot(X, w) + b
    
    def sigmoid(self, wx_b):
        # Probability F(X) for where Y = 1
        return 1/(1 + np.exp(-(wx_b)))

    def weights(self, X, Y, Y_h):
        m = X.shape[0]
        return (1/m) * (X.T @ (sigmoid(X @ Y_) - Y))

    def compute_weights_bias(self, w, b, tX, tY, vX, vY):
        P = self.sigmoid(self.linear_mod(w, tX, b))
        weight = self.weights(tX, tY, P)
        bias = np.mean(np.subtract(P, tY))
        return weight, bias
    
    def cost(self, Y, P):
        logP = np.log(P, where=(P>0))
        p1 = np.multiply(Y, logP)
        p2 = 1 - Y
        p3 = 1 - P
        p4 = np.log(p3, where=((p3)>0))
        cost = -(p1+np.multiply(p2, p4))
        return cost
    
    def prediction(self, w, b, X):
        m = X.size[0]
        P = self.sigmoid(self.linear_mod(w, tX, b))
        Y_preds = np.zeros(1, m)
        for i in range(m):
            if P[0, i] > 0.5:
                Y_preds[0, i] = 1
            else:
                Y_preds[0, i] = 0
        return Y_preds

    def calculate(self, w, b, tX, tY, vX, vY):
        t_costs = []
        v_costs = []
        w = w
        b = b
        for i in range(self.ITERATIONS):            
#             Calculate gradient descent for training
#             - this is linear model
#             Calculate gradient descent for validation
#             - this is linear model

#             Calculate sigmold for training
#             - pass above into sigmoid
#             Calculate sigmold for validation
#             - pass above info sigmoid 

#             Calculate parameters (deriv weights/bias) with training data
#             - pass in x train, y train, training sigmoid output

#             Update weights and bias
#             - update the weights and bias

#             Compute mean log loss for training
#             - compute with y train y training sigmoid 
#             Compute mean log loss for validation
#             - compute with y valid y valid sigmoid 

#             Return losses and weights and bias 
            v_wxb = self.linear_mod(w, vX, b)
            t_wxb = self.linear_mod(w, tX, b)
            weight, beta = self.compute_weights_bias(w, b, tX, tY, vX, vY)
            
            P = self.sigmoid(t_wxb)
            vP = self.sigmoid(v_wxb)
            
            # Loss Calc
            t_cost = self.cost(tY, P)
            v_cost = self.cost(vY, vP)
            t_costs.append(t_cost)
            v_costs.append(v_cost)
            
            # Gradient Recalc
            w = w - self.LEARNING_RATE * weight
            b = b - self.LEARNING_RATE * beta
            # Record the costs
            if i % 100 == 0:
                print("TMean Cost after iteration %i: %f" % (i, np.mean(t_costs)))
                print("VMean Cost after iteration %i: %f" % (i, np.mean(v_cost)))
            

        losses = {
            "TR": t_costs,
            "VAL": v_costs
        }
        return w, b, losses