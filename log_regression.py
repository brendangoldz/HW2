import numpy as np
from data_handler import DataHandler


class LogisticalRegression():
    tr_log_loss = np.array(0)
    val_log_loss = np.array(0)

    TERMINATION_VALUE = 2^-32
    ITERATIONS = 1000
    LEARNING_RATE = 0.005

    def linear_mod(self, w, X, b):
        # print(w.shape)
        # print(X.T.shape)
        return np.dot(X, w.T) + b
    
    def sigmoid(self, wx_b):
        # Probability F(X) for where Y = 1
        return 1/(1 + np.exp(-(wx_b)))

    def weights(self, X, Y, Y_h):
        m = X.shape[0]
        return (1/m) * (X.T @ (Y_h - Y))

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
        return np.squeeze(cost)
    
    def prediction(self, w, b, X, thresh=0.5):
        m = X.size[0]
        P = self.sigmoid(self.linear_mod(w, tX, b))
        Y_preds = np.zeros(1, m)
        for i in range(m):
            if P[i, 0] > thresh:
                Y_preds[i, 0] = 1
            else:
                Y_preds[i, 0] = 0
        return Y_preds

    def calculate(self, w, b, tX, tY, vX, vY):
        t_costs = list()
        v_costs = list()

        w = w.reshape((1,tX.shape[1]))
        b = b
        for i in range(self.ITERATIONS):
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