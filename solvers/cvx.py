import numpy as np
import cvxpy as cvx

class CVX():
    def __init__(self, X_train, y_train, t=0.01):
        self.X = X_train
        self.y = y_train
        self.X_major = self.X[np.where(self.y == 1)[0]]
        self.y_major = self.y[np.where(self.y == 1)[0]]
        self.X_minor = self.X[np.where(self.y == -1)[0]]
        self.y_minor = self.y[np.where(self.y == -1)[0]]

        self.t = t
        self.m = -(1/len(self.y_minor))*(self.X_minor.T @ np.ones(len(self.y_minor))) \
                 - (1/len(self.y_major))*(self.X_major.T @ np.ones(len(self.y_major)))
        self.w = cvx.Variable(self.X.shape[1])
        self.u = cvx.Variable(self.X.shape[0])
        self.z = cvx.Variable()

    def run(self):
        print('Running CVX...')
        loss = cvx.sum(cvx.logistic(cvx.multiply(-self.y, self.u)))
        obj = cvx.Minimize(loss)
        consts = [(self.X @ self.w) - self.u == 0, (self.m.T @ self.w) - self.z == 0, self.z >= 0]
        prob = cvx.Problem(obj, consts)
        prob.solve()
        self.w = self.w.value
        print('Done!\n')