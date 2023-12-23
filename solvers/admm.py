import logging
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np


class ADMM:
    def __init__(self, X_train, y_train, t=0.01):
        self.X = X_train
        self.y = y_train
        self.X_major = self.X[np.where(self.y == 1)[0]]
        self.y_major = self.y[np.where(self.y == 1)[0]]
        self.X_minor = self.X[np.where(self.y == -1)[0]]
        self.y_minor = self.y[np.where(self.y == -1)[0]]

        self.t = t
        self.m = -(1 / len(self.y_minor)) * (self.X_minor.T @ np.ones(len(self.y_minor))) - (1 / len(self.y_major)) * (
            self.X_major.T @ np.ones(len(self.y_major))
        )
        self.w = np.zeros(self.X.shape[1])
        self.u = self.X @ self.w
        self.z = self.m @ self.w
        self.a = np.zeros(len(self.u))
        self.b = 0

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger("admm_logger")

    def loss(self):
        losses = np.log(1 + np.exp(np.multiply(-self.y, self.u)))
        if self.z >= 0:
            return np.sum(losses)
        else:
            return math.inf

    def w_update(self):
        q = ((-self.X.T) @ self.a) + self.t * (self.X.T @ self.u) + (self.t * self.z * self.m)
        term = self.t * (self.X.T @ self.X) + self.t * (self.m @ self.m.T)
        self.w = np.linalg.solve(term, q)

    def z_update(self):
        temp = (self.m @ self.w) + (self.b / self.t)

        if temp >= 0:
            self.z = (1 / self.t) * temp
        else:
            self.z = 0

    def minimize_u_lagrangian(self, a, y, term):
        start = 1
        num_iter = 0
        sol = 0
        dec = 0
        tol = 1e-6

        while start == 1 or (dec / 2) >= tol:
            num_iter += 1
            if num_iter >= 1:
                start = 0
            grad = -a + (self.t * sol) - (self.t * term) + (-y / (np.exp(y * sol) + 1))
            hes = self.t + ((y**2) * np.exp(y * sol)) / ((np.exp(y * sol) + 1) ** 2)
            step = -(1 / hes) * grad
            sol += step
            dec = (1 / hes) * (grad**2)

        return sol

    def run(self, diff=1e-2):
        iter_count = 0
        loss_prev = 0
        loss_cur = self.loss()
        self.logger.info(f"Starting loss: {loss_cur}")
        start = 1
        loss_hist = [loss_cur]

        while start == 1 or abs(loss_prev - loss_cur) >= diff:
            if iter_count >= 1:
                start = 0
            iter_count += 1

            self.w_update()

            term = self.X @ self.w
            for i in range(len(self.u)):
                self.u[i] = self.minimize_u_lagrangian(self.a[i], self.y[i], term[i])

            self.z_update()

            self.a += self.t * ((self.X @ self.w) - self.u)
            self.b += self.t * ((self.m @ self.w) - self.z)

            loss_prev = loss_cur
            loss_cur = self.loss()
            loss_hist.append(loss_cur)
            self.logger.info(f"Loss after iteration {iter_count}: {loss_cur}")

        if not os.path.exists("results"):
            os.makedirs("results")

        # plot results
        self.logger.info(f"ADMM converged with loss delta of {abs(loss_prev - loss_cur)}")
        plt.figure()
        plt.yscale("log")
        plt.plot(range(iter_count + 1), loss_hist, "-o")
        plt.title("ADMM Train Loss")
        plt.savefig(f"results/admm {time.asctime()}.png")
