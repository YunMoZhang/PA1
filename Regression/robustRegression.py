import numpy as np
from cvxopt import matrix, solvers

class RR:
    def __init__(self, _sampy, _phi):
        self.sampy = _sampy
        self.phi = _phi
        self.theta = None

    def cal_theta(self):       
        D = len(self.phi)
        n = len(self.sampy)
        c = matrix(np.concatenate((np.zeros(D), np.ones(n)), axis = 0))
        a1 = np.concatenate((-self.phi.T, -np.identity(n)), axis = 1)
        a2 = np.concatenate((self.phi.T, -np.identity(n)), axis = 1)
        A = matrix(np.concatenate((a1, a2), axis = 0))
        b = matrix(np.concatenate((-self.sampy, self.sampy), axis = 0))
        sol = solvers.lp(c, A, b)
        theta = np.array(sol['x'][0:D])
        theta = np.concatenate(theta, axis = 0)
        self.theta = theta
        return theta

    def predict(self, test_phi):
        pred = test_phi.T @ self.theta
        return pred