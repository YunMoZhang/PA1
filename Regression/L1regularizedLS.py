import numpy as np
from cvxopt import matrix, solvers

class LASSO:
    def __init__(self, _sampy, _phi, _lam):
        self.sampy = _sampy
        self.phi = _phi
        self.lam = _lam
        self.theta = None

    def cal_theta(self):       
        n = len(self.phi)
        p1 = np.concatenate((self.phi @ self.phi.T, -self.phi @ self.phi.T ), axis = 1)
        p2 = np.concatenate((-self.phi @ self.phi.T, self.phi @ self.phi.T ), axis = 1)
        P = matrix(np.concatenate((p1, p2), axis = 0))
        q = matrix(self.lam * np.ones(2*n) - np.concatenate((self.phi @ self.sampy, -self.phi @ self.sampy), axis = 0))
        G = matrix(-np.identity(2*n))
        h = matrix(np.zeros(2*n))
        sol = solvers.qp(P, q, G, h)
        theta_plus = np.array(sol['x'][0:n])
        theta_minus = np.array(sol['x'][n:2*n])
        theta = np.concatenate(theta_plus - theta_minus, axis=0)
        self.theta = theta
        return theta

    def predict(self, test_phi):
        pred = test_phi.T @ self.theta
        return pred