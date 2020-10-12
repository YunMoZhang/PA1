import numpy as np

class RLS:
    def __init__(self, _sampy, _phi, _lam):
        self.sampy = _sampy
        self.phi = _phi
        self.lam = _lam
        self.theta = None

    def cal_theta(self): 
        theta = np.linalg.inv(self.phi @ self.phi.T + self.lam * np.identity(len(self.phi))) @ self.phi @ self.sampy
        self.theta = theta
        return theta

    def predict(self, test_phi):
        pred = test_phi.T @ self.theta
        return pred