import numpy as np

class LS:
    def __init__(self, _sampy, _phi):
        self.sampy = _sampy
        self.phi = _phi
        self.theta = None

    def cal_theta(self):
        theta = np.linalg.inv(self.phi @ self.phi.T) @ self.phi @ self.sampy
        self.theta = theta
        return theta
    
    def predict(self, test_phi):
        pred = test_phi.T @ self.theta
        return pred
