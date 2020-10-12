import numpy as np

class BR:
    def __init__(self, _sampy, _phi, _alpha, _sigma):
        self.sampy = _sampy
        self.phi = _phi
        self.alpha = _alpha
        self.sigma = _sigma
        self.miu = None
        self.Sigma = None

    def cal_para(self): 
        n = len(self.phi)
        I = np.identity(n)
        Sigma = np.linalg.inv((1.0 / self.alpha) * I + (1.0 / (self.sigma ** 2)) * self.phi @ self.phi.T)
        self.Sigma = Sigma
        miu = 1.0 / (self.sigma ** 2) * Sigma @ self.phi @ self.sampy
        self.miu = miu
        return miu, Sigma

    def predict(self, test_phi):
        pred_miu = test_phi.T @ self.miu
        pred_sigma = test_phi.T @ self.Sigma @ test_phi
        return pred_miu, pred_sigma