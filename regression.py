import numpy as np
import matplotlib.pyplot as plt

class Transformation(object):
    def __init__(self, x):
        self._x = x

class Regressions(object):
    def __init__():
        

class LS(object):
    """
    Least square method:
    Phi is input feature matrix w/ dim of D * N (np.ndarray);
    Y is the output vector w/ dim of N (np.ndarray);
    """
    def __init__(self, phi, y):
        self._y = y
        self._phi = phi
        self._theta = None
        
    def cal_theta(self):
        """
        Parameter estimate:
        theta is estimated parameters w/ dim D (np.ndarray);
        """
        theta = np.linalg.inv(self._phi @ self._phi.T) @ self._phi @ self._y
        self._theta = theta
        return theta
    
    def predict(self, test_phi):
        """
        Prediction f_* for input new_Phi:
        new_X is input data vector w/ dim of N' (np.ndarray);
        f is learned fcn output vector w/ dim of N' (np.ndarray);
        """
        pred = test_phi.T @ self._theta
        return pred