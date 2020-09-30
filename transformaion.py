import numpy as np
from itertools import chain, combinations
from itertools import combinations_with_replacement 

class Transformation(object):
    def __init__(self, x, degree, interaction = False):
        self._x = x
        self._degree = degree
        self._interaction = interaction

    def transfrom_wo_inter(self):
        n = len(self._x)
        feats_n = ( len(self._x) if type(self.X[0]) == np.ndarray else 1)
        phi = np.zeros((feats_n * self._degree +1, n))   
        for i in range(0,n):
            temp = [1]
            for j in range(1, self._degree + 1):
                temp = np.append(temp, np.power(self.X[i], j))
            phi[:, i] = temp    
        return phi

    def transform_w_inter(self):
