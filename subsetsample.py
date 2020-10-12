import numpy as np

def subsetsamp(sampx, sampy, percent):
    subset_size = int(len(sampx) * percent)
    randomNums = np.random.randint(len(sampx), size = subset_size)
    subsampx = []
    subsampy = []
    for i in randomNums:
        subsampx.append(sampx[i])
        subsampy.append(sampy[i])
    return subsampx, subsampy