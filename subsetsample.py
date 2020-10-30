import numpy as np

def subsetsamp(sampx, sampy, percent):
    subset_size = int(len(sampx) * percent / 100)
    randomNums = np.random.choice(np.array(range(len(sampx))), size = subset_size, replace = False)
    subsampx = sampx[randomNums]
    subsampy = sampy[randomNums]
    return subsampx, subsampy

def addoutlier(sampy):
    subset_size = int(len(sampy) * 0.1)
    randomNums = np.random.choice(np.array(range(len(sampy))), size = subset_size, replace = False)
    added_sampy = sampy
    added_sampy[randomNums] = sampy[randomNums] * np.random.randint(2,6)
    return added_sampy