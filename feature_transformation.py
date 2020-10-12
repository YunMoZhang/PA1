import numpy as np

def transform_part1(x, degree = 5):
    n = len(x)
    phi = np.zeros((degree + 1, n))   
    for i in range(n):
        temp = [1]
        for j in range(1, degree + 1):
            temp.append(x[i] ** j)
        phi[:, i] = temp    
    return phi

def transform_part2(x, degree = 1, inter = False):
    if(degree > 2):
        print("sorry, degree of part2's polynomial couldn't larger than 2 in this version")
        return
    n = len(x)
    feats_n = len(x[0])
    theta_n = feats_n * degree
    if inter is True:
        theta_n += (feats_n^2 - feats_n) / 2
    phi = np.zeros(theta_n, n)
    for i in range(n):
        temp = []
        for d in range(1, degree + 1):
            temp.extend([p ** d for p in x[i]])
        if inter is True:
            for j in range(feats_n):
                for k in range(j + 1, feats_n):
                    temp.append(x[i][j] * x[i][k])
        phi[:, i] = temp
    return phi
        
