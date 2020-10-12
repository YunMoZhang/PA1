def MSE(test_y, true_y):
    diff = test_y - true_y
    MSE = sum([i**2 for i in diff])/len(diff)
    return MSE

def MAE(test_y, true_y):
    diff = test_y - true_y
    MAE = sum([abs(i) for i in diff])/len(diff)
    return MAE