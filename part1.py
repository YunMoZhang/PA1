from Regression.LeastSquare import *
from Regression.regularizedLS import *
from Regression.L1regularizedLS import *
from Regression.robustRegression import *
from Regression.BayesianRegression import *
from evaluation import *
from feature_transformation import *
from Plot import *
from subsetsample import *

import numpy as np 


# load data
sampx = np.loadtxt('/home/yunmzhang2/cs5487/pa1-zym/PA-1-data-text/polydata_data_sampx.txt')
sampy = np.loadtxt('/home/yunmzhang2/cs5487/pa1-zym/PA-1-data-text/polydata_data_sampy.txt')
polyx = np.loadtxt('/home/yunmzhang2/cs5487/pa1-zym/PA-1-data-text/polydata_data_polyx.txt')
polyy = np.loadtxt('/home/yunmzhang2/cs5487/pa1-zym/PA-1-data-text/polydata_data_polyy.txt')
thtrue = np.loadtxt('/home/yunmzhang2/cs5487/pa1-zym/PA-1-data-text/polydata_data_thtrue.txt')

################# (b) #################
phi = transform_part1(sampx) 
test_phi = transform_part1(polyx)

############# LS ############
'''
ls = LS(sampy, phi)
theta_ls = ls.cal_theta()


pred_ls = ls.predict(test_phi)
mse_ls = MSE(pred_ls, polyy)

print("MSE:" + str(mse_ls))

print("theta_esti:" + str(theta_ls))
print("theta_true:" + str(thtrue))

plot(sampx, sampy, polyx, polyy, pred_ls, "LS")
'''
########### RLS ############
'''
lam_rls = 0.1
rls = RLS(sampy, phi, lam_rls)
theta_rls = rls.cal_theta()

pred_rls = rls.predict(test_phi)
mse_rls = MSE(pred_rls, polyy)

print("MSE:" + str(mse_rls))

print("theta_esti:" + str(theta_rls))
print("theta_true:" + str(thtrue))

plot(sampx, sampy, polyx, polyy, pred_rls, "RLS")
'''
########### LASSO ############
'''
lam_lasso = 0.001
lasso = LASSO(sampy, phi, lam_lasso)
theta_lasso = lasso.cal_theta()

pred_lasso = lasso.predict(test_phi)
mse_lasso = MSE(pred_lasso, polyy)

print("MSE:" + str(mse_lasso))

print("theta_esti:" + str(theta_lasso))
print("theta_true:" + str(thtrue))

plot(sampx, sampy, polyx, polyy, pred_lasso, "LASSO")
'''
############# RR ############
'''
rr = RR(sampy, phi)
theta_rr = rr.cal_theta()

pred_rr = rr.predict(test_phi)
mse_rr = MSE(pred_rr, polyy)

print("MSE:" + str(mse_rr))

print("theta_esti:" + str(theta_rr))
print("theta_true:" + str(thtrue))

plot(sampx, sampy, polyx, polyy, pred_rr, "RR")
'''
############# BR #############
'''
alpha = 5
sigma = np.sqrt(2)

br = BR(sampy, phi, alpha, sigma)
[theta_miu, theta_sigma] = br.cal_para()

[pred_miu, pred_sigma] = br.predict(test_phi)
mse_br = MSE(pred_miu, polyy)

print("MSE:" + str(mse_br))

print("theta_esti:" + str(theta_miu))
print("theta_true:" + str(thtrue))

plot_BR(sampx, sampy, polyx, polyy, pred_miu, pred_sigma)
'''
################# (c) #################
