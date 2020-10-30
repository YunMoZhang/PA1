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

trainx = np.loadtxt('/home/yunmzhang2/cs5487/pa1-zym/PA-1-data-text/count_data_trainx.txt')
trainy = np.loadtxt('/home/yunmzhang2/cs5487/pa1-zym/PA-1-data-text/count_data_trainy.txt')
testx = np.loadtxt('/home/yunmzhang2/cs5487/pa1-zym/PA-1-data-text/count_data_testx.txt')
testy = np.loadtxt('/home/yunmzhang2/cs5487/pa1-zym/PA-1-data-text/count_data_testy.txt')

trainx = trainx.T
trainy = trainy.T
testx = testx.T
testy = testy.T
'''
################# (a) #################

phi = transform_part2(trainx) 
test_phi = transform_part2(testx)

############# LS ############
print("LS")
ls = LS(trainy, phi)
theta_ls = ls.cal_theta()


pred_ls = ls.predict(test_phi)
mse_ls = MSE(pred_ls, testy)
mae_ls = MAE(pred_ls, testy)

print("MSE:" + str(mse_ls))
print("MAE:" + str(mae_ls))
print("theta_esti:" + str(theta_ls))


plot_part2(testy, pred_ls, "LS")

########### RLS ############
print("RLS")
lam_rls = 0.1
rls = RLS(trainy, phi, lam_rls)
theta_rls = rls.cal_theta()

pred_rls = rls.predict(test_phi)
mse_rls = MSE(pred_rls, testy)
mae_rls = MAE(pred_rls, testy)

print("MSE:" + str(mse_rls))
print("MAE:" + str(mae_rls))
print("theta_esti:" + str(theta_rls))


plot_part2(testy, pred_rls, "RLS")

########### LASSO ############
print("LASSO")
lam_lasso = 0.001
lasso = LASSO(trainy, phi, lam_lasso)
theta_lasso = lasso.cal_theta()

pred_lasso = lasso.predict(test_phi)
mse_lasso = MSE(pred_lasso, testy)
mae_lasso = MAE(pred_lasso, testy)

print("MSE:" + str(mse_lasso))
print("MAE:" + str(mae_lasso))
print("theta_esti:" + str(theta_lasso))

plot_part2(testy, pred_lasso, "LASSO")

############# RR ############
print("RR")
rr = RR(trainy, phi)
theta_rr = rr.cal_theta()

pred_rr = rr.predict(test_phi)
mse_rr = MSE(pred_rr, testy)
mae_rr = MAE(pred_rr, testy)

print("MSE:" + str(mse_rr))
print("MAE:" + str(mae_rr))
print("theta_esti:" + str(theta_rr))

plot_part2(testy, pred_rr, "RR")

############# BR #############
print("BR")
alpha = 5
sigma = np.sqrt(2)

br = BR(trainy, phi, alpha, sigma)
[theta_miu, theta_sigma] = br.cal_para()

[pred_miu, pred_sigma] = br.predict(test_phi)
mse_br = MSE(pred_miu, testy)
mae_br = MAE(pred_miu, testy)

print("MSE:" + str(mse_br))
print("MAE:" + str(mae_br))
print("theta_esti:" + str(theta_miu))

plot_part2(testy, pred_miu, "BR")
'''
'''
################# (b) #################

phi = transform_part2(trainx, degree = 2) 
test_phi = transform_part2(testx, degree = 2)

############# LS ############
print("LS")
ls = LS(trainy, phi)
theta_ls = ls.cal_theta()


pred_ls = ls.predict(test_phi)
mse_ls = MSE(pred_ls, testy)
mae_ls = MAE(pred_ls, testy)

print("MSE:" + str(mse_ls))
print("MAE:" + str(mae_ls))
print("theta_esti:" + str(theta_ls))


plot_part2(testy, pred_ls, "order2-LS")

########### RLS ############
print("RLS")
lam_rls = 0.1
rls = RLS(trainy, phi, lam_rls)
theta_rls = rls.cal_theta()

pred_rls = rls.predict(test_phi)
mse_rls = MSE(pred_rls, testy)
mae_rls = MAE(pred_rls, testy)

print("MSE:" + str(mse_rls))
print("MAE:" + str(mae_rls))
print("theta_esti:" + str(theta_rls))


plot_part2(testy, pred_rls, "order2-RLS")

########### LASSO ############
print("LASSO")
lam_lasso = 0.001
lasso = LASSO(trainy, phi, lam_lasso)
theta_lasso = lasso.cal_theta()

pred_lasso = lasso.predict(test_phi)
mse_lasso = MSE(pred_lasso, testy)
mae_lasso = MAE(pred_lasso, testy)

print("MSE:" + str(mse_lasso))
print("MAE:" + str(mae_lasso))
print("theta_esti:" + str(theta_lasso))

plot_part2(testy, pred_lasso, "order2-LASSO")

############# RR ############
print("RR")
rr = RR(trainy, phi)
theta_rr = rr.cal_theta()

pred_rr = rr.predict(test_phi)
mse_rr = MSE(pred_rr, testy)
mae_rr = MAE(pred_rr, testy)

print("MSE:" + str(mse_rr))
print("MAE:" + str(mae_rr))
print("theta_esti:" + str(theta_rr))

plot_part2(testy, pred_rr, "order2-RR")

############# BR #############
print("BR")
alpha = 5
sigma = np.sqrt(2)

br = BR(trainy, phi, alpha, sigma)
[theta_miu, theta_sigma] = br.cal_para()

[pred_miu, pred_sigma] = br.predict(test_phi)
mse_br = MSE(pred_miu, testy)
mae_br = MAE(pred_miu, testy)

print("MSE:" + str(mse_br))
print("MAE:" + str(mae_br))
print("theta_esti:" + str(theta_miu))

plot_part2(testy, pred_miu, "order2-BR")
'''
################# (b-with cross) #################

phi = transform_part2(trainx, degree = 2, inter = True) 
test_phi = transform_part2(testx, degree = 2, inter = True)

############# LS ############
print("LS")
ls = LS(trainy, phi)
theta_ls = ls.cal_theta()


pred_ls = ls.predict(test_phi)
mse_ls = MSE(pred_ls, testy)
mae_ls = MAE(pred_ls, testy)

print("MSE:" + str(mse_ls))
print("MAE:" + str(mae_ls))
print("theta_esti:" + str(theta_ls))


plot_part2(testy, pred_ls, "order2-cro-LS")

########### RLS ############
print("RLS")
lam_rls = 0.1
rls = RLS(trainy, phi, lam_rls)
theta_rls = rls.cal_theta()

pred_rls = rls.predict(test_phi)
mse_rls = MSE(pred_rls, testy)
mae_rls = MAE(pred_rls, testy)

print("MSE:" + str(mse_rls))
print("MAE:" + str(mae_rls))
print("theta_esti:" + str(theta_rls))


plot_part2(testy, pred_rls, "order2-cro-RLS")

########### LASSO ############
print("LASSO")
lam_lasso = 0.001
lasso = LASSO(trainy, phi, lam_lasso)
theta_lasso = lasso.cal_theta()

pred_lasso = lasso.predict(test_phi)
mse_lasso = MSE(pred_lasso, testy)
mae_lasso = MAE(pred_lasso, testy)

print("MSE:" + str(mse_lasso))
print("MAE:" + str(mae_lasso))
print("theta_esti:" + str(theta_lasso))

plot_part2(testy, pred_lasso, "order2-cro-LASSO")

############# RR ############
print("RR")
rr = RR(trainy, phi)
theta_rr = rr.cal_theta()

pred_rr = rr.predict(test_phi)
mse_rr = MSE(pred_rr, testy)
mae_rr = MAE(pred_rr, testy)

print("MSE:" + str(mse_rr))
print("MAE:" + str(mae_rr))
print("theta_esti:" + str(theta_rr))

plot_part2(testy, pred_rr, "order2-cro-RR")

############# BR #############
print("BR")
alpha = 5
sigma = np.sqrt(2)

br = BR(trainy, phi, alpha, sigma)
[theta_miu, theta_sigma] = br.cal_para()

[pred_miu, pred_sigma] = br.predict(test_phi)
mse_br = MSE(pred_miu, testy)
mae_br = MAE(pred_miu, testy)

print("MSE:" + str(mse_br))
print("MAE:" + str(mae_br))
print("theta_esti:" + str(theta_miu))

plot_part2(testy, pred_miu, "order2-cro-BR")
