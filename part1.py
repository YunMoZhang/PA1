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

# phi = transform_part1(sampx) 
# test_phi = transform_part1(polyx)

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
'''
for percent in range(10, 100, 5):
    mse_sum = 0
    trial_num = 5

    print("Percentage:" + str(percent))
    for trial in range(trial_num):
        subsampx, subsampy = subsetsamp(sampx, sampy, percent)
        phi = transform_part1(subsampx) 
        test_phi = transform_part1(polyx)
        ############# LS ############     

        ls = LS(subsampy, phi)
        theta_ls = ls.cal_theta()

        pred_ls = ls.predict(test_phi)
        mse_ls = MSE(pred_ls, polyy)
        mse_sum += mse_ls
        print("Trial" + str(trial + 1) + ":")
        print("MSE:" + str(mse_ls))

        print("theta_esti:" + str(theta_ls))
        print("theta_true:" + str(thtrue))
        print("")

        plot(subsampx, subsampy, polyx, polyy, pred_ls, "_sub" + str(percent) + "LS")
        
        ########### RLS ############
        
        lam_rls = 0.1
        rls = RLS(subsampy, phi, lam_rls)
        theta_rls = rls.cal_theta()

        pred_rls = rls.predict(test_phi)
        mse_rls = MSE(pred_rls, polyy)
        mse_sum += mse_rls
        print("Trial" + str(trial + 1) + ":")
        print("MSE:" + str(mse_rls))

        print("theta_esti:" + str(theta_rls))
        print("theta_true:" + str(thtrue))
        print("")
        plot(subsampx, subsampy, polyx, polyy, pred_rls, "_sub" + str(percent) + "RLS")
        
        ########### LASSO ############
        
        lam_lasso = 0.001
        lasso = LASSO(subsampy, phi, lam_lasso)
        theta_lasso = lasso.cal_theta()

        pred_lasso = lasso.predict(test_phi)
        mse_lasso = MSE(pred_lasso, polyy)

        mse_sum += mse_lasso
        #print("Trial" + str(trial + 1) + ":")

        #print("MSE:" + str(mse_lasso))

        #print("theta_esti:" + str(theta_lasso))
        #print("theta_true:" + str(thtrue))

        plot(subsampx, subsampy, polyx, polyy, pred_lasso, "_sub" + str(percent) + "LASSO")
        
        ############# RR ############
        '
        rr = RR(subsampy, phi)
        theta_rr = rr.cal_theta()

        pred_rr = rr.predict(test_phi)
        mse_rr = MSE(pred_rr, polyy)
        mse_sum += mse_rr
        #print("MSE:" + str(mse_rr))

        #print("theta_esti:" + str(theta_rr))
        #print("theta_true:" + str(thtrue))

        plot(subsampx, subsampy, polyx, polyy, pred_rr, "_sub" + str(percent) + "RR")
        
        ############# BR #############
           
        alpha = 5
        sigma = np.sqrt(2)

        br = BR(subsampy, phi, alpha, sigma)
        [theta_miu, theta_sigma] = br.cal_para()

        [pred_miu, pred_sigma] = br.predict(test_phi)
        mse_br = MSE(pred_miu, polyy)
        mse_sum += mse_br
        #print("MSE:" + str(mse_br))

        #print("theta_esti:" + str(theta_miu))
        #print("theta_true:" + str(thtrue))

        plot_BR(subsampx, subsampy, polyx, polyy, pred_miu, pred_sigma, "_sub" + str(percent) + "BR")
        
        

    mse_ave = mse_sum/trial_num
    print("Average Error:" + str(mse_ave))
'''
'''
################# (d) #################

phi = transform_part1(sampx) 
test_phi = transform_part1(polyx)

added_sampy = addoutlier(sampy)

############# LS ############
print("LS:")
ls = LS(added_sampy, phi)
theta_ls = ls.cal_theta()


pred_ls = ls.predict(test_phi)
mse_ls = MSE(pred_ls, polyy)

print("MSE:" + str(mse_ls))

print("theta_esti:" + str(theta_ls))
print("theta_true:" + str(thtrue))

plot(sampx, added_sampy, polyx, polyy, pred_ls, "addoutlier_LS")

########### RLS ############
print("RLS:")
lam_rls = 0.1
rls = RLS(added_sampy, phi, lam_rls)
theta_rls = rls.cal_theta()

pred_rls = rls.predict(test_phi)
mse_rls = MSE(pred_rls, polyy)

print("MSE:" + str(mse_rls))

print("theta_esti:" + str(theta_rls))
print("theta_true:" + str(thtrue))

plot(sampx, added_sampy, polyx, polyy, pred_rls, "addoutlier_RLS")

########### LASSO ############
print("LASSO:")
lam_lasso = 0.001
lasso = LASSO(added_sampy, phi, lam_lasso)
theta_lasso = lasso.cal_theta()

pred_lasso = lasso.predict(test_phi)
mse_lasso = MSE(pred_lasso, polyy)

print("MSE:" + str(mse_lasso))

print("theta_esti:" + str(theta_lasso))
print("theta_true:" + str(thtrue))

plot(sampx, added_sampy, polyx, polyy, pred_lasso, "addoutlier_LASSO")

############# RR ############
print("RR:")
rr = RR(added_sampy, phi)
theta_rr = rr.cal_theta()

pred_rr = rr.predict(test_phi)
mse_rr = MSE(pred_rr, polyy)

print("MSE:" + str(mse_rr))

print("theta_esti:" + str(theta_rr))
print("theta_true:" + str(thtrue))

plot(sampx, added_sampy, polyx, polyy, pred_rr, "addoutlier_RR")

############# BR #############
print("BR:")
alpha = 5
sigma = np.sqrt(2)

br = BR(added_sampy, phi, alpha, sigma)
[theta_miu, theta_sigma] = br.cal_para()

[pred_miu, pred_sigma] = br.predict(test_phi)
mse_br = MSE(pred_miu, polyy)

print("MSE:" + str(mse_br))

print("theta_esti:" + str(theta_miu))
print("theta_true:" + str(thtrue))

plot_BR(sampx, added_sampy, polyx, polyy, pred_miu, pred_sigma, "addoutlier_BR")
'''


################# (e) #################

phi = transform_part1(sampx) 
test_phi = transform_part1(polyx)



############# LS ############
print("LS:")
ls = LS(sampy, phi)
theta_ls = ls.cal_theta()


pred_ls = ls.predict(test_phi)
mse_ls = MSE(pred_ls, polyy)

print("MSE:" + str(mse_ls))

print("theta_esti:" + str(theta_ls))
print("theta_true:" + str(thtrue))

plot(sampx, sampy, polyx, polyy, pred_ls, "order10_LS")

########### RLS ############
print("RLS:")
lam_rls = 0.1
rls = RLS(sampy, phi, lam_rls)
theta_rls = rls.cal_theta()

pred_rls = rls.predict(test_phi)
mse_rls = MSE(pred_rls, polyy)

print("MSE:" + str(mse_rls))

print("theta_esti:" + str(theta_rls))
print("theta_true:" + str(thtrue))

plot(sampx, sampy, polyx, polyy, pred_rls, "order10_RLS")

########### LASSO ############
print("LASSO:")
lam_lasso = 0.001
lasso = LASSO(sampy, phi, lam_lasso)
theta_lasso = lasso.cal_theta()

pred_lasso = lasso.predict(test_phi)
mse_lasso = MSE(pred_lasso, polyy)

print("MSE:" + str(mse_lasso))

print("theta_esti:" + str(theta_lasso))
print("theta_true:" + str(thtrue))

plot(sampx, sampy, polyx, polyy, pred_lasso, "order10_LASSO")

############# RR ############
print("RR:")
rr = RR(sampy, phi)
theta_rr = rr.cal_theta()

pred_rr = rr.predict(test_phi)
mse_rr = MSE(pred_rr, polyy)

print("MSE:" + str(mse_rr))

print("theta_esti:" + str(theta_rr))
print("theta_true:" + str(thtrue))

plot(sampx, sampy, polyx, polyy, pred_rr, "order10_RR")

############# BR #############
print("BR:")
alpha = 5
sigma = np.sqrt(2)

br = BR(sampy, phi, alpha, sigma)
[theta_miu, theta_sigma] = br.cal_para()

[pred_miu, pred_sigma] = br.predict(test_phi)
mse_br = MSE(pred_miu, polyy)

print("MSE:" + str(mse_br))

print("theta_esti:" + str(theta_miu))
print("theta_true:" + str(thtrue))

plot_BR(sampx, sampy, polyx, polyy, pred_miu, pred_sigma, "order10_BR")
