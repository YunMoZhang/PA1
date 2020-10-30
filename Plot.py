import matplotlib.pyplot as plt
import numpy as np

def plot(sampx, sampy, polyx, polyy, predicty, reg_name = ""):
    plt.scatter(sampx, sampy, c='b', label = "sample")
    plt.plot(polyx, predicty, c='b', label = "estimated" + reg_name + "function")
    plt.plot(polyx, polyy, c='r', label = "true function")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc = 'best')
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
    plt.savefig("./figures-e/part1_t2" + reg_name, dpi = 100)
    plt.show() 
    plt.close()  

def plot_BR(sampx, sampy, polyx, polyy, pred_miu, pred_sigma, reg_name = "BR.png"):
    plt.scatter(sampx, sampy, c='b', label = "sample")
    plt.plot(polyx, pred_miu, c='b', label = "estimatedBR")
    plt.plot(polyx, polyy, c='r', label = "true function")
    plt.fill_between(polyx, pred_miu - np.diag(pred_sigma), pred_miu + np.diag(pred_sigma), label = "stand devia around mean")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc = 'best')
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
    plt.savefig("./figures-e/part1_t2" + reg_name, dpi = 100)
    plt.show()
    plt.close()
