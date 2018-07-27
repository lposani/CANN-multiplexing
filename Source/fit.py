from imports import *

def model_func_exp(t, A, K, C):
    return A * np.exp(K * t) + C

def fit_exp_linear(t, y, C=0):
    y = y - C
    t=t[y>0]
    y=y[y>0]
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K

def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func_exp, t, y, maxfev=100000)
    A, K, C = opt_parms
    return A, K, C

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def fit_bimodal(data):
    y, x, _ = plt.hist(data, 100, alpha=.3, label='data')
    params,cov=sp.optimize.curve_fit(bimodal,x,y,maxfev=100000)