import numpy as np
import scipy.stats as sp
#from Kernels import smoothing_spline_kernel
from scipy.fftpack import fft, fftshift, fftfreq
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
import operator
from functools import reduce

# Define functions
# Function to generate AR(2) model
def gen_ar1(ar_inputs):
    size = len(ar_inputs[0])
    y_0 = ar_inputs[1]
    sigma = ar_inputs[2]
    alpha1 = ar_inputs[3]
    e = sigma*np.random.randn(size)
    x = np.zeros(size)
    for i in np.arange(1,size,1):
        if i == 1:
            x[i] = y_0
        else:
            x[i] = alpha1*x[i-1] + e[i]
    return x

def gen_ar2(ar_inputs):
    size = len(ar_inputs[0])
    y_0 = ar_inputs[1]
    sigma = ar_inputs[2]
    alpha1 = ar_inputs[3]
    alpha2 = ar_inputs[4]
    e = sigma*np.random.randn(size)
    x = np.zeros(size)
    for i in np.arange(1,size,1):
        if i == 1:
            x[i] = y_0
        if i == 2:
            x[i] = y_0
        else:
            x[i] = alpha1*x[i-1] + alpha2*x[i-2] + e[i]
    return x

def covariance(covariance_inputs, kernel):
    if kernel == "squared_exponential":
        train = covariance_inputs[0]
        test = covariance_inputs[1]
        l_scale = np.exp(covariance_inputs[2][0])
        var = np.exp(covariance_inputs[2][1])
        noise_var = np.exp(covariance_inputs[2][2])
        return squared_exponential_kernel(train, test, l_scale, var)

    if kernel == "matern_32":
        train = covariance_inputs[0]
        test = covariance_inputs[1]
        l_scale = np.exp(covariance_inputs[2][0])
        var = np.exp(covariance_inputs[2][1])
        noise_var = np.exp(covariance_inputs[2][2])
        return matern_kernel_32(train, test, l_scale, var)

    if kernel == "matern_52":
        train = covariance_inputs[0]
        test = covariance_inputs[1]
        l_scale = np.exp(covariance_inputs[2][0])
        var = np.exp(covariance_inputs[2][1])
        noise_var = np.exp(covariance_inputs[2][2])
        return matern_kernel_52(train, test, l_scale, var)

    if kernel == "periodic":
        train = covariance_inputs[0]
        test = covariance_inputs[1]
        l_scale = np.exp(covariance_inputs[2][0])
        var = np.exp(covariance_inputs[2][1])
        period = np.exp(covariance_inputs[2][2])
        noise_var = np.exp(covariance_inputs[2][3])
        return periodic(train, test, l_scale, var, period)

    if kernel == "linear":
        train = covariance_inputs[0]
        test = covariance_inputs[1]
        var_bias = np.exp(covariance_inputs[2][0])
        var = np.exp(covariance_inputs[2][1])
        c = np.exp(covariance_inputs[2][2])
        noise_var = np.exp(covariance_inputs[2][3])
        return linear_kernel(train, test, var_bias, var, c)

    if kernel == "neural_network":
        train = covariance_inputs[0]
        test = covariance_inputs[1]
        l_scale = np.exp(covariance_inputs[2][0])
        bias_l_scale = np.exp(covariance_inputs[2][1])
        var = np.exp(covariance_inputs[2][2])
        noise_var = np.exp(covariance_inputs[2][3])
        return neural_network_kernel(train, test, l_scale, bias_l_scale, var)

    if kernel == "smoothing_spline":
        train = covariance_inputs[0]
        test = covariance_inputs[1]
        c = np.exp(covariance_inputs[2][0])
        n = np.exp(covariance_inputs[2][1])
        var = np.exp(covariance_inputs[2][2])
        noise_var = np.exp(covariance_inputs[2][3])
        return smoothing_spline_kernel(train, test, c, n, var, noise_var)

    if kernel == "squared_exponential_x_neural_network":
        train = covariance_inputs[0]
        test = covariance_inputs[1]
        l_scale_sqe = np.exp(covariance_inputs[2][0])
        var_sqe = np.exp(covariance_inputs[2][1])
        l_scale_nn = np.exp(covariance_inputs[2][2])
        bias_l_scale_nn = np.exp(covariance_inputs[2][3])
        var_nn = np.exp(covariance_inputs[2][4])
        noise_var = np.exp(covariance_inputs[2][5])
        return squared_exponential_kernel(train, test, l_scale_sqe, var_sqe) * neural_network_kernel(train, test, l_scale_nn, bias_l_scale_nn, var_nn)

    if kernel == "squared_exponential_+_neural_network":
        train = covariance_inputs[0]
        test = covariance_inputs[1]
        l_scale_sqe = np.exp(covariance_inputs[2][0])
        var_sqe = np.exp(covariance_inputs[2][1])
        l_scale_nn = np.exp(covariance_inputs[2][2])
        bias_l_scale_nn = np.exp(covariance_inputs[2][3])
        var_nn = np.exp(covariance_inputs[2][4])
        noise_var = np.exp(covariance_inputs[2][5])
        return squared_exponential_kernel(train, test, l_scale_sqe, var_sqe) + neural_network_kernel(train, test, l_scale_nn, bias_l_scale_nn, var_nn)

    if kernel == "periodic_+_neural_network":
        train = covariance_inputs[0]
        test = covariance_inputs[1]
        l_scale = np.exp(covariance_inputs[2][0])
        var = np.exp(covariance_inputs[2][1])
        period = np.exp(covariance_inputs[2][2])
        l_scale_nn = np.exp(covariance_inputs[2][3])
        bias_l_scale_nn = np.exp(covariance_inputs[2][4])
        var_nn = np.exp(covariance_inputs[2][5])
        noise_var = np.exp(covariance_inputs[2][6])
        return periodic(train, test, l_scale, var, period) + neural_network_kernel(train, test, l_scale_nn, bias_l_scale_nn, var_nn)

    if kernel == "periodic_+_squared_exponential":
        train = covariance_inputs[0]
        test = covariance_inputs[1]
        l_scale_sqe = np.exp(covariance_inputs[2][0])
        var_sqe = np.exp(covariance_inputs[2][1])
        l_scale = np.exp(covariance_inputs[2][2])
        var = np.exp(covariance_inputs[2][3])
        period = np.exp(covariance_inputs[2][4])
        noise_var = np.exp(covariance_inputs[2][5])
        return np.add(periodic(train, test, l_scale, var, period), squared_exponential_kernel(train, test, l_scale_sqe, var_sqe))

    if kernel == "periodic_x_squared_exponential":
        train = covariance_inputs[0]
        test = covariance_inputs[1]
        l_scale_sqe = np.exp(covariance_inputs[2][0])
        var_sqe = np.exp(covariance_inputs[2][1])
        l_scale = np.exp(covariance_inputs[2][2])
        var = np.exp(covariance_inputs[2][3])
        period = np.exp(covariance_inputs[2][4])
        noise_var = np.exp(covariance_inputs[2][5])
        return periodic(train, test, l_scale, var, period) * squared_exponential_kernel(train, test, l_scale_sqe, var_sqe)


def posterior_predictive(y, covariance_inputs_xx, covariance_inputs_x_xstar, covariance_inputs_xstar_x, covariance_inputs_xstar_xstar, kernel):
    # Generate Kernel: Train/train, train/test, test/train, test/test
    if kernel == "squared_exponential":
        k_xx = covariance(covariance_inputs_xx, kernel) + np.identity(len(y)) * np.exp(covariance_inputs_xx[2][2]) # TRAIN/TRAIN
        k_x_xstar = covariance(covariance_inputs_x_xstar, kernel) # TRAIN/TEST
        k_xstar_x = covariance(covariance_inputs_xstar_x, kernel)# TEST/TRAIN
        k_xstar_xstar = covariance(covariance_inputs_xstar_xstar, kernel) # TEST/TEST
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "matern_32":
        k_xx = covariance(covariance_inputs_xx, kernel) + np.identity(len(y)) * np.exp(covariance_inputs_xx[2][2])  # TRAIN/TRAIN
        k_x_xstar = covariance(covariance_inputs_x_xstar, kernel)  # TRAIN/TEST
        k_xstar_x = covariance(covariance_inputs_xstar_x, kernel)  # TEST/TRAIN
        k_xstar_xstar = covariance(covariance_inputs_xstar_xstar, kernel)  # TEST/TEST
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "matern_52":
        k_xx = covariance(covariance_inputs_xx, kernel) + np.identity(len(y)) * np.exp(covariance_inputs_xx[2][2]) # TRAIN/TRAIN
        k_x_xstar = covariance(covariance_inputs_x_xstar, kernel)  # TRAIN/TEST
        k_xstar_x = covariance(covariance_inputs_xstar_x, kernel)  # TEST/TRAIN
        k_xstar_xstar = covariance(covariance_inputs_xstar_xstar, kernel)  # TEST/TEST
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "periodic":
        k_xx = covariance(covariance_inputs_xx, kernel) + np.identity(len(y)) * np.exp(covariance_inputs_xx[2][3])  # TRAIN/TRAIN
        k_x_xstar = covariance(covariance_inputs_x_xstar, kernel)  # TRAIN/TEST
        k_xstar_x = covariance(covariance_inputs_xstar_x, kernel)  # TEST/TRAIN
        k_xstar_xstar = covariance(covariance_inputs_xstar_xstar, kernel)  # TEST/TEST
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "linear":
        k_xx = covariance(covariance_inputs_xx, kernel) + np.identity(len(y)) * np.exp(covariance_inputs_xx[2][3]) # TRAIN/TRAIN
        k_x_xstar = covariance(covariance_inputs_x_xstar, kernel)  # TRAIN/TEST
        k_xstar_x = covariance(covariance_inputs_xstar_x, kernel)  # TEST/TRAIN
        k_xstar_xstar = covariance(covariance_inputs_xstar_xstar, kernel)  # TEST/TEST
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "neural_network":
        k_xx = np.add(covariance(covariance_inputs_xx, kernel), (np.identity(len(y)) * np.exp(covariance_inputs_xx[2][3])))  # TRAIN/TRAIN
        k_x_xstar = covariance(covariance_inputs_x_xstar, kernel)  # TRAIN/TEST
        k_xstar_x = covariance(covariance_inputs_xstar_x, kernel)  # TEST/TRAIN
        k_xstar_xstar = covariance(covariance_inputs_xstar_xstar, kernel)  # TEST/TEST
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "smoothing_spline":
        k_xx = covariance(covariance_inputs_xx, kernel) + (np.identity(len(y)) * np.exp(covariance_inputs_xx[2][2]))  # TRAIN/TRAIN
        k_x_xstar = covariance(covariance_inputs_x_xstar, kernel)  # TRAIN/TEST
        k_xstar_x = covariance(covariance_inputs_xstar_x, kernel)  # TEST/TRAIN
        k_xstar_xstar = covariance(covariance_inputs_xstar_xstar, kernel)  # TEST/TEST
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "squared_exponential_x_neural_network":
        k_xx = np.add(covariance(covariance_inputs_xx, kernel) , (np.identity(len(y)) * np.exp(covariance_inputs_xx[2][5])))  # TRAIN/TRAIN
        k_x_xstar = covariance(covariance_inputs_x_xstar, kernel)  # TRAIN/TEST
        k_xstar_x = covariance(covariance_inputs_xstar_x, kernel)  # TEST/TRAIN
        k_xstar_xstar = covariance(covariance_inputs_xstar_xstar, kernel)  # TEST/TEST
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "squared_exponential_+_neural_network":
        k_xx = covariance(covariance_inputs_xx, kernel) + np.identity(len(y))*np.exp(covariance_inputs_xx[2][5])  # TRAIN/TRAIN
        k_x_xstar = covariance(covariance_inputs_x_xstar, kernel)  # TRAIN/TEST
        k_xstar_x = covariance(covariance_inputs_xstar_x, kernel)  # TEST/TRAIN
        k_xstar_xstar = covariance(covariance_inputs_xstar_xstar, kernel)  # TEST/TEST
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "periodic_+_neural_network":
        k_xx = covariance(covariance_inputs_xx, kernel) + (np.identity(len(y))*np.exp(covariance_inputs_xx[2][6]))  # TRAIN/TRAIN
        k_x_xstar = covariance(covariance_inputs_x_xstar, kernel)  # TRAIN/TEST
        k_xstar_x = covariance(covariance_inputs_xstar_x, kernel)  # TEST/TRAIN
        k_xstar_xstar = covariance(covariance_inputs_xstar_xstar, kernel)  # TEST/TEST
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "periodic_+_squared_exponential":
        k_xx = covariance(covariance_inputs_xx, kernel) + np.identity(len(y)) * np.exp(covariance_inputs_xx[2][5])  # TRAIN/TRAIN
        k_x_xstar = covariance(covariance_inputs_x_xstar, kernel)  # TRAIN/TEST
        k_xstar_x = covariance(covariance_inputs_xstar_x, kernel)  # TEST/TRAIN
        k_xstar_xstar = covariance(covariance_inputs_xstar_xstar, kernel)  # TEST/TEST
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "periodic_x_squared_exponential":
        k_xx = covariance(covariance_inputs_xx, kernel) + np.identity(len(y)) * np.exp(covariance_inputs_xx[2][5])  # TRAIN/TRAIN
        k_x_xstar = covariance(covariance_inputs_x_xstar, kernel)  # TRAIN/TEST
        k_xstar_x = covariance(covariance_inputs_xstar_x, kernel)  # TEST/TRAIN
        k_xstar_xstar = covariance(covariance_inputs_xstar_xstar, kernel)  # TEST/TEST
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

def posterior_predictive_hyperparameters(y, train, test, theta, kernel):
    if kernel == "squared_exponential":
        l_scale = theta[0]
        var = theta[1]
        noise_var = theta[2]
        k_xx = squared_exponential_kernel(train, train, l_scale, var) + np.eye(len(train)) * noise_var
        k_x_xstar = squared_exponential_kernel(train, test, l_scale, var)
        k_xstar_x = squared_exponential_kernel(test, train, l_scale, var)
        k_xstar_xstar = squared_exponential_kernel(test, test, l_scale, var)
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "matern_32":
        l_scale = theta[0]
        var = theta[1]
        noise_var = theta[2]
        k_xx = matern_kernel_32(train, train, l_scale, var) + np.eye(len(train)) * noise_var
        k_x_xstar = matern_kernel_32(train, test, l_scale, var)
        k_xstar_x = matern_kernel_32(test, train, l_scale, var)
        k_xstar_xstar = matern_kernel_32(test, test, l_scale, var)
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "matern_52":
        l_scale = theta[0]
        var = theta[1]
        noise_var = theta[2]
        k_xx = matern_kernel_52(train, train, l_scale, var) + np.eye(len(train)) * noise_var
        k_x_xstar = matern_kernel_52(train, test, l_scale, var)
        k_xstar_x = matern_kernel_52(test, train, l_scale, var)
        k_xstar_xstar = matern_kernel_52(test, test, l_scale, var)
        # Generate Covariance matrix - predictive distribution
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "periodic":
        l_scale = theta[0]
        var = theta[1]
        period = theta[2]
        noise_var = theta[3]
        k_xx = periodic(train, train, l_scale, var, period) + np.eye(len(train)) * noise_var
        k_x_xstar = periodic(train, test, l_scale, var, period)
        k_xstar_x = periodic(test, train, l_scale, var, period)
        k_xstar_xstar = periodic(test, test, l_scale, var, period)
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "linear":
        var_bias = theta[0]
        var = theta[1]
        c = theta[2]
        noise_var = theta[3]
        k_xx = linear_kernel(train, train, var_bias, var, c) + np.eye(len(train)) * noise_var
        k_x_xstar = linear_kernel(train, test, var_bias, var, c)
        k_xstar_x = linear_kernel(test, train, var_bias, var, c)
        k_xstar_xstar = linear_kernel(test, test, var_bias, var, c)
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "neural_network":
        l_scale = theta[0]
        bias_l_scale = theta[1]
        var = theta[2]
        noise_var = theta[3]
        k_xx = neural_network_kernel(train, train, l_scale, bias_l_scale, var) + np.eye(len(train)) * noise_var
        k_x_xstar = neural_network_kernel(train, test, l_scale, bias_l_scale, var)
        k_xstar_x = neural_network_kernel(test, train, l_scale, bias_l_scale, var)
        k_xstar_xstar = neural_network_kernel(test, test, l_scale, bias_l_scale, var)
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "smoothing_spline":
        c = theta[0]
        n = theta[1]
        var = theta[2]
        noise_var = theta[3]
        k_xx = smoothing_spline_kernel(train, train, c, n, var, noise_var) + np.eye(len(train)) * noise_var
        k_x_xstar = smoothing_spline_kernel(train, test, c, n, var, noise_var)
        k_xstar_x = smoothing_spline_kernel(test, train, c, n, var, noise_var)
        k_xstar_xstar = smoothing_spline_kernel(test, test, c, n, var, noise_var)
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "squared_exponential_x_neural_network":
        l_scale_sqe = theta[0]
        var_sqe = theta[1]
        l_scale_nn = theta[2]
        bias_l_scale_nn = theta[3]
        var_nn = theta[4]
        noise_var = theta[5]
        k_xx = (neural_network_kernel(train, train, l_scale_nn, bias_l_scale_nn, var_nn) * squared_exponential_kernel(train, train, l_scale_sqe, var_sqe)) + np.eye(len(train)) * noise_var
        k_x_xstar = (neural_network_kernel(train, test, l_scale_nn, bias_l_scale_nn, var_nn) * squared_exponential_kernel(train, test, l_scale_sqe, var_sqe))
        k_xstar_x = (neural_network_kernel(test, train, l_scale_nn, bias_l_scale_nn, var_nn) * squared_exponential_kernel(test, train, l_scale_sqe, var_sqe))
        k_xstar_xstar = (neural_network_kernel(test, test, l_scale_nn, bias_l_scale_nn, var_nn) * squared_exponential_kernel(test, test, l_scale_sqe, var_sqe))
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "squared_exponential_+_neural_network":
        l_scale_sqe = theta[0]
        var_sqe = theta[1]
        l_scale_nn = theta[2]
        bias_l_scale_nn = theta[3]
        var_nn = theta[4]
        noise_var = theta[5]
        k_xx = (neural_network_kernel(train, train, l_scale_nn, bias_l_scale_nn, var_nn) + squared_exponential_kernel(train, train, l_scale_sqe, var_sqe)) + np.eye(len(train)) * noise_var
        k_x_xstar = (neural_network_kernel(train, test, l_scale_nn, bias_l_scale_nn, var_nn) + squared_exponential_kernel(train, test,l_scale_sqe, var_sqe))
        k_xstar_x = (neural_network_kernel(test, train, l_scale_nn, bias_l_scale_nn, var_nn) + squared_exponential_kernel(test, train,l_scale_sqe, var_sqe))
        k_xstar_xstar = (neural_network_kernel(test, test, l_scale_nn, bias_l_scale_nn, var_nn) + squared_exponential_kernel(test, test,l_scale_sqe,var_sqe))
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "periodic_+_neural_network":
        l_scale = theta[0]
        var = theta[1]
        period = theta[2]
        l_scale_nn = theta[3]
        bias_l_scale_nn = theta[4]
        var_nn = theta[5]
        noise_var = theta[6]
        k_xx = periodic(train, train, l_scale, var, period) + neural_network_kernel(train, train, l_scale_nn, bias_l_scale_nn, var_nn) + np.eye(len(train)) * noise_var
        k_x_xstar = periodic(train, test, l_scale, var, period) + neural_network_kernel(train, test, l_scale_nn, bias_l_scale_nn, var_nn)
        k_xstar_x = periodic(test, train, l_scale, var, period) + neural_network_kernel(test, train, l_scale_nn, bias_l_scale_nn, var_nn)
        k_xstar_xstar = periodic(test, test, l_scale, var, period) + neural_network_kernel(test, test, l_scale_nn, bias_l_scale_nn, var_nn)
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "periodic_+_squared_exponential":
        l_scale_sqe = theta[0]
        var_sqe = theta[1]
        l_scale = theta[2]
        var = theta[3]
        period = theta[4]
        noise_var = theta[5]
        k_xx = periodic(train, train, l_scale, var, period) + squared_exponential_kernel(train, train, l_scale_sqe, var_sqe) + np.eye(len(train)) * noise_var
        k_x_xstar = periodic(train, test, l_scale, var, period) + squared_exponential_kernel(train, test, l_scale_sqe, var_sqe)
        k_xstar_x = periodic(test, train, l_scale, var, period) + squared_exponential_kernel(test, train, l_scale_sqe, var_sqe)
        k_xstar_xstar = periodic(test, test, l_scale, var, period) + squared_exponential_kernel(test, test, l_scale_sqe, var_sqe)
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star

    if kernel == "periodic_x_squared_exponential":
        l_scale_sqe = theta[0]
        var_sqe = theta[1]
        l_scale = theta[2]
        var = theta[3]
        period = theta[4]
        noise_var = theta[5]
        k_xx = periodic(train, train, l_scale, var, period) * squared_exponential_kernel(train, train, l_scale_sqe, var_sqe) + np.eye(len(train)) * noise_var
        k_x_xstar = periodic(train, test, l_scale, var, period) * squared_exponential_kernel(train, test, l_scale_sqe, var_sqe)
        k_xstar_x = periodic(test, train, l_scale, var, period) * squared_exponential_kernel(test, train, l_scale_sqe, var_sqe)
        k_xstar_xstar = periodic(test, test, l_scale, var, period) * squared_exponential_kernel(test, test, l_scale_sqe, var_sqe)
        f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y))
        cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
        return f_star, cov_f_star


# Defining prior distributions - probability density functions
# Define Tau Squared Inverse Gamma PDF
# Define beta MVN PDF

def beta_c_rvs(tau_squared_c, x):
    rvs = sp.multivariate_normal.rvs(0, tau_squared_c, x)
    return rvs

def tau_squared_rvs(a, b):
    rvs = sp.invgamma.rvs(a)
    return rvs*b

def logpdf_lscale(log_l_scale):
    l_scale = np.exp(log_l_scale)
    logpdf = sp.norm.logpdf(l_scale, 20, 1)
    return logpdf

def logpdf_var(log_var):
    var = np.exp(log_var)
    logpdf = sp.norm.logpdf(var, 10, 1)
    return logpdf

def logpdf_period(log_period):
    period = np.exp(log_period)
    logpdf = sp.norm.logpdf(period, 10, 1)
    return logpdf

def logpdf_noise_var(log_noise_var):
    noise_var = np.exp(log_noise_var)
    logpdf = sp.norm.logpdf(noise_var, 10, 1)
    return logpdf

def logpdf_alpha(log_alpha):
    alpha = np.exp(log_alpha)
    logpdf = sp.norm.logpdf(alpha, 0, 100)
    return logpdf

def logpdf_beta(beta, tau_squared):
    logpdf = sp.multivariate_normal.logpdf(x = beta, cov = tau_squared*np.identity(len(beta)))
    return logpdf

def logpdf_n(log_n):
    n = np.exp(log_n)
    logpdf = sp.invgamma.logpdf(n, 100)
    return logpdf

# Define beta proposed
def nu_p_beta(beta_c):
    beta_proposal = beta_c + sp.multivariate_normal.rvs(np.zeros(len(beta_c)), 0.1*np.identity(len(beta_c)))
    return beta_proposal

def nu_p_lscale(lscale):
    noise = sp.norm.rvs(0, .2)
    rvs = lscale + noise
    return rvs

def nu_p_var(var):
    noise = sp.norm.rvs(0, 0.1)
    rvs = var + noise
    return rvs

def nu_p_period(period):
    noise = sp.norm.rvs(0, 0.1)
    rvs = period + noise
    return rvs

def nu_p_noise_var(noise_var):
    noise = sp.norm.rvs(0, .1)
    rvs = noise_var + noise
    return rvs

def nu_p_alpha(alpha):
    noise = sp.norm.rvs(0, .1)
    rvs = alpha + noise
    return rvs

# Define Likelihood Function
def whittlelikelihood(y, train, test, theta, kernel): # y should be the log adjusted periodogram
    f_star, cov_f_star = posterior_predictive_hyperparameters(y, train, test, theta, kernel)
    g = f_star
    likelihood = -np.sum(g + np.exp(y)/np.exp(g))
    return likelihood

# Define kernels
def squared_exponential_kernel(train, test, l_scale, var):
    omega = np.zeros((len(train), len(test)))
    for i in range(len(train)):
        for j in range(len(test)):
            tau = train[i] - test[j]
            omega[i, j] = var * np.exp(-(tau**2)/(2*l_scale**2)) # train/test
    return omega

def matern_kernel_32(train, test, l_scale, var):
    omega = np.zeros((len(train), len(test)))
    for i in range(len(train)):
        for j in range(len(test)):
            tau = train[i] - test[j]
            omega[i,j] = var*((1 + (np.sqrt(3*tau**2))/l_scale)) * np.exp(-(np.sqrt(3*tau**2))/l_scale)
    return omega

def matern_kernel_52(train, test, l_scale, var):
    omega = np.zeros((len(train), len(test)))
    for i in range(len(train)):
        for j in range(len(test)):
            tau = train[i] - test[j]
            omega[i,j] = var * ((1 + (np.sqrt(5*tau**2)/l_scale) + (5*tau**2)/(3*l_scale**2)) * np.exp((-np.sqrt(5*tau**2)/l_scale)))
    return omega

def periodic(train, test, l_scale, var, period): # This is periodic kernel using warped input
    omega = np.zeros((len(train), len(test)))
    for i in range(len(train)):
        for j in range(len(test)):
            tau = train[i] - test[j]
            omega[i, j] = var * np.exp((-2*np.sin(np.pi*tau*(1/period))**2)/l_scale**2)
    return omega

def neural_network_kernel(train, test, l_scale, bias_l_scale, var):
    sigma_matrix = np.diag([l_scale, bias_l_scale])
    omega = np.zeros((len(train), len(test)))
    for i in range(len(train)):
        for j in range(len(test)):
            numerator = (2*np.dot(np.transpose(([1, train[i]])), np.dot(sigma_matrix, ([1, test[j]]))))
            denominator = np.sqrt((1 + np.dot(2 * np.transpose(([1, train[i]])), np.dot(sigma_matrix, ([1, train[i]])))) * (1 + np.dot(2 * np.transpose(([1, test[j]])), np.dot(sigma_matrix, ([1, test[j]])))))
            omega[i,j] = var * (2/np.pi)*(np.arcsin(numerator/denominator))
    return omega

def linear_kernel(train, test, var_bias, var, c):
    omega = np.zeros((len(train), len(test)))
    for i in range(len(train)):
        for j in range(len(test)):
            omega[i, j] = var_bias + var * (train[i]-c)*(test[j]-c)
    return omega

def eigen_decomposition(omega, num_eigen):
    # Eigen Decomposition with basis function reconstruction
    vals, vecs = np.linalg.eig(omega)
    diag = np.diag(np.sqrt(vals))
    X_complete = np.dot(vecs, diag)  # Full X matrix
    vals = vals[0:num_eigen]
    vecs = vecs[:, 0:num_eigen]
    diag = diag[0:num_eigen, 0:num_eigen]
    X_eigen = np.array(np.dot(vecs, diag))

    return X_eigen

def analytic_spectrum_ar1(sigma, phi_1, frequencies):
    numerator = (sigma**2)
    denominator = 1 + (phi_1**2) - (2*phi_1*np.cos(2*np.pi*frequencies))
    return numerator/denominator

def analytic_spectrum_ar2(sigma, phi_1, phi_2, frequencies):
    numerator = sigma**2
    denominator = (1 + phi_1**2 + phi_2**2 - 2*phi_1*(1-phi_2)*np.cos(2*np.pi*frequencies) - 2*phi_2*np.cos(4*np.pi*frequencies))
    return numerator/denominator

def rmse(predicted, actual):
    n = len(predicted)
    return (1/n) * np.sqrt(np.sum(predicted-actual)**2)

def maximum_likelihood_estimate(mcmc_values):
    n = len(mcmc_values)
    total_value = np.sum(mcmc_values)
    max_likelihood = n/total_value
    return max_likelihood

def log_likelihood_exponential(mcmc_values, ground_truth, maximum_likelihood):
    n = len(mcmc_values)
    logpdf = np.log(maximum_likelihood) - maximum_likelihood*ground_truth
    return logpdf

def symmetric_kl_divergence(ground_truth, spectral_estimate):
    kl = np.sum(ground_truth*np.log(ground_truth/spectral_estimate) + spectral_estimate*np.log(spectral_estimate/ground_truth))
    return kl

def neg_loglikelihood(y, train, test, theta, kernel):
    l_scale_sqe = theta[0]
    var_sqe = theta[1]
    l_scale_nn = theta[2]
    bias_l_scale_nn = theta[3]
    var_nn = theta[4]
    noise_var = theta[5]

    k_xx = (neural_network_kernel(train, train, l_scale_nn, bias_l_scale_nn, var_nn) * squared_exponential_kernel(train,train,l_scale_sqe,var_sqe)) + np.eye(len(train)) * noise_var
    k_x_xstar = (neural_network_kernel(train, test, l_scale_nn, bias_l_scale_nn, var_nn) * squared_exponential_kernel(train, test,l_scale_sqe,var_sqe))
    k_xstar_x = (neural_network_kernel(test, train, l_scale_nn, bias_l_scale_nn, var_nn) * squared_exponential_kernel(test, train,l_scale_sqe,var_sqe))
    k_xstar_xstar = (neural_network_kernel(test, test, l_scale_nn, bias_l_scale_nn, var_nn) * squared_exponential_kernel(test, test,l_scale_sqe,var_sqe))

    top = np.concatenate((k_xx, k_x_xstar), axis=0)
    bottom = np.concatenate((k_xstar_x, k_xstar_xstar), axis=0)
    omega = np.concatenate((top, bottom), axis=1)

    neg_log_likelihood = 0.5 * np.dot(np.dot(np.transpose(y), np.linalg.inv(omega)), y) + 0.5 * np.log(np.linalg.det(omega)) + (N_gen/2)*np.log(2*np.pi)
    return neg_log_likelihood

