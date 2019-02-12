import numpy as np
from kernels import *
import sklearn.gaussian_process.kernels as kern


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N



# Given the current values of the mean and covariances hypers this is computing the posterior (or predictive) mean and covariance
def predictiveGP(function, sigma_square_f, Omega, N, x, x_star, lenghtscale_gt):
    ## Compute K given the known lenghscale and the current sigma^2_f to get the complete covariance function of the GP
    ## at different inputs
    x_star = x_star[:,np.newaxis]
    x = x[:,np.newaxis]
    k_xstar_x = sigma_square_f*kern.RBF(lenghtscale_gt)(x_star, x)
    k_xstar_xstar = sigma_square_f*kern.RBF(lenghtscale_gt)(x_star, x_star)
    k_xx = sigma_square_f*Omega + np.eye(N) * 0.001
    k_x_xstar = sigma_square_f*kern.RBF(lenghtscale_gt)(x, x_star)

    f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), function))
    cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
    return f_star, cov_f_star


def predictiveGP_spline(function, tau_square, Omega, N, x, x_star):
    ## Compute K given the known lenghscale and the current sigma^2_f to get the complete covariance function of the GP
    ## at different inputs
    x_star = x_star[:,np.newaxis]
    x = x[:,np.newaxis]
    k_xstar_x = tau_square*thin_plate_spline_kernel_xpart(x_star, x)
    k_xstar_xstar = tau_square*thin_plate_spline_kernel_xpart(x_star, x_star)
    k_xx = tau_square*Omega + np.eye(N) * 0.001
    k_x_xstar = tau_square*thin_plate_spline_kernel_xpart(x, x_star)

    f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), function))
    cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)
    return f_star, cov_f_star



# Given the current values of the mean and covariances hypers this is computing the posterior (or predictive) mean and covariance
def posteriorGP_GaussianLik(l_scale, var, noise_var, N, x, x_star, y_gp):
    omega = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            omega[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x[i] - x[j]) ** 2))

    # Generate Covariance matrix - predictive distribution
    k_xx = omega + np.eye(N) * noise_var  # K(X,X)

    k_x_xstar = np.zeros((N, len(x_star)))  # K(X, X_star)
    for i in range(N):
        for j in range(len(x_star)):
            k_x_xstar[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x[i] - x[j]) ** 2))

    k_xstar_x = np.zeros((len(x_star), N))  # K(X_star, X)
    for i in range(len(x_star)):
        for j in range(N):
            k_xstar_x[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x[i] - x[j]) ** 2))

    k_xstar_xstar = np.zeros((len(x_star), len(x_star)))  # K(X_star, X_star)
    for i in range(len(x_star)):
        for j in range(len(x_star)):
            k_xstar_xstar[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x[i] - x[j]) ** 2))

    f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y_gp))
    cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)

    return f_star, cov_f_star

def posteriorGP_NonGaussianLik(prior_parameters, mcmc_parameters, mean_vector_mcmc, N, x, x_star, y_gp):
    print('prior_parameters[0]', prior_parameters[0])
    print('prior_parameters[1]', prior_parameters[1])
    print('prior_parameters[2]', prior_parameters[2])

    k_xx = rbf_kernel(N, x, prior_parameters[0], prior_parameters[1]) + np.eye(N) * prior_parameters[2]
    k_x_xstar = rbf_kernel(N, x, prior_parameters[0], prior_parameters[1], x_star)
    k_xstar_x = rbf_kernel(N, x_star, prior_parameters[0], prior_parameters[1], x)
    k_xstar_xstar = rbf_kernel(N, x_star, prior_parameters[0], prior_parameters[1], x_star)

    posterior_covariance = rbf_kernel(N, x, mcmc_parameters[0], mcmc_parameters[1]) + np.eye(N) * mcmc_parameters[2]

    f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), mean_vector_mcmc))
    cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar) + np.dot(np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), posterior_covariance),np.dot(np.linalg.inv(k_xx), k_x_xstar))

    return f_star, cov_f_star



def posteriorGP_mean_function_spline(tau, alpha0, N, x, x_star, y_gp):
    k_xx = thin_plate_spline_kernel(x, tau) + np.eye(N) * 0.01
    k_x_xstar = thin_plate_spline_kernel(x, tau, x_star = x_star)
    k_xstar_x = thin_plate_spline_kernel(x_star, tau, x_star = x)
    k_xstar_xstar = thin_plate_spline_kernel(x_star, tau, x_star = x_star)



    f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), (np.repeat(alpha0, len(x)))))
    cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)

    f_star = np.repeat(alpha0, len(x_star))
    cov_f_star = k_xstar_xstar


    # print('fstar', f_star)

    # plt.imshow(cov_f_star)
    # plt.show()

    # plt.imshow(k_xstar_xstar)
    # plt.show()
    # plt.imshow(k_xstar_x)
    # plt.show()
    # plt.imshow(k_xx)
    # plt.show()
    # plt.imshow(k_x_xstar)
    # plt.show()


    return f_star, cov_f_star


