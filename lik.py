import numpy as np
import scipy
from kernels import *
import scipy.stats as sp

def Gaussian_neg_loglikelihood(hyperparameters_log, N, y_gp, x, Sigma, mean_vector):
    l_scale = np.exp(hyperparameters_log[0])
    var = np.exp(hyperparameters_log[1])
    noise_var = np.exp(hyperparameters_log[2])

    # Update Covariance Matrix
    Sigma = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Sigma[i, j] = var*np.exp(-1/(2*l_scale**2)*((x[i]-x[j])**2))
    Sigma += np.eye(N) * noise_var
    # This is computing the negative log likelihood for a Gaussian lik
    neg_log_likelihood = (0.5 * np.dot(np.dot(np.transpose(y_gp - mean_vector), np.linalg.inv(Sigma)), (y_gp - mean_vector)) 
                        + 0.5 * np.log(np.linalg.det(Sigma)) + (N/2)*np.log(2*np.pi))
    return neg_log_likelihood


def Poisson_loglikelihood(XBeta, y_counts):
    # This function is evaluating the log-likelihood give the function given by XBeta
    # This is computing the negative log likelihood for a Poisson lik which is equal to 
    # sum_n (y_i*XBeta_i - exp(XBeta_i) - log(y_i!))
    neg_log_likelihood = np.sum((y_counts*XBeta) - np.exp(XBeta) - scipy.special.gammaln(y_counts + 1.))
    return neg_log_likelihood


def Poisson_neg_loglikelihood_spline(pars, N, y_gp, x):
    # This function is doing a MC evaluation of the neg log likelihood
    tau = np.exp(pars[0])
    alpha0 = pars[1]

    # Update Covariance Matrix
    Sigma = thin_plate_spline_kernel(x, tau)
    #Sigma += np.eye(N) * 0.01

    posterior_covariance = Sigma - np.dot(np.dot(Sigma, np.linalg.inv(Sigma + np.eye(N) * 10.)), Sigma)
    posterior_mean = alpha0 + np.dot(Sigma, np.dot(np.linalg.inv(Sigma + np.eye(N) * 10.), (y_gp - alpha0)))

    # Sample froma multivariate normal - Here I am sampling from the prior distribution
    sample = np.random.multivariate_normal(posterior_mean, posterior_covariance, size = 1)
    
    #print('y_gp',y_gp.shape)
    # This is computing the negative log likelihood for a Gaussian lik
    neg_log_likelihood = (np.mean(np.sum(-(y_gp*sample) + np.exp(sample) + scipy.special.gammaln(y_gp + 1.), axis = 1)) 
                        - sp.multivariate_normal.logpdf(sample, np.repeat(1,N), thin_plate_spline_kernel(x, 1.)))

    return neg_log_likelihood





