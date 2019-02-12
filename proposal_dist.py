import numpy as np
import scipy.stats as sp

# All these functions are adding noise to the parameter value
def nu_p_lscale(lscale):
    noise = sp.norm.rvs(0, 0.1)
    rvs = lscale + noise
    return rvs

def nu_p_var(var):
    noise = sp.norm.rvs(0, 0.1)
    rvs = var + noise
    return rvs

def nu_p_noise_var(noise_var):
    noise = sp.norm.rvs(0, 0.1)
    rvs = noise_var + noise
    return rvs

def nu_p_mean_vector(mean_vector, N):
    noise = np.random.multivariate_normal(np.zeros(N), np.diag(0.1*np.ones(N)))
    rvs = mean_vector + noise
    return rvs


def q_tau(tau):
    noise = sp.norm.rvs(0, 0.1)
    rvs = tau + noise
    return rvs


def q_alpha0(alpha0):
    noise = sp.norm.rvs(0, 0.1)
    rvs = alpha0 + noise
    return rvs
