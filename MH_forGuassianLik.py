# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as sp
import numpy as np
from scipy.optimize import minimize
from matplotlib import cm
from scipy.stats.kde import gaussian_kde
from lik import *
from proposal_dist import *
from GP_stuff import *
from performances import * 
from plot_MCMC import *

# set seed
# set seed & global MCMC Parameters
n_iterations = 1500
np.random.seed(123)
# Burn in
burn_in_percentage = .4
burn_in = int(burn_in_percentage * n_iterations)

#Number of generations
N = 10
x_min = 0
x_max = 5
x = np.linspace(x_min, x_max, N)

# Kernel Hyperparameters
# gt means GT
l_scale_gt = .2
var_gt = .2
noise_var_gt = 0.1
true_value_pars = np.array((l_scale_gt, var_gt, noise_var_gt))

# Generate mean & covariance of GP
# This is computing the RBF kernel
Sigma = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        Sigma[i, j] = var_gt * np.exp(-1 / (2 * l_scale_gt ** 2) * ((x[i] - x[j]) ** 2))

# Sample from GP and generate the data according to Gaussian lik
y_gp_noiseless = sp.multivariate_normal.rvs(mean = 2*np.ones(N), cov=Sigma)
y_gp = y_gp_noiseless + np.sqrt(noise_var_gt)*np.random.randn(N)

# #Plot ground truth data
plt.plot(x,y_gp_noiseless)
plt.plot(x,y_gp,'.r')
plt.title("Data")
plt.show()



l_scale_i = np.log(2)
var_i = np.log(3)
noise_var_i = np.log(4)
mean_vector_i = np.repeat(1, N)
hyperparameters_log = np.array([np.log(l_scale_i), np.log(var_i), np.log(noise_var_i)])


#Optimise Likelihood 
# Minimize the NEGATIVE log-likelihood
result_likelihood = minimize(Gaussian_neg_loglikelihood, x0=hyperparameters_log, method='L-BFGS-B', options={'maxiter':1500}, args = (N, y_gp, x, Sigma, mean_vector_i))


# Optimisation
l_scale_opt = np.exp(result_likelihood.x[0])
var_opt = np.exp(result_likelihood.x[1])
noise_var_opt = np.exp(result_likelihood.x[2])

print('These are the results from optimisation', result_likelihood)



# Arrays of Samples & Convergence Criteria
var_samples =[]
l_scale_samples = []
noise_var_samples = []
mean_vector_samples = []
samples = []
acceptance = []
accepted_steps = 0

# Initialise Hyperparameters
log_l_scale_c = np.log(4)
log_var_c = np.log(3)
log_noise_var_c = np.log(2)
mean_vector_c = np.repeat(1, N)


# Metropolis Algorithm
# c stands for current, p stands for proposed
while len(l_scale_samples) < n_iterations:
    if len(l_scale_samples) % 1 == 0:
        print('MCMC Iterations: \t'+str(len(l_scale_samples))+'\t / \t'+str(n_iterations))

    # Define Nu_proposal hyperparameters
    log_l_scale_p = nu_p_lscale(log_l_scale_c)
    log_var_p = nu_p_var(log_var_c)
    log_noise_var_p = nu_p_var(log_noise_var_c)
    mean_vector_p = nu_p_mean_vector(mean_vector_c, N)

    # Log likelihoods for numeric stability
    # I am using the log thus the ratio p(proposed)/p(current) is a difference. Because I am considering the negative for both num and den
    # the ratio does not change
    # I compute the ratio in log terms and then I convert it back by taking the exp
    num = - Gaussian_neg_loglikelihood([log_l_scale_p, log_var_p, log_noise_var_p], N, y_gp, x, Sigma, mean_vector_p) 
    denom = - Gaussian_neg_loglikelihood([log_l_scale_c, log_var_c, log_noise_var_c], N, y_gp, x, Sigma, mean_vector_c)
    logR = num - denom

    R = np.exp(logR)
    # Acceptance Criteria
    # I am using 0 cause I am doing everything in log terms
    log_alpha = min(0, logR)


    acceptance.append(np.exp(log_alpha))

    Z = sp.uniform.rvs(0, 1)

    ## Accepted 
    if np.exp(log_alpha) > Z:
        accepted_steps = accepted_steps + 1
        log_l_scale_c = log_l_scale_p
        log_var_c = log_var_p
        log_noise_var_c = log_noise_var_p
        mean_vector_c = mean_vector_p

    ## Rejected 
    l_scale_samples.append(np.exp(log_l_scale_c))
    var_samples.append(np.exp(log_var_c))
    noise_var_samples.append(np.exp(log_noise_var_c))
    mean_vector_samples.append(mean_vector_c)

acceptance_ratio = (accepted_steps*1.) / (n_iterations*1.)
print('Acceptance Ratio: '+str(acceptance_ratio))


# perf_optimisation = performance_optimisation(result_likelihood, true_value_pars)
# perf_MCMC = performance_MCMC(l_scale_samples, var_samples, noise_var_samples, burn_in)
# print("Optimisation RMSE", perf_optimisation)
# print("MCMC RMSE", perf_MCMC)


l_scale_mcmc = np.mean(np.array(l_scale_samples[burn_in:]))
var_mcmc = np.mean(np.array(var_samples[burn_in:]))
noise_var_mcmc = np.mean(np.array(noise_var_samples[burn_in:]))
mean_vector_mcmc = np.mean(np.array(mean_vector_samples)[burn_in:,:], axis = 0)


mcmc_parameters = np.array((l_scale_mcmc, var_mcmc, noise_var_mcmc))
gt_parameters = np.array((l_scale_gt, var_gt, noise_var_gt))
opt_parameters = np.array((l_scale_opt, var_opt, noise_var_opt))
print('These are the results from MCMC', mcmc_parameters)

np.save('l_scale_samples', l_scale_samples)
np.save('var_samples', var_samples)
np.save('noise_var_samples', noise_var_samples)
np.save('mean_vector_mcmc', mean_vector_mcmc)


samples_vector = [np.array(l_scale_samples[burn_in:]), np.array(var_samples[burn_in:]), np.array(noise_var_samples[burn_in:])]
trace_plots_Gaussian(samples_vector)


plot_prediction_GaussianLik(gt_parameters, mcmc_parameters, opt_parameters, y_gp_noiseless, y_gp, N, x)


mean_prior_parameters = [1., 1., 1.]
mean_mcmc_parameters = [l_scale_mcmc, var_mcmc, noise_var_mcmc]
plot_prediction_NonGaussianLik(mean_prior_parameters, mean_mcmc_parameters, mean_vector_mcmc, y_gp_noiseless, y_gp, N, x)

