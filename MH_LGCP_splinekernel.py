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
from kernels import *

# set seed & global MCMC Parameters
n_iterations = 1000
np.random.seed(123)
# Burn in
burn_in_percentage = .4
burn_in = int(burn_in_percentage * n_iterations)

# Number of generations
N = 50
x_min = 1.
x_max = 5.
x = np.linspace(x_min, x_max, N)

# Kernel Hyperparameters
# gt means Ground Truth
tau_gt = 1.
alpha0_gt = 1.
true_value_pars = np.array((tau_gt, alpha0_gt))

# Generate mean & covariance of GP
# This is computing the RBF kernel
Sigma = thin_plate_spline_kernel(x, tau_gt)
print('sigma', Sigma.shape)

# eigen_vals, eigen_vec = np.linalg.eig(Sigma)
# for i in range(10):
#     plt.plot(eigen_vec[:,i])
# plt.show()

# for i in range(10):
#     plt.plot(Sigma[:,i])
# plt.show()

# Sample from GP and generate the data 
log_lambda = sp.multivariate_normal.rvs(mean = np.repeat(alpha0_gt, N), cov=Sigma)
y_gp = np.random.poisson(np.exp(log_lambda))


## Plot ground truth data
plt.plot(x,np.exp(log_lambda),'blue')
plt.plot(x,y_gp,'.r')
plt.show()



# Arrays of Samples & Convergence Criteria
tau_samples =[]
alpha0_samples = []

samples = []
acceptance = []
accepted_steps = 0

# Initialise Hyperparameters for MCMC - These are initialised to the true values
log_tau_c = np.log(1.) # approx 0.
alpha0_c = 1. # approx 1.


# Metropolis Algorithm
# c stands for current, p stands for proposed
while len(tau_samples) < n_iterations:
    if len(tau_samples) % 1 == 0:
        print('MCMC Iterations: \t'+str(len(tau_samples))+'\t / \t'+str(n_iterations))

    # Propose a new value 
    # Propose a new value for log_tau
    log_tau_p = q_tau(log_tau_c)
    alpha0_p = q_alpha0(alpha0_c)


    # Log likelihoods for numeric stability
    # I am using the log thus the ratio p(proposed)/p(current) is a difference. Because I am considering the negative for both num and den
    # the ratio does not change
    # I compute the ratio in log terms and then I convert it back by taking the exp
    # We define the prior for the log parameters to be log normals
    pars_p = np.concatenate((np.reshape(log_tau_p, (1,)), np.reshape(alpha0_p, (1,))), axis = 0)
    pars_c = np.concatenate((np.reshape(log_tau_c, (1,)), np.reshape(alpha0_c, (1,))), axis = 0)

    num = (- Poisson_neg_loglikelihood_spline(pars_p, N, y_gp, x) 
            + sp.lognorm.logpdf(log_tau_p, s = 0.1, scale = 1.) 
            + sp.norm.logpdf(alpha0_p, loc = 1., scale = 0.5))
                                                          
    denom = (- Poisson_neg_loglikelihood_spline(pars_c, N, y_gp, x) 
            + sp.lognorm.logpdf(log_tau_c, s = 0.1, scale = 1) 
            + sp.norm.logpdf(alpha0_c, loc = 1., scale = 0.5))

    # num = (- Poisson_neg_loglikelihood_spline(pars_p, N, y_gp, x) )
                                                          
    # denom = (- Poisson_neg_loglikelihood_spline(pars_c, N, y_gp, x) )

    
    logR = num-denom



    R = np.exp(logR)
    # Acceptance Criteria
    # I am using 0 cause I am doing everything in log terms
    log_alpha = min(0, logR)


    acceptance.append(np.exp(log_alpha))

    Z = sp.uniform.rvs(0, 1)

    ## Accepted 
    if np.exp(log_alpha) > Z:
        print('I have accepted')
        accepted_steps = accepted_steps + 1
        print('accepted_steps', accepted_steps)
        log_tau_c = log_tau_p
        alpha0_c = alpha0_p

    ## Rejected 
    tau_samples.append(np.exp(log_tau_c))
    alpha0_samples.append(alpha0_c)


acceptance_ratio = (accepted_steps*1.) / (n_iterations*1.)
print(acceptance_ratio)
print('Acceptance Ratio: '+str(acceptance_ratio))


tau_mcmc = np.mean(np.array(tau_samples[burn_in:]))
alpha0_mcmc = np.mean(np.array(alpha0_samples[burn_in:]))


# Produce trace plots for mcmc chains
samples_vector = [np.array(tau_samples[burn_in:]), np.array(alpha0_samples[burn_in:])]
trace_plots_splines(samples_vector)


# Produce prediction plots
mcmc_parameters = np.concatenate((np.reshape(tau_mcmc, (1,)), np.reshape(alpha0_mcmc, (1,))), axis = 0)
plot_spline_prediction_LGCP(mcmc_parameters, y_gp, N, x, log_lambda)



#plot_parameter_posterior(mcmc_parameters_samples, gt_parameters, opt_parameters, burn_in)

