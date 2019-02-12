# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as sp
import numpy as np
from scipy.optimize import minimize
from matplotlib import cm
from scipy.stats.kde import gaussian_kde
import statsmodels.api as sm
import sklearn.gaussian_process.kernels as kern
from lik import *
from proposal_dist import *
from GP_stuff import *
from performances import * 
from plot_MCMC import *
from kernels import *
from util import *


# set seed & global MCMC Parameters
n_iterations = 5000
np.random.seed(123)
# Burn in
burn_in_percentage = .4
burn_in = int(burn_in_percentage * n_iterations)

# Number of generations
N = 100
x_min = 0
x_max = 5
x = np.linspace(x_min, x_max, N)
x_star = np.linspace(0, 5, 100)

# Kernel Hyperparameters
# gt means Ground Truth
lenghtscale_gt = 1.
sigma_square_f_gt = 1.
true_value_pars = np.array((lenghtscale_gt, sigma_square_f_gt))

# Generate mean & covariance of GP
# This is computing the RBF kernel but WITHOUT the sigma^2_f
# This is the matrix Omega in Sally's notation
# Omega = rbf_kernel(x, 1., lenghtscale_gt)
Omega = kern.RBF(lenghtscale_gt)(x[:,np.newaxis])
D, Q = np.linalg.eig(Omega) 
D = np.real(D)
Q = np.real(Q)
sqrtD = np.sqrt(np.clip(D,0,np.inf))

for i in xrange(10):
    plt.plot(Q[:,i])
plt.show()

num_eigen = 10
X_subset, diag_subset = eigen_decomposition(Omega, num_eigen)

np.save('X_subset', X_subset)
np.save('diag_subset', diag_subset)

#X_matrix = np.dot(Q, np.sqrt(np.diag(np.clip(D,0,np.inf))))
X_matrix = np.dot(Q, np.diag(sqrtD))
Beta_process = sp.multivariate_normal.rvs(mean = np.zeros(N), cov=sigma_square_f_gt*np.eye(N))


f_process = np.dot(X_matrix, Beta_process)
intensity_function = np.exp(f_process)

# print('X_matrix', X_matrix)
# print('X_matrix', Q)
# print('np.sqrt(np.diag(np.clip(D,0,np.inf)))', np.sqrt(np.diag(np.clip(D,0,np.inf))))
# print('Beta_process', Beta_process)



# Define intensity and generate the data 
intensity_function = np.exp(f_process)
y_counts = np.random.poisson(intensity_function)


## Plot ground truth data
plt.plot(x,intensity_function,'blue')
plt.plot(x,y_counts,'.r')
plt.show()

##### Evaluate likelihood
# value = Poisson_neg_loglikelihood(f_process, y_counts)
# print('value', value)


## Optimise Likelihood 
# l_scale_i = np.log(2.)
# var_i = np.log(3.)
# mean_vector_i = np.zeros(N)
# optimisation_pars = np.concatenate((np.reshape(np.log(l_scale_i), (1,)), np.reshape(np.log(var_i), (1,)), mean_vector_i), axis = 0)
## Minimize the NEGATIVE log-likelihood
#result_likelihood = minimize(Poisson_neg_loglikelihood, x0=optimisation_pars, method='L-BFGS-B', options={'maxiter':1}, args = (N, y_gp, x))

## Save results from optimisation
# l_scale_opt = np.exp(result_likelihood.x[0])
# var_opt = np.exp(result_likelihood.x[1])
# mean_opt = result_likelihood.x[2:]
l_scale_opt = np.exp(np.log(2.))
var_opt = np.exp(np.log(3.))
mean_opt = np.zeros(N)


# Arrays of Samples & Convergence Criteria
sigma_f_square_samples =[]
lenghtscale_samples = []
Beta_process_samples = []
pred_mean_samples = []
pred_covariance_samples = []

samples = []
acceptance = []
accepted_steps = 0

### Initialise Hyperparameters for MCMC - values very close to the gt values
### In order to initialise the Beta process compute the MLE coefficients of a Poisson regression on the data
if num_eigen < N:
    covariates = X_subset
    response = y_counts
    poisson_model = sm.GLM(response, covariates, family=sm.families.Poisson())
    results = poisson_model.fit()
    beta_estimate = results.params
    var_beta_estimate = np.linalg.inv(- poisson_model.information(results.params))
    ###This is the proposal distribution for Beta process
    ### q(Beta) hyperparameter - If X_matrix does not change in the algorithm then this parameters are fixed
    Beta_process_c = beta_estimate
    beta_hat_c = beta_estimate
    cov_beta_hat_c = var_beta_estimate
    beta_hat_p = beta_estimate
    cov_beta_hat_p = var_beta_estimate   
else:
    ### This is the proposal distribution for Beta process
    ### q(Beta) hyperparameter - If X_matrix does not change in the algorithm then this parameters are fixed
    Beta_process_c = np.zeros(10)
    beta_hat_c = np.zeros(N)
    cov_beta_hat_c = np.eye(N)
    beta_hat_p = np.zeros(N)
    cov_beta_hat_p = np.eye(N)

print('beta_estimate', beta_estimate)
print('var_beta_estimate', var_beta_estimate)



sigma_square_f_c = 1. # approx 1.


### Prior hyperparameter
b_inverse_gamma = 1.



### MCMC Algorithm
### c stands for current, p stands for proposed
while len(Beta_process_samples) < n_iterations:
    if len(Beta_process_samples) % 1 == 0:
        print('MCMC Iterations: \t'+str(len(Beta_process_samples))+'\t / \t'+str(n_iterations))

    ### Draw sigma_square_f from in Inverse Gamma distribution - Gibbs sample, draw from the full conditional
    ### a is the shape par (alpha), scale is the b parameter (beta)
    ### The support of the inverse gamma is (0, infinity) thus we can operate on the parameter itself instead of the log transform
    ### The proposed value is always accepted cause this is a Gibbs step
    sigma_square_f_p = sp.invgamma.rvs(a = num_eigen/2., scale = b_inverse_gamma + np.dot(Beta_process_c, np.transpose(Beta_process_c))/2.)

    ### Draw lenghscale - MH step
    ### We can avoid this step here if we fix the lenghscale. In this case there is no need to recompute the matrix X at each time
    ### We keep it fixed to the true value

    ### Draw Beta process - MH step with no symmetric distributions
    ### The proposal distribution of the Beta process is N(beta_hat, cov_beta_hat) where the beta_hat are the coefficient of a poisson regression
    ### Need to get other parameters for the proposal, currently the proposal is N(0,I)
    # beta_hat_p = np.zeros(N)
    # cov_beta_hat_p = np.eye(N)
    Beta_process_p = np.random.multivariate_normal(beta_hat_p, cov_beta_hat_p)


    # Log likelihoods for numeric stability
    # I am using the log thus the ratio p(proposed)/p(current) is a difference. Because I am considering the negative for both num and den
    # the ratio does not change
    # I compute the ratio in log terms and then I convert it back by taking the exp

    # prior_beta_process_p = sp.multivariate_normal.pdf(Beta_process_p, np.zeros(N), np.eye(N)*sigma_square_f_p)
    # prior_beta_process_c = sp.multivariate_normal.pdf(Beta_process_c, np.zeros(N), np.eye(N)*sigma_square_f_c)
    prior_beta_process_p = sp.multivariate_normal.pdf(Beta_process_p, np.zeros(num_eigen), np.eye(num_eigen)*sigma_square_f_p)
    prior_beta_process_c = sp.multivariate_normal.pdf(Beta_process_c, np.zeros(num_eigen), np.eye(num_eigen)*sigma_square_f_c)
    q_beta_process_c = sp.multivariate_normal.pdf(Beta_process_c, beta_hat_c, sigma_square_f_c*cov_beta_hat_c)
    q_beta_process_p = sp.multivariate_normal.pdf(Beta_process_p, beta_hat_p, sigma_square_f_p*cov_beta_hat_p)
    # Xbeta_p = np.dot(X_matrix, Beta_process_p)
    # Xbeta_c = np.dot(X_matrix, Beta_process_c)
    Xbeta_p = np.dot(X_subset, Beta_process_p)
    Xbeta_c = np.dot(X_subset, Beta_process_c)


    num = Poisson_loglikelihood(Xbeta_p, y_counts) + prior_beta_process_p + q_beta_process_c
    denom = Poisson_loglikelihood(Xbeta_c, y_counts) + prior_beta_process_c + q_beta_process_p
    logR = num - denom


    # Acceptance Criteria
    # I am using 0 cause I am doing everything in log terms
    log_alpha = min(0, logR)


    acceptance.append(np.exp(log_alpha))

    Z = sp.uniform.rvs(0, 1)

    ## Accepted 
    if np.exp(log_alpha) > Z:
        print('I have accepted')
        accepted_steps = accepted_steps + 1
        Beta_process_c = Beta_process_p
        sigma_square_f_c = sigma_square_f_p


    ### At each step I do prediction 
    ### Given the current accepted function I compute the prediction as K_xstar_xK_xx-1f, K_xstar_xstar - k_xstarx Kxx-1K_xstar_x 
    #pred_mean, pred_covariance = predictiveGP(np.dot(X_matrix, Beta_process_c), sigma_square_f_c, Omega, N, x, x_star, lenghtscale_gt)

    pred_mean, pred_covariance = predictiveGP(np.dot(X_subset, Beta_process_c), sigma_square_f_c, Omega, N, x, x_star, lenghtscale_gt)

    ### Store object of the MCMC
    sigma_f_square_samples.append(sigma_square_f_c)
    Beta_process_samples.append(Beta_process_c)
    pred_mean_samples.append(pred_mean)
    pred_covariance_samples.append(pred_covariance)


acceptance_ratio = (accepted_steps*1.) / (n_iterations*1.)
print(acceptance_ratio)
print('Acceptance Ratio: ' + str(acceptance_ratio))

# #perf_optimisation = performance_optimisation(result_likelihood, true_value_pars)
# perf_MCMC = performance_MCMC_Poisson(l_scale_samples, var_samples, burn_in)
# #print("Optimisation RMSE", perf_optimisation)
# print("MCMC RMSE", perf_MCMC)


sigma_f_square_mcmc = np.mean(np.array(sigma_f_square_samples[burn_in:]))

Beta_process_mcmc = np.mean(np.array(Beta_process_samples)[burn_in:, :], axis = 0)
### What I get from the prediciton at each steps are samples from the predictive distribution
pred_intensity_function_mcmc = np.mean(np.exp(np.array(pred_mean_samples)[burn_in:, :]), axis = 0)
#pred_intensity_function_mcmc = np.exp(np.array(pred_mean_samples)[burn_in:, :])[1499,:]


# Produce trace plots for mcmc chains
samples_vector = [np.array(sigma_f_square_samples[burn_in:]), np.array(Beta_process_samples)[burn_in:, :]]
trace_plots_Poisson(samples_vector)


# Produce prediction plots
plot_prediction_LGCP(pred_intensity_function_mcmc, y_counts, x, x_star, intensity_function)


