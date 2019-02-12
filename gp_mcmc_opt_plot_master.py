# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as sp
import numpy as np
from scipy.optimize import minimize
from matplotlib import cm
from scipy.stats.kde import gaussian_kde
from Utilities import gen_ar2, gen_ar1, analytic_spectrum_ar1, analytic_spectrum_ar2, posterior_predictive_hyperparameters

# set seed
# set seed & global MCMC Parameters
n_iterations = 2000
np.random.seed(123)
# Burn in
burn_in_percentage = .4
burn_in = int(burn_in_percentage * n_iterations)

#Number of generations
N_gen = 100
x_gen_min = 0
x_gen_max = 5
x_gen = np.linspace(x_gen_min, x_gen_max, N_gen)

# Kernel Hyperparameters
l_scale_gt = .2
var_gt = .2
noise_var_gt = 0.1

# Generate mean & covariance of GP
# This is computing the RBF kernel
omega = np.zeros((N_gen, N_gen))
for i in range(N_gen):
    for j in range(N_gen):
        omega[i, j] = var_gt * np.exp(-1 / (2 * l_scale_gt ** 2) * ((x_gen[i] - x_gen[j]) ** 2))

# Sample from GP
y_gp_noiseless = sp.multivariate_normal.rvs(mean = np.zeros(N_gen), cov=omega)
y_gp = y_gp_noiseless + np.sqrt(noise_var_gt)*np.random.randn(N_gen)

# #Plot ground truth data
plt.plot(x_gen,y_gp_noiseless)
plt.plot(x_gen,y_gp,'.r')
plt.show()


l_scale_i = np.log(2)
var_i = np.log(3)
noise_var_i = np.log(4)
hyperparameters_log = np.array([np.log(l_scale_i), np.log(var_i), np.log(noise_var_i)])

def neg_loglikelihood(log_hyperparameters):
    l_scale = np.exp(log_hyperparameters[0])
    var = np.exp(log_hyperparameters[1])
    noise_var = np.exp(log_hyperparameters[2])

    # Update Covariance Matrix
    omega = np.zeros((N_gen, N_gen))
    for i in range(N_gen):
        for j in range(N_gen):
            omega[i, j] = var*np.exp(-1/(2*l_scale**2)*((x_gen[i]-x_gen[j])**2))
    omega += np.eye(N_gen) * noise_var
    # This is computing the negative log likelihood for a Gaussian lik
    neg_log_likelihood = 0.5 * np.dot(np.dot(np.transpose(y_gp), np.linalg.inv(omega)), y_gp) + 0.5 * np.log(np.linalg.det(omega)) + (N_gen/2)*np.log(2*np.pi)
    return neg_log_likelihood

#Optimise Likelihood 
# Minimize the NEGATIVE log-likelihood
result_likelihood = minimize(neg_loglikelihood, x0=hyperparameters_log, method='L-BFGS-B', options={'maxiter':1500})

# Optimisation
l_scale_opt = np.exp(result_likelihood.x[0])
var_opt = np.exp(result_likelihood.x[1])
noise_var_opt = np.exp(result_likelihood.x[2])

print(result_likelihood)

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

def rmse(samples):
    mode = sp.mode(samples)
    rmse = np.sqrt(np.sum((i - mode[0])**2) / len(samples))
    return rmse




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
def covariance_matrix(l_scale, var, noise_var):
    omega = np.zeros((N_gen, N_gen))
    for i in range(N_gen):
        for j in range(N_gen):
            omega[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x_gen[i] - x_gen[j]) ** 2))

    # Generate Covariance matrix - predictive distribution
    k_xx = omega + np.eye(N_gen) * noise_var  # K(X,X)

    k_x_xstar = np.zeros((N_gen, len(x_star)))  # K(X, X_star)
    for i in range(N_gen):
        for j in range(len(x_star)):
            k_x_xstar[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x_gen[i] - x_gen[j]) ** 2))

    k_xstar_x = np.zeros((len(x_star), N_gen))  # K(X_star, X)
    for i in range(len(x_star)):
        for j in range(N_gen):
            k_xstar_x[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x_gen[i] - x_gen[j]) ** 2))

    k_xstar_xstar = np.zeros((len(x_star), len(x_star)))  # K(X_star, X_star)
    for i in range(len(x_star)):
        for j in range(len(x_star)):
            k_xstar_xstar[i, j] = var * np.exp(-1 / (2 * l_scale ** 2) * ((x_gen[i] - x_gen[j]) ** 2))

    f_star = np.dot(k_xstar_x, np.dot(np.linalg.inv(k_xx), y_gp))
    cov_f_star = k_xstar_xstar - np.dot(np.dot(k_xstar_x, np.linalg.inv(k_xx)), k_x_xstar)

    return f_star, cov_f_star


#Debug Plot of Log Likelihood
"""
a_min = -10 #np.max([np.min([var_gt,np.exp(result_likelihood.x[0])])-.1,0.01])#0.01
a_max = 10 #np.max([var_gt,np.exp(result_likelihood.x[0])])+.1
a_N = 20
b_min = -10 # np.max([np.min([l_scale_gt,np.exp(result_likelihood.x[1])])-.1,0.01])#0.01
b_max = 10 # np.max([l_scale_gt,np.exp(result_likelihood.x[1])])+.1
b_N = 20

x = np.linspace(a_min,a_max,a_N)
y = np.linspace(b_min,b_max,b_N)
X, Y = np.meshgrid(x,y)
Z = np.zeros((a_N,b_N))

a_idx = 0
for a in x:
    b_idx = 0
    for b in y:
        print(100*a_idx/len(x))
        Z[b_idx,a_idx] = neg_loglikelihood([np.log(a),np.log(b)])
        b_idx += 1
    a_idx += 1
"""

# Initialise Hyperparameters
log_l_scale_c = np.log(4)
log_var_c = np.log(3)
log_noise_var_c = np.log(2)

# Arrays of Samples & Convergence Criteria
var_samples =[]
l_scale_samples = []
noise_var_samples = []
samples = []
acceptance = []
accepted_steps = 0

# Initialise Hyperparameters
log_l_scale_c = np.log(4)
log_var_c = np.log(3)
log_noise_var_c = np.log(.5)

# Arrays of Samples & Convergence Criteria
l_scale_samples = []
var_samples =[]
noise_var_samples = []
samples = []
acceptance = []
accepted_steps = 0

# Metropolis Algorithm
# c stands for current, p stands for proposed
while len(l_scale_samples) < n_iterations:
    if len(l_scale_samples) % 1 == 0:
        print('MCMC Iterations: \t'+str(len(l_scale_samples))+'\t / \t'+str(n_iterations))

    # Define Nu_proposal hyperparameters
    log_l_scale_p = nu_p_lscale(log_l_scale_c)
    log_var_p = nu_p_var(log_var_c)
    log_noise_var_p = nu_p_var(log_noise_var_c)

    # Log likelihoods for numeric stability
    # I am using the log thus the ratio p(proposed)/p(current) is a difference. Because I am considering the negative for both num and den
    # the ratio does not change
    # I compute the ratio in log terms and then I convert it back by taking the exp
    num = -neg_loglikelihood([log_l_scale_p, log_var_p, log_noise_var_p])# + logpdf_lscale(l_scale_p) + logpdf_var(var_p)
    denom = -neg_loglikelihood([log_l_scale_c, log_var_c, log_noise_var_c])# + logpdf_lscale(l_scale_c) + logpdf_var(var_c)
    logR = num-denom

    R = np.exp(logR)

    # Acceptance Criteria
    log_alpha = min(0, logR)

    acceptance.append(np.exp(log_alpha))

    Z = sp.uniform.rvs(0, 1)

    if np.exp(log_alpha) > Z:
        accepted_steps = accepted_steps + 1
        log_l_scale_c = log_l_scale_p
        log_var_c = log_var_p
        log_noise_var_c = log_noise_var_p
    l_scale_samples.append(np.exp(log_l_scale_c))
    var_samples.append(np.exp(log_var_c))
    noise_var_samples.append(np.exp(log_noise_var_c))

acceptance_ratio = accepted_steps / n_iterations
print('Acceptance Ratio: '+str(acceptance_ratio))

plt.plot(l_scale_samples)
plt.title("Traceplot: Lengthscale")
plt.show()

plt.plot(var_samples)
plt.title("Traceplot: Variance")
plt.show()

plt.plot(noise_var_samples)
plt.title("Traceplot: Noise Variance")
plt.show()

# Kernel Density Estimation plots
l_scale_post_burn = l_scale_samples[burn_in:]
var_post_burn = var_samples[burn_in:]# Look at all samples post burn-in
noise_var_post_burn = noise_var_samples[burn_in:]
# Make arrays
l_scale_post_burn = np.array(l_scale_post_burn)
var_post_burn = np.array(var_post_burn)
noise_var_post_burn = np.array(noise_var_post_burn)

# Lengthscale, post burn-in
kde = gaussian_kde(l_scale_post_burn)
dist_space = np.linspace(np.min(l_scale_post_burn), np.max(l_scale_post_burn), 100)
f, axarr = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
axarr[0].plot(dist_space, kde(dist_space))
axarr[1].plot(l_scale_post_burn)
plt.title("KDE & Traceplot: Lengthscale")
plt.show()

# Variance, post burn-in
kde = gaussian_kde(var_post_burn)
dist_space = np.linspace(np.min(var_post_burn), np.max(var_post_burn), 100)
f, axarr = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
axarr[0].plot(dist_space, kde(dist_space))
axarr[1].plot(var_post_burn)
plt.title("KDE & Traceplot: Variance")
plt.show()

# Noise Variance, post burn-in
kde = gaussian_kde(noise_var_post_burn)
dist_space = np.linspace(np.min(noise_var_post_burn), np.max(noise_var_post_burn), 100)
f, axarr = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
axarr[0].plot(dist_space, kde(dist_space))
axarr[1].plot(noise_var_post_burn)
plt.title("KDE & Traceplot: Noise Variance")
plt.show()

# RMSE Calculations
# Optimisation RMSE
l_scale_rmse_opt = np.exp(result_likelihood.x[0]) - l_scale_gt
var_rmse_opt = np.exp(result_likelihood.x[1]) - var_gt
noise_var_rmse_opt = np.exp(result_likelihood.x[2]) - noise_var_gt
optimisation_rmse = np.array([l_scale_rmse_opt, var_rmse_opt, noise_var_rmse_opt])
print("Optimisation RMSE", optimisation_rmse)

# MCMC RMSE
l_scale_rmse_mcmc = rmse(l_scale_samples[burn_in:])
var_rmse_mcmc = rmse(var_samples[burn_in:])
noise_var_rmse_mcmc = rmse(noise_var_samples[burn_in:])
mcmc_rmse = np.array([l_scale_rmse_mcmc, var_rmse_mcmc, noise_var_rmse_mcmc])
print("MCMC RMSE", mcmc_rmse)



########## I am here




# Plot Predictive Posterior Distribution
x_star = np.linspace(0, 5, 100)
f_star, cov_f_star = covariance_matrix(l_scale_gt, var_gt, noise_var_gt)
gp_predictive = sp.multivariate_normal.rvs(f_star, cov_f_star)

# MCMC recovered Gaussian process
# Kernel Hyperparameters
l_scale_mcmc = np.mean(l_scale_post_burn)
var_mcmc = np.mean(var_post_burn)
noise_var_mcmc = np.mean(noise_var_post_burn)

# MCMC & Optimisation Hyperparameters generated covariance matrix
f_star_mcmc, cov_f_star_mcmc = covariance_matrix(l_scale_mcmc, var_mcmc, noise_var_mcmc)
cov_f_star_mcmc_diag = np.diag(cov_f_star_mcmc)

f_star_opt, cov_f_star_opt = covariance_matrix(l_scale_opt, var_opt, noise_var_opt)
cov_f_star_opt_diag = np.diag(cov_f_star_opt)

# Generate mean & covariance of GP
omega_mcmc = np.zeros((N_gen, N_gen))
for i in range(N_gen):
    for j in range(N_gen):
        omega[i, j] = var_mcmc * np.exp(-1 / (2 * l_scale_mcmc ** 2) * ((x_gen[i] - x_gen[j]) ** 2))
# omega_mcmc = np.identity(N_gen)*noise_var_mcmc
gp_mcmc = sp.multivariate_normal.rvs(np.zeros(N_gen), cov=omega_mcmc)

# Plot Recovered Gaussian Process with Predictive Posterior distribution and learned hyperparameters via MCMC & Optimisation
cov_f_star_diag = np.diag(cov_f_star)
upper_bound_mcmc = f_star + (2*np.sqrt(cov_f_star_mcmc_diag))
lower_bound_mcmc = f_star - (2*np.sqrt(cov_f_star_mcmc_diag))
upper_bound_opt = f_star_opt + (2*np.sqrt(cov_f_star_opt_diag))
lower_bound_opt = f_star_opt - (2*np.sqrt(cov_f_star_opt_diag))
plt.fill_between(x_star, upper_bound_mcmc, lower_bound_mcmc, facecolor = 'blue', alpha=0.3, interpolate=True) # Bounds based on MCMC
#plt.fill_between(x_star, upper_bound_opt, lower_bound_opt, facecolor = 'blue', alpha=0.3, interpolate=True) # Bounds based on Optimisation
gt, = plt.plot(x_star, y_gp_noiseless) # Ground Truth
mcmc, = plt.plot(x_star, f_star_mcmc)
opt, = plt.plot(x_star, f_star_opt)
plt.plot(x_gen,y_gp,'.r')
plt.legend([gt, mcmc, opt], ['Ground Truth', 'MCMC', 'Optimisation'])
plt.title("MCMC, Optimisation and Ground Truth Estimates with 95% CI")
plt.savefig("MCMC vs Opt vs Post Pred")
plt.show()


# Plots
N = len(l_scale_samples[burn_in:])
X = np.linspace(0, 1, N)
Y = np.linspace(-0.5, 1, N)
X, Y = np.meshgrid(X,Y)

# Mean Vector & Covariance matrix
mu = np.array([np.mean(l_scale_samples[burn_in:]), np.mean(var_samples[burn_in:])])
cov = np.reshape(np.cov(l_scale_samples[burn_in:], var_samples[burn_in:]), (2,2))

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, cov)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
#cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)
ax.set_title('Optimisation estimate vs MCMC Distribution')
ax.view_init(45, 45)

#Plot ground truth solution
a_gt = l_scale_gt
b_gt = var_gt
x_gt = [a_gt,a_gt]
y_gt = [b_gt,b_gt]
z_gt = [np.min(np.min(Z)),np.max(np.max(Z))]
ax.plot(x_gt,y_gt,z_gt,'r',linewidth=2)
#ax.plot(a_gt,b_gt,0,"x",color='r')

#Plot optimiser solution
a_opt = np.exp(result_likelihood.x[1])
b_opt = np.exp(result_likelihood.x[0])
x_opt = [a_opt,a_opt]
y_opt = [b_opt,b_opt]
z_opt = [np.min(np.min(Z)),np.max(np.max(Z))]
ax.plot(x_opt,y_opt,z_opt,'b',linewidth=2)
#ax.plot(a_opt,b_opt,0,"x",color='b')

plt.xlabel("lengthscale")
plt.ylabel("variance")
plt.savefig("Lengthscale vs Variance: 3d plot")
plt.show()
