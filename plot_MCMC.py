import numpy as np
import matplotlib.pyplot as plt
from GP_stuff import *
import scipy.stats as sp

def MCMC_plots(l_chain, var_chain, noise_chain, burn_in):

	### Plot the results
	plt.plot(l_chain)
	plt.title("Traceplot: Lengthscale")
	plt.show()

	plt.plot(var_chain)
	plt.title("Traceplot: Variance")
	plt.show()

	plt.plot(noise_chain)
	plt.title("Traceplot: Noise Variance")
	plt.show()


	# Kernel Density Estimation plots
	l_scale_post_burn = np.array(l_chain[burn_in:])
	var_post_burn = np.array(var_chain[burn_in:]) # Look at all samples post burn-in
	noise_var_post_burn = np.array(noise_chain[burn_in:])

	## Given the chain post burn in these are plotting the posterior density estimation
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



def plot_prediction_GaussianLik(gt_parameters, mcmc_parameters, opt_parameters, y_gp_noiseless, y_gp, N, x):
	# Plot ground truth Posterior Distribution
	x_star = np.linspace(0, 5, 100)
	f_star_mcmc, cov_f_star_mcmc = posteriorGP_GaussianLik(mcmc_parameters[0], mcmc_parameters[1], mcmc_parameters[2], N, x, x_star, y_gp)
	cov_f_star_mcmc_diag = np.diag(cov_f_star_mcmc)

	# Plot Recovered Gaussian Process with Predictive Posterior distribution and learned hyperparameters via MCMC & Optimisation
	upper_bound_mcmc = f_star_mcmc + (2*np.sqrt(cov_f_star_mcmc_diag))
	lower_bound_mcmc = f_star_mcmc - (2*np.sqrt(cov_f_star_mcmc_diag))
	plt.fill_between(x_star, upper_bound_mcmc, lower_bound_mcmc, facecolor = 'blue', alpha=0.3, interpolate=True) # Bounds based on MCMC
	#plt.fill_between(x_star, upper_bound_opt, lower_bound_opt, facecolor = 'blue', alpha=0.3, interpolate=True) # Bounds based on Optimisation
	gt, = plt.plot(x_star, y_gp_noiseless) # Ground Truth
	mcmc, = plt.plot(x_star, f_star_mcmc)
	plt.plot(x, y_gp,'.r')
	plt.legend([gt, mcmc], ['Ground Truth', 'MCMC'])
	plt.title("MCMC pred ")
	#plt.savefig("MCMC vs Opt vs Post Pred")
	plt.show()

def plot_prediction_NonGaussianLik(prior_parameters, mcmc_parameters, mean_vector_mcmc, y_gp_noiseless, y_gp, N, x):
	# Plot ground truth Posterior Distribution
	x_star = np.linspace(0, 5, 100)



	f_star_mcmc, cov_f_star_mcmc = posteriorGP_NonGaussianLik(prior_parameters, mcmc_parameters, mean_vector_mcmc, N, x, x_star, y_gp)
	cov_f_star_mcmc_diag = np.diag(cov_f_star_mcmc)


	# Plot Recovered Gaussian Process with Predictive Posterior distribution and learned hyperparameters via MCMC & Optimisation
	upper_bound_mcmc = f_star_mcmc + (2*np.sqrt(cov_f_star_mcmc_diag))
	lower_bound_mcmc = f_star_mcmc - (2*np.sqrt(cov_f_star_mcmc_diag))
	plt.fill_between(x_star, upper_bound_mcmc, lower_bound_mcmc, facecolor = 'blue', alpha=0.3, interpolate=True) # Bounds based on MCMC
	#plt.fill_between(x_star, upper_bound_opt, lower_bound_opt, facecolor = 'blue', alpha=0.3, interpolate=True) # Bounds based on Optimisation
	gt, = plt.plot(x_star, y_gp_noiseless) # Ground Truth
	mcmc, = plt.plot(x_star, f_star_mcmc)
	plt.plot(x, y_gp,'.r')
	plt.legend([gt, mcmc], ['Ground Truth', 'MCMC'])
	plt.title("MCMC pred")
	#plt.savefig("MCMC vs Opt vs Post Pred")
	plt.show()


def plot_prediction_LGCP(pred_mean_function_mcmc, y_gp, x, x_star, intensity_function):
	# Plot ground truth Posterior Distribution
	intensity, = plt.plot(x, intensity_function) # Ground Truth
	mcmc, = plt.plot(x_star, pred_mean_function_mcmc)
	plt.plot(x, y_gp,'.r')
	plt.legend([intensity, mcmc], ['Ground Truth Intensity', 'MCMC'])
	plt.title("MCMC and Ground Truth Estimates")
	plt.savefig("MCMCPred")
	plt.show()


def plot_spline_prediction_LGCP(mcmc_parameters, y_gp, N, x, log_lambda):
	# Plot ground truth Posterior Distribution
	x_star = np.linspace(1., 5., 50)

	print('mcmc_parameters', mcmc_parameters)
	f_star_mcmc, cov_f_star_mcmc = posteriorGP_mean_function_spline(mcmc_parameters[0], mcmc_parameters[1], N, x, x_star, y_gp)
	cov_f_star_mcmc_diag = np.diag(cov_f_star_mcmc)

	# Sample fucntions from the posterior on f
	expected_samples_intensity = np.mean(np.exp(np.random.multivariate_normal(f_star_mcmc, cov_f_star_mcmc, 1000)), axis = 0)
	print('expected_samples_intensity', expected_samples_intensity)

	intensity, = plt.plot(x, np.exp(log_lambda)) # Ground Truth
	mcmc, = plt.plot(x_star, expected_samples_intensity)
	# func, = plt.plot(x_star, f_star_mcmc)
	plt.plot(x, y_gp,'.r')
	plt.legend([intensity, mcmc], ['Ground Truth Intensity', 'MCMC'])
	plt.title("MCMC and Ground Truth Estimates")
	plt.savefig("MCMC_Pred")
	plt.show()





def plot_parameter_posterior(mcmc_parameters_samples, gt_parameters, opt_parameters, burn_in):
	N = len(mcmc_parameters_samples[0][burn_in:])
	X = np.linspace(0, 1, N)
	Y = np.linspace(-0.5, 1, N)
	X, Y = np.meshgrid(X,Y)
	# Mean of the mcmc samples for each parameter
	mu = np.array([np.mean(mcmc_parameters_samples[0][burn_in:]), np.mean(mcmc_parameters_samples[1][burn_in:])])
	# Variance of the mcmc samples for each parameter
	cov = np.reshape(np.cov(mcmc_parameters_samples[0][burn_in:], mcmc_parameters_samples[1][burn_in:]), (2,2))
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
	#ax.set_title('Optimisation estimate vs MCMC Distribution')
	ax.view_init(45, 45)

	#Plot ground truth solution
	a_gt = gt_parameters[0]
	b_gt = gt_parameters[1]
	x_gt = [a_gt,a_gt]
	y_gt = [b_gt,b_gt]
	z_gt = [np.min(np.min(Z)),np.max(np.max(Z))]
	ax.plot(x_gt,y_gt,z_gt,'red',linewidth=2, label = 'Truth')
	#ax.plot(a_gt,b_gt,0,"x",color='r')

	#Plot optimiser solution
	a_opt = opt_parameters[1]
	b_opt = opt_parameters[0]
	x_opt = [a_opt,a_opt]
	y_opt = [b_opt,b_opt]
	z_opt = [np.min(np.min(Z)),np.max(np.max(Z))]
	ax.plot(x_opt,y_opt,z_opt,'blue',linewidth=2, label = 'Optimisation')
	#ax.plot(a_opt,b_opt,0,"x",color='b')
	ax.legend(loc='best')
	plt.xlabel("lengthscale")
	plt.ylabel("variance")
	plt.show()


def trace_plots_Gaussian(samples_vector):
    plt.plot(samples_vector[0])
    plt.title("Trace plot for lengthscale")
    plt.show()

    plt.plot(samples_vector[1])
    plt.title("Trace plot for sigma_f^2")
    plt.show()

    plt.plot(samples_vector[2])
    plt.title("Trace plot for sigma^2")
    plt.show()

def trace_plots_splines(samples_vector):
    plt.plot(samples_vector[0])
    plt.title("Trace plot for lengthscale")
    plt.savefig("Trace_tau")
    plt.show()

    plt.plot(samples_vector[1])
    plt.title("Trace plot for sigma^2")
    plt.savefig("Trace_alpha0")
    plt.show()


def trace_plots_Poisson(samples_vector):
    plt.plot(samples_vector[0])
    plt.title("Trace plot for sigma_f^2")
    plt.savefig("Trace1")
    plt.show()

    plt.plot(samples_vector[1][:,0])
    plt.title("Trace plot for process")
    plt.savefig("Trace2")
    plt.show()

