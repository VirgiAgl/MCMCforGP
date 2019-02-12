import numpy as np
import scipy.stats as sp

def rmse(samples):
    mode = sp.mode(samples)
    rmse = np.sqrt(np.sum((samples - mode[0])**2) / len(samples))
    return rmse

def performance_optimisation(results_optimisation, true_value_pars):
	# Optimisation RMSE
	l_scale_rmse_opt = np.exp(results_optimisation.x[0]) - true_value_pars[0]
	var_rmse_opt = np.exp(results_optimisation.x[1]) - true_value_pars[1]
	noise_var_rmse_opt = np.exp(results_optimisation.x[2]) - true_value_pars[2]
	optimisation_rmse = np.array([l_scale_rmse_opt, var_rmse_opt, noise_var_rmse_opt])
	return optimisation_rmse

def performance_MCMC(l_chain, var_chain, noise_chain, burn_in):
	# MCMC RMSE
	l_scale_rmse_mcmc = rmse(l_chain[burn_in:])
	var_rmse_mcmc = rmse(var_chain[burn_in:])
	noise_var_rmse_mcmc = rmse(noise_chain[burn_in:])
	mcmc_rmse = np.array([l_scale_rmse_mcmc, var_rmse_mcmc, noise_var_rmse_mcmc])
	return mcmc_rmse


def performance_MCMC_Poisson(l_chain, var_chain, burn_in):
	# MCMC RMSE
	l_scale_rmse_mcmc = rmse(l_chain[burn_in:])
	var_rmse_mcmc = rmse(var_chain[burn_in:])
	mcmc_rmse = np.array([l_scale_rmse_mcmc, var_rmse_mcmc])
	return mcmc_rmse