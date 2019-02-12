import numpy as np 
import matplotlib.pyplot as plt

def rbf_kernel(x, var, leng, x_star = None):
	N = len(x)
	if x_star is None:
		Sigma = np.zeros((N, N))
		for i in range(N):
			for j in range(N):
				Sigma[i, j] = var * np.exp(-1 / (2 * leng ** 2) * ((x[i] - x[j]) ** 2))
	else:
		N_star = len(x_star)
		Sigma = np.zeros((N, N_star))
		for i in range(N):
			for j in range(N_star):
				Sigma[i, j] = var * np.exp(-1 / (2 * leng ** 2) * ((x[i] - x_star[j]) ** 2))

	return Sigma 



def thin_plate_spline_kernel(x, tau, m = 1, d=1, x_star = None):
	if x_star is None:
		Sigma = np.zeros((len(x), len(x)))
		for i in range(len(x)):
			for j in range(len(x)):
				Sigma[i, j] = tau**2 * min(x[i], x[j]) 
	else:
		Sigma = np.zeros((len(x), len(x_star)))
		for i in range(len(x)):
			for j in range(len(x_star)):
				Sigma[i, j] = tau**2 * min(x[i], x_star[j]) 
	return Sigma



def thin_plate_spline_kernel_xpart(x, m = 1, d=1, x_star = None):
	if x_star is None:
		Sigma = np.zeros((len(x), len(x)))
		for i in range(len(x)):
			for j in range(len(x)):
				Sigma[i, j] = min(x[i], x[j]) 
	else:
		Sigma = np.zeros((len(x), len(x_star)))
		for i in range(len(x)):
			for j in range(len(x_star)):
				Sigma[i, j] = min(x[i], x_star[j]) 
	return Sigma
