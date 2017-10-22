"""
Example file for carrying out state and parameter estimation in the Lorenz 96
system using the weak-constrained variational method.

Varanneal implements the variational annealing algorithm and uses automatic
differentiation to do the action minimization at each step.
"""

import numpy as np
#import varanneal
from varanneal import va_ode
import sys, time

# Define the model
def l96(t, x, k):
    return np.roll(x,1,1) * (np.roll(x,-1,1) - np.roll(x,2,1)) - x + k

D = 20

################################################################################
# Action/annealing parameters
################################################################################
# Measured variable indices
Lidx = [0, 2, 4, 6, 8, 10, 14, 16]
# RM, RF0
RM = 1.0 / (0.5**2)
RF0 = 4.0e-6
# alpha, and beta ladder
alpha = 1.5
beta_array = np.linspace(0, 60, 61)

################################################################################
# Load observed data
################################################################################
data = np.load("l96_D20_dt0p025_N161_sm0p5_sec1_mem1.npy")
times_data = data[:, 0]
#t0 = times_data[0]
#tf = times_data[-1]
dt_data = times_data[1] - times_data[0]
N_data = len(times_data)

data = data[:, 1:]
data = data[:, Lidx]

################################################################################
# Initial path/parameter guesses
################################################################################
# Same sampling rate for data and forward mapping
dt_model = dt_data
N_model = N_data
X0 = (20.0*np.random.rand(N_model * D) - 10.0).reshape((N_model, D))

# Sample forward mapping twice as f
#dt_model = dt_data / 2.0
#meas_nskip = 2
#N_model = (N_data - 1) * meas_nskip + 1
#X0 = (20.0*np.random.rand(N_model * D) - 10.0).reshape((N_model, D))

# Below lines are for initializing measured components to data; instead, we
# use the convenience option "init_to_data=True" in the anneal() function below.
#for i,l in enumerate(Lidx):
#    Xinit[:, l] = data[:, i]
#Xinit = Xinit.flatten()

# Parameters
Pidx = [0]  # indices of estimated parameters
# Initial guess
P0 = np.array([4.0 * np.random.rand() + 6.0])  # Static parameter
#Pinit = 4.0 * np.random.rand(N_model, 1) + 6.0  # Time-dependent parameter

################################################################################
# Annealing
################################################################################
# Initialize Annealer
anneal1 = va_ode.Annealer()
# Set the Lorenz 96 model
anneal1.set_model(l96, D)
# Load the data into the Annealer object
anneal1.set_data(data, t=times_data)

# Add bounds
state_bounds = (-10, 10)
param_bounds = (8, 8.8)
bounds_full = []
for iT in range(D):
	bounds_full.append(state_bounds)
bounds_full.append(param_bounds)

# Run the annealing using L-BFGS-B
BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 'maxiter':1000000}
tstart = time.time()
anneal1.anneal(X0, P0, alpha, beta_array, RM, RF0, Lidx, Pidx, dt_model=dt_model,
               init_to_data=True, bounds=bounds_full, disc='SimpsonHermite', 
               method='L-BFGS-B', opt_args=BFGS_options, adolcID=0)
print("\nADOL-C annealing completed in %f s."%(time.time() - tstart))

# Save the results of annealing
anneal1.save_paths("paths.npy")
anneal1.save_params("params.npy")
anneal1.save_action_errors("action_errors.npy")
