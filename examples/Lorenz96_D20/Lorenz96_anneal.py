"""
Example file for carrying out state and parameter estimation in the Lorenz 96
system using the weak-constrained variational method.

Varanneal implements the variational annealing algorithm and uses automatic
differentiation to do the action minimization at each step.
"""

import numpy as np
import varanneal
import sys, time

# Define the model
def l96(t, x, k):
    return np.roll(x,1,1) * (np.roll(x,-1,1) - np.roll(x,2,1)) - x + k

D = 20

# Action/annealing parameters
Lidx = [0, 2, 4, 8, 10, 12, 14, 16]
L = len(Lidx)
# RM, RF
RM = 1.0 / (0.5**2)
RF0 = 4.0e-6
# set alpha and beta values
alpha = 1.5
beta_array = np.linspace(0.0, 100.0, 101)

# Load observed data
data = np.load("l96_D20_dt0p025_N161_sm0p5_sec1_mem1.npy")
times = data[:, 0]
t0 = times[0]
tf = times[-1]
dt = times[1] - times[0]
N = len(times)

# Initial path/parameter guessees
data = data[:, 1:]
data = data[:, Lidx]
Xinit = (20.0*np.random.rand(N*D) - 10.0).reshape((N,D))
for i,l in enumerate(Lidx):
    Xinit[:, l] = data[:, i]
Xinit = Xinit.flatten()

# Parameters
# For P, we just need something; we'll fill in the initial guess below
P = np.array([0.0])
Pidx = [0]  # indices of estimated parameters
Pinit = 4.0 * np.random.rand() + 6.0  # initial parameter guess
XP0 = np.append(Xinit, Pinit)

# Initialize Annealer
anneal1 = varanneal.Annealer()
# Define the model
anneal1.set_model(l96, P, D)
# Load the data into the Annealer object
anneal1.set_data(data, dt, t=times)
# Run the annealing using L-BFGS-B
BFGS_options = {'gtol':1.0e-12, 'ftol':1.0e-12, 'maxfun':1000000, 'maxiter':1000000}

tstart = time.time()
anneal1.anneal(XP0, alpha, beta_array, RM, RF0, Lidx, Pidx,
               disc='SimpsonHermite', method='L-BFGS-B', opt_args=BFGS_options,
               adolcID=0)
print("\nADOL-C annealing completed in %f s."%(time.time() - tstart))

# Save the results of annealing
anneal1.save_paths("paths.npy")
anneal1.save_params("params.npy")
anneal1.save_action_errors("action_errors.npy")
