# Testing data assimilation with 1-D harmonic oscillator.
# Uses the "classic" action (only meas and model error, no time delay).
# Minimization routine is (unconstrained) L-BFGS.

import numpy as np
import pyanneal

# define the vector field
def l96(t, x, k):
    return np.roll(x,1,1) * (np.roll(x,-1,1) - np.roll(x,2,1)) - x + k

# twin experiment parameters
D = 5
Lidx = [0, 2, 4]
L = len(Lidx)

# load observed data, and set initial path guess
data = np.load('../sample_l96_D5_N300_sm0p1.npy')[:-1]
times = data[:, 0]
t0 = times[0]
tf = times[-1]
dt = times[1] - times[0]
N = len(times)

data = data[:, 1:]
data = data[:, Lidx]
Xinit = (20.0*np.random.rand(N*D) - 10.0).reshape((N,D))
for i,l in enumerate(Lidx):
    Xinit[:, l] = data[:, i]
Xinit = Xinit.flatten()

# parameters
#P = 8.17*np.ones(D)
P = np.array([8.17])
#P = (P,)
Pidx = [0]
#Pidx = []
Xinit = np.append(Xinit, 2.0)

# RM, RF
#RM = np.resize(np.eye(L),(L,L))/(0.2**2)
RM = 1.0 / (0.1**2)
#RF0 = 0.0001*np.resize(np.eye(D),(D,D)) * dt**2
#RF0_val = .0001 * dt**2
#RF0 = 0.0001 * dt**2
#RF0_val = .0001 * dt**2
RF0 = 0.0001

# set alpha and beta values
alpha = 1.5
beta_array = np.linspace(0.0, 72.0, 73)

# initialize a twin experiment
twin1 = pyanneal.TwinExperiment(l96, dt, D, Lidx, RM, RF0, Y=data, t=times, P=P, Pidx=Pidx)
# run the annealing
twin1.anneal(Xinit, alpha, beta_array, method='L-BFGS-B', disc='SimpsonHermite')

twin1.save_paths("paths_mem0.npy")
twin1.save_params("params_mem0.npy")
twin1.save_action_errors("action_errors_mem0.npy")
