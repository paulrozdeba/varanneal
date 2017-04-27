# Testing data assimilation with 1-D harmonic oscillator.
# Uses the "classic" action (only meas and model error, no time delay).
# Minimization routine is (unconstrained) L-BFGS.

import numpy as np
import pyanneal
import sys

try:
    TASKID = int(sys.argv[1])
except:
    TASKID = 0

# define the vector field
def l96(t, x, k):
    return np.roll(x,1,1) * (np.roll(x,-1,1) - np.roll(x,2,1)) - x + k

# twin experiment parameters
D = 20
Lidx = [0, 2, 4, 8, 10, 12, 14, 16]
L = len(Lidx)

# load observed data, and set initial path guess
data = np.load("l96_D20_dt0p025_N161_sm0p5_sec1_mem1.npy")
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
Pinit = 4.0 * np.random.rand() + 6.0
Xinit = np.append(Xinit, Pinit)

# RM, RF
#RM = np.resize(np.eye(L),(L,L))/(0.2**2)
RM = 1.0 / (0.5**2)
#RF0 = 0.0001*np.resize(np.eye(D),(D,D)) * dt**2
#RF0_val = .0001 * dt**2
#RF0 = 0.0001 * dt**2
#RF0_val = .0001 * dt**2
RF0 = 4.0e-6

# set alpha and beta values
alpha = 1.5
beta_array = np.linspace(0.0, 100.0, 101)

# initialize a twin experiment
twin1 = pyanneal.TwinExperiment(l96, dt, D, Lidx, RM, RF0, Y=data, t=times, P=P, Pidx=Pidx, adolcID=TASKID)
# run the annealing
twin1.opt_args = {'gtol':1.0e-12, 'ftol':1.0e-12, 'maxfun':1000000, 'maxiter':1000000}
twin1.anneal(Xinit, alpha, beta_array, method='L-BFGS-B', disc='SimpsonHermite')

twin1.save_paths("paths.npy")
twin1.save_params("params.npy")
twin1.save_action_errors("action_errors.npy")
