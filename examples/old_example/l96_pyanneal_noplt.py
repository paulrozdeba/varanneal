# Testing data assimilation with 1-D harmonic oscillator.
# Uses the "classic" action (only meas and model error, no time delay).
# Minimization routine is (unconstrained) L-BFGS.

import numpy as np
import pyanneal
from scipy.integrate import ode

# define the vector field
def l96(x, t, k):
    return np.roll(x,1,1) * (np.roll(x,-1,1) - np.roll(x,2,1)) - x + k

def l96_jac(x, t, k):
    D = x.shape[1]
    N = x.shape[0]
    J = np.zeros((N,D,D), dtype='float')
    for i in range(D):
        J[:,i,(i-1)%D] = x[:,(i+1)%D] - x[:,(i-2)%D]
        J[:,i,(i+1)%D] = x[:,(i-1)%D]
        J[:,i,(i-2)%D] = -x[:,(i-1)%D]
        J[:,i,i] = -1.0
    return J

def l96_ode(t, x, k):
    return np.roll(x,1) * (np.roll(x,-1) - np.roll(x,2)) - x + k

def l96_jac_ode(t, x, k):
    D = x.shape[0]
    J = np.zeros((D,D), dtype='float')
    for i in range(D):
        J[i,(i-1)%D] = x[(i+1)%D] - x[(i-2)%D]
        J[i,(i+1)%D] = x[(i-1)%D]
        J[i,(i-2)%D] = -x[(i-1)%D]
        J[i,i] = -1.0
    return J

# twin experiment parameters
N = 301
dt = 0.02
t0 = 0.0
tf = t0 + dt*(N-1)
D = 20
Lidx = (0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19)
L = len(Lidx)

# time recording
times = np.linspace(t0, tf, N)

# parameters
P = (8.17*np.ones(D),)
#P = (P,)
Pidx = ()

################################################################################
# generate twin data
#np.random.seed(19870328)
#ttrans = np.arange(0.0, 100.0+dt, dt)
#Ntrans = len(ttrans)
#x0_trans = 20.0*np.random.rand(D) - 10.0
#x_trans = np.empty((Ntrans,D), dtype='float')
#x_trans[0] = x0_trans
#
#itg = ode(l96_ode, l96_jac_ode)
#itg.set_integrator('dop853')
#itg.set_f_params(P)
#itg.set_jac_params(P)
#
## integrate transient behavior
#itg.set_initial_value(x0_trans, 0.0)
#for i,tnext in enumerate(ttrans[1:]):
#    x_trans[i+1] = itg.integrate(tnext)
#
#x_td = np.empty((N,D), dtype='float')
#x_td[0] = x_trans[-1]
#itg.set_initial_value(x_td[0], t0)
#for i,tnext in enumerate(times[1:]):
#    x_td[i+1] = itg.integrate(tnext)
#x_td += 0.2 * np.random.randn(N,D)
#
#Y = np.reshape(x_td, (N,D))
##Y = Y[:,Lidx]
#
#np.save('l96_twindata_D20_dt0p02.npy', np.hstack((times.reshape(N,1), Y)))
#exit(0)

#plt.plot(times, x_td)
#plt.savefig('l96_td.png')
#exit(0)
################################################################################

# RM, RF
RM = np.resize(np.eye(L),(L,L))/(0.2**2)
RF0 = 0.0001*np.resize(np.eye(D),(D,D)) * dt**2
RF0_val = .0001 * dt**2

# initial guess for the path
data = np.load('l96_twindata_D20_L14_dt0p02.npy')
Xinit = (20.0*np.random.rand(N*D) - 10.0).reshape((N,D))
for i,l in enumerate(Lidx):
    Xinit[:,l] = data[:,i+1]
Xinit = Xinit.flatten()

# set alpha and beta values
alpha = 1.5
beta_array = np.linspace(0.0, 70.0, 71)

# initialize a twin experiment
twin1 = pyanneal.TwinExperiment(l96, Lidx, RM, RF0, data_file='l96_twindata_D20_L14_dt0p02.npy',
                                P=P, Pidx=Pidx)
twin1.anneal(Xinit, alpha, beta_array, method='lbfgs_scipy', disc='impeuler')
twin1.save_as_minAone(savedir='path/')
