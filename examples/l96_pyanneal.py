# Testing data assimilation with 1-D harmonic oscillator.
# Uses the "classic" action (only meas and model error, no time delay).
# Minimization routine is (unconstrained) L-BFGS.

import numpy as np
import pyanneal
from scipy.integrate import ode
import sys

try:
    plotflag = sys.argv[1]
except:
    plotflag = False
if plotflag is '-p':
    import matplotlib.pyplot as plt
    plotflag = True
else:
    plotflag = False

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
#N = 301
#N = 30
#dt = 0.02
#t0 = 0.0
#tf = t0 + dt*(N-1)
D = 5
#Lidx = (0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19)
#Lidx = (0, 5, 10, 15)
Lidx = [0, 2, 4]
L = len(Lidx)

# load observed data, and set initial path guess
#data = np.load('l96_twindata_D20_L14_dt0p02.npy')[:N]
data = np.load('sample_l96_D5_N300_sm0p1.npy')[-17:]
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

# time recording
#times = np.linspace(t0, tf, N)

# parameters
#P = 8.17*np.ones(D)
P = np.array([8.17])
#P = (P,)
#Pidx = [0]
Pidx = []
#Xinit = np.append(Xinit, 6.0)

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
#RM = np.resize(np.eye(L),(L,L))/(0.2**2)
RM = 1.0 / (0.1**2)
#RF0 = 0.0001*np.resize(np.eye(D),(D,D)) * dt**2
#RF0_val = .0001 * dt**2
#RF0 = 0.0001 * dt**2
#RF0_val = .0001 * dt**2
RF0 = 0.0001

# set alpha and beta values
alpha = 1.5
beta_array = np.linspace(0.0, 80.0, 81)

# initialize a twin experiment
#twin1 = pyanneal.TwinExperiment(l96, Lidx, RM, RF0, data_file='l96_twindata_D20_L14_dt0p02.npy',
#                                P=P, Pidx=Pidx)
twin1 = pyanneal.TwinExperiment(l96, dt, D, Lidx, RM, RF0, Y=data, t=times, P=P, Pidx=Pidx)
twin1.anneal(Xinit, alpha, beta_array, method='L-BFGS-B', disc='SimpsonHermite')
twin1.save_as_minAone(savedir='path/')

if plotflag == True:
    plt.plot(np.log10(twin1.A_array), 'k.', ms=5)
    plt.show()
    #plt.savefig('actionplot.png')

    plt.plot(np.log10(twin1.fe_array/(RF0_val*alpha**beta_array)), np.log10(twin1.me_array), 'k.', ms=5)
    plt.show()

    Xfinal = np.reshape(twin1.minpaths[-1], (N,D))
    alldata = np.load('l96_twindata_D20_L20.npy')[:,1:]

    #fig,ax = plt.subplots(D, 1, sharex=True)
    #fig.set_tight_layout(True)
    #for i in range(D):
    #    ax[i].plot(times, Xfinal[:,i], 'k-')
    #    ax[i].plot(times, alldata[:,i], 'r.', ms=5)
    #plt.show()
    #plt.savefig('finalpath.png')
