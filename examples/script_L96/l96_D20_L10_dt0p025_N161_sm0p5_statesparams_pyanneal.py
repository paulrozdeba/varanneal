import numpy as np
import pyanneal
import itertools as itt
import sys

secID = int(sys.argv[1])
memID = int(sys.argv[2])
initID = int(sys.argv[3])
SGE_TASK_ID = int(sys.argv[4])

np.random.seed(12906345 + initID)

# define the vector field
def l96(t, x, k):
    return np.roll(x,1,1) * (np.roll(x,-1,1) - np.roll(x,2,1)) - x + k

# twin experiment parameters
D = 20
Lidx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
L = len(Lidx)

# load observed data, and set initial path guess
data = np.load('../../noisy_samples_D20_dt0p025_N161_sm0p5/sec%d/l96_D20_dt0p025_N161_sm0p5_sec%d_mem%d.npy'%(secID, secID, memID))
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
Pinit = 4.0*np.random.rand() + 6.0
Xinit = np.append(Xinit, Pinit)

# RM, RF
#RM = np.resize(np.eye(L),(L,L))/(0.2**2)
RM = 1.0 / (0.5**2)
#RF0 = 0.0001*np.resize(np.eye(D),(D,D)) * dt**2
#RF0_val = .0001 * dt**2
#RF0 = 0.0001 * dt**2
#RF0_val = .0001 * dt**2
RF0 = 4.0E-6

# set alpha and beta values
alpha = 1.5
beta_array = np.linspace(0.0, 100.0, 101)

# initialize a twin experiment
twin1 = pyanneal.TwinExperiment(l96, dt, D, Lidx, RM, RF0, Y=data, t=times, P=P,
                                Pidx=Pidx, adolcID=SGE_TASK_ID)
# run the annealing
twin1.opt_args = {'gtol':1.0e-12, 'ftol':1.0e-12, 'maxfun':1000000, 'maxiter':1000000}
twin1.anneal(Xinit, alpha, beta_array, method='L-BFGS-B', disc='SimpsonHermite')

twin1.save_paths("sec%d/mem%d/paths_sec%d_mem%d_%d.npy"%(secID, memID, secID, memID, initID))
twin1.save_params("sec%d/mem%d/params_sec%d_mem%d_%d.npy"%(secID, memID, secID, memID, initID))
twin1.save_action_errors("sec%d/mem%d/aerr_sec%d_mem%d_%d.npy"%(secID, memID, secID, memID, initID))
