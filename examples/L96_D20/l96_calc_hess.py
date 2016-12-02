"""
Calculates the Hessian of A, given a Lorenz 96 data set and the model.
The data is supplied by you.  The path estimate is either supplied by you 
as well, or comes from running the annealing algorithm.
"""

import numpy as np
import scipy.sparse as spsparse
import pyanneal
import itertools as itt
import sys
import time

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors

# Define the Lorenz 96 vector field
def l96(t, x, k):
    karr = np.array(k)
    return (1.0 + karr) * np.roll(x,1,1) * (np.roll(x,-1,1) - np.roll(x,2,1)) - x + karr

# twin experiment parameters
D = 20
Lidx = [0, 2, 4, 8, 12, 14, 16]
#Lidx = [0, 2, 4, 8, 10, 12, 14, 16]
L = len(Lidx)

# Load observed data
data = np.load("l96_D20_dt0p025_N161_sm0p5_sec1_mem1.npy")[:11]
# Get times from data
times = data[:, 0]
t0 = times[0]
tf = times[-1]
dt = times[1] - times[0]
N = len(times)
# Get only the observed components specified in Lidx
data = data[:, 1:]
data = data[:, Lidx]

# RM, RF
RM = 1.0 / (0.5**2)
RF0 = 4.0E-6

# set alpha and beta values
alpha = 1.5
beta_array = np.linspace(0.0, 100.0, 101, dtype=np.int64)
gammas_all = RF0 * alpha**beta_array

# Now, specify the path estimate.
# In this example we will load the path and parameters from file.
# For now we want the lowest-action paths.  The indices of these paths are 
# saved in a file along with the estimates.
minpaths_indices = np.loadtxt("annealing_path_estimates_L7/sorted_indices.dat", dtype=np.int64)
# Load in the minimum-action path and parameter estimates
Nbeta = beta_array.shape[0]
minpaths = np.zeros((Nbeta, N, D+1), dtype=np.float64)
minparams = np.zeros(Nbeta, dtype=np.float64)
for i in range(Nbeta):
    minpath_idx = minpaths_indices[i, 0]
    minpaths[i] = np.load("annealing_path_estimates_L7/paths_sec1_mem1_%d.npy"%(minpath_idx+1,))[i, :11]
    minparams[i] = np.load("annealing_path_estimates_L7/params_sec1_mem1_%d.npy"%(minpath_idx+1,))[i]

## Set up a pyanneal object, feed in data and stuff, then calculate the Hessian.
## For example, let's look at beta=33
#beta = 42
#X = minpaths[beta][:, 1:]
#P = np.array([minparams[beta],], dtype=np.float64)
#XP = np.append(X, P[0])
#Pidx = [0]  # just needed for syntactical reasons
#RF = RF0 * alpha**beta
#
#twin1 = pyanneal.TwinExperiment(l96, dt, D, Lidx, RM, RF, Y=data, t=times, P=P, Pidx=Pidx, adolcID=0)
#twin1.XP = XP
#twin1.disc = twin1.disc_SimpsonHermite
#twin1.RF = RF
#twin1.tape_A()
##print len(XP)
##print N*D + 1
##grad = np.zeros(len(XP), dtype=np.float64)
##hessian = np.zeros((len(XP), len(XP)), dtype=np.float64)
##Aval = twin1.alglib_lm_A_FGH_hess(twin1.XP, grad, hessian)
##print hessian
#print("Calculating gradient...")
#grad = twin1.scipy_A_grad(twin1.XP)
#print("Calculating Hessian...")
#hessian = twin1.hessian_eval(twin1.XP)
#hessian = spsparse.bsr_matrix(hessian)
##print hessian[:, -2:]
##print np.sqrt(np.dot(grad, grad))
#print np.linalg.det(hessian.toarray()/RF)

# Calculate Hessian determinants
det_array = np.zeros_like(beta_array, dtype=np.float64)
for i,beta in enumerate(beta_array):
    print(i)
    tstart = time.time()
    X = minpaths[beta][:, 1:]
    P = np.array([minparams[beta],], dtype=np.float64)
    XP = np.append(X, P[0])
    Pidx = [0]  # just needed for syntactical reasons
    RF = RF0 * alpha**beta

    twin1 = pyanneal.TwinExperiment(l96, dt, D, Lidx, RM, RF, Y=data, t=times, P=P, Pidx=Pidx, adolcID=0)
    twin1.XP = XP
    twin1.disc = twin1.disc_SimpsonHermite
    twin1.RF = RF
    twin1.tape_A()
    print("Calculating Hessian...")
    hessian = twin1.hessian_eval(twin1.XP)
    hessian = spsparse.bsr_matrix(hessian)
    print("Calculating Hessian determinant...")
    #det_temp = np.abs(np.linalg.det(hessian))
    eigsys = spsparse.linalg.eigs(hessian, k=77, which='LM')
    eigvals = eigsys[0]
    print eigvals
    exit(0)
    det_temp = np.prod(np.sqrt(eigvals))
    print det_temp
    det_array[i] = det_temp
    print("Done in %f s.\n"%(time.time() - tstart))

plt.semilogy(det_array)
plt.show()
exit(0)

#print np.linalg.eig(hessian)
eigsys = spsparse.linalg.eigs(hessian, k=51)
plt.plot(eigsys[0])
print(np.prod(eigsys[0]))
plt.show()

# Plot the gradient and Hessian
# gradient
fig,ax = plt.subplots(1, 1)
fig.set_tight_layout(True)

pltrange = np.linspace(0, len(XP), len(XP), dtype=np.int64)
plt.plot(pltrange, grad)
plt.show()

# Hessian
fig,ax = plt.subplots(1, 1)
fig.set_tight_layout(True)

surf = ax.pcolormesh(pltrange, pltrange, np.abs(hessian.toarray()), rasterized=True, norm=mplcolors.LogNorm())
#ax.set_xlabel(r"$\gamma$")
#ax.set_ylabel(r"$n$")
cbar = fig.colorbar(surf, ax=ax)
cbar.set_label(r"$\|x^{[\gamma]}_\mathsf{est} - x_\mathsf{true}\|^2$")
#ax.set_xlim((pltrange[-10], pltrange[-1]))
#ax.set_ylim((pltrange[0], pltrange[-1]))

plt.show()
