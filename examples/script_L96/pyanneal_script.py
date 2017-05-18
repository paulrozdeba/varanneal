import numpy as np
import adolc
import time
import scipy.optimize as opt
import sys, time

secID = int(sys.argv[1])
memID = int(sys.argv[2])
initID = int(sys.argv[3])
adolcID = int(sys.argv[4])

################################################################################
# Load data, set up parameters, etc.
################################################################################
np.random.seed(12906345 + initID)

# define the vector field
def l96(t, x, k):
    return np.roll(x,1,1) * (np.roll(x,-1,1) - np.roll(x,2,1)) - x + k

f = l96

# twin experiment parameters
D = 20
Lidx = [0, 2, 4, 8, 10, 12, 14, 16]
L = len(Lidx)

# load observed data, and set initial path guess
data = np.load('/home/prozdeba/projects/dynamical_reg/lorenz96/twin_data/D20_dt0p025_N161_sm0p5/noisy_samples/sec%d/l96_D20_dt0p025_N161_sm0p5_sec%d_mem%d.npy'%(secID, secID, memID))
times = data[:, 0]
t = times
t0 = times[0]
tf = times[-1]
dt = times[1] - times[0]
#N = len(times)
N = 17
nstart = 0

data = data[:N, 1:]
data = data[:, Lidx]
Xinit = (20.0*np.random.rand(N*D) - 10.0).reshape((N,D))
for i,l in enumerate(Lidx):
    Xinit[:, l] = data[:, i]
Xinit = Xinit.flatten()

# parameters
#P = 8.17*np.ones(D)
P = np.array([8.17])
#P = (P,)
#Pidx = [0]
Pidx = []
Pinit = 4.0*np.random.rand() + 6.0
#Xinit = np.append(Xinit, Pinit)
XP0 = Xinit

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

RF = RF0

################################################################################

NP = len(P)
NPest = len(Pidx)

t = t[nstart:(nstart + N)]
#Y = data.flatten()
Y = data[nstart:(nstart + N)]
stim = None
if stim is not None:
    stim = stim[nstart:(nstart + N)]

################################################################################
# Define the Gaussian action
################################################################################
def disc_SimpsonHermite(x, p):
    if stim is not None:
        pn = (p, stim[:-2:2])
        pmid = (p, stim[1:-1:2])
        pnp1 = (p, stim[2::2])
    else:
        pn = p
        pmid = p
        pnp1 = p

    fn = f(t[:-2:2], x[:-2:2], pn)
    fmid = f(t[1:-1:2], x[1:-1:2], pmid)
    fnp1 = f(t[2::2], x[2::2], pnp1)

    #disc_vec = np.zeros((N - 1, D), dtype="object")
    disc_vec = np.zeros((N - 1, D), dtype=x.dtype)
    disc_vec[:-1:2] = (fn + 4.0*fmid + fnp1) * (2.0*dt)/6.0
    disc_vec[1::2] = (x[:-2:2] + x[2::2])/2.0 + (fn - fnp1) * (2.0*dt)/8.0

    return disc_vec

disc = disc_SimpsonHermite

# Gaussian action
def A_gaussian_direct(XP):
    """
    Calculate the Gaussian action all in one go.
    """
    # Extract state and parameters from XP
    if NPest == 0:
        x = np.reshape(XP, (N, D))
        p = P
    elif NPest == NP:
        x = np.reshape(XP[:-NP], (N, D))
        p = XP[-NP:]
    else:
        x = np.reshape(XP[:-NPest], (N, D))
        p = []
        j = NPest
        for i in xrange(NP):
            if i in Pidx:
                p.append(XP[-j])
                j -= 1
            else:
                p.append(P[i])

    # Measurement error
    diff = x[:, Lidx] - Y

    if type(RM) == np.ndarray:
        # Contract RM with error
        if RM.shape == (N, L):
            merr = np.sum(RM * diff * diff)
        elif RM.shape == (N, L, L):
            merr = 0.0
            for i in xrange(N):
                merr = merr + np.dot(diff[i], np.dot(RM[i], diff[i]))
        else:
            print("ERROR: RM is in an invalid shape.")
    else:
        merr = RM * np.sum(diff * diff)

    # Model error
    if disc.__name__ == "disc_SimpsonHermite":
        disc_vec = disc(x, p)
        diff1 = x[2::2] - x[:-2:2] - disc_vec[::2]
        diff2 = x[1::2] - disc_vec[1::2]
        #diff = np.reshape(np.hstack((diff1, diff2)), (N - 1, D))
    else:
        diff = x[1:] - x[:-1] - disc(x, p)

    if type(RF) == np.ndarray:
        # Contract RF with the model error time series terms
        if RF.shape == (N - 1, D):
            if disc.__name__ == "disc_SimpsonHermite":
                ferr1 = np.sum(RF[::2] * diff1 * diff1)
                ferr2 = np.sum(RF[1::2] * diff2 * diff2)
                ferr = ferr1 + ferr2
            else:
                ferr = np.sum(RF * diff * diff)

        elif RF.shape == (N - 1, D, D):
            if disc.__name__ == "disc_SimpsonHermite":
                ferr1 = 0.0
                ferr2 = 0.0
                for i in xrange((N - 1) / 2):
                    ferr1 = ferr1 + np.dot(diff1[i], np.dot(RF[2*i], diff1[i]))
                    ferr2 = ferr2 + np.dot(diff2[i], np.dot(RF[2*i+1], diff2[i]))
                ferr = ferr1 + ferr2
            else:
                ferr = 0.0
                for i in xrange(N - 1):
                    ferr = ferr + np.dot(diff[i], np.dot(RF[i], diff))

        else:
            print("ERROR: RF is in an invalid shape.")

    else:
        if disc.__name__ == "disc_SimpsonHermite":
            ferr1 = RF * np.sum(diff1 * diff1)
            ferr2 = RF * np.sum(diff2 * diff2)
            ferr = ferr1 + ferr2
        else:
            ferr = RF * np.sum(diff * diff)

    return merr/(L*N) + ferr/(D*(N-1))

A = A_gaussian_direct

################################################################################
# Initialize the annealing procedure
################################################################################
betaidx = 0
beta = beta_array[betaidx]
Nbeta = len(beta_array)

# array to store minimizing paths
init_to_data = True
minpaths = np.zeros((Nbeta, len(XP0)), dtype='float')
if init_to_data == True:
    X0r = np.reshape(XP0[:N*D], (N, D))
    X0r[:, Lidx] = Y[:]
    if NPest > 0:
        P0 = XP0[-NPest:]
        XP0 = np.append(X0r.flatten(), P0)
    else:
        XP0 = X0r.flatten()
minpaths[0] = XP0

# set current RF
#RF = RF0 * alpha**beta

# array to store minimum action values
A_array = np.zeros(Nbeta, dtype='float')
me_array = np.zeros(Nbeta, dtype='float')
fe_array = np.zeros(Nbeta, dtype='float')

# Store optimization bounds. Will only be used if the chosen
# optimization routine supports it.
bounds = None

# Optimization method
method = 'L-BFGS-B'

# Optimization method options
opt_args = {'gtol':1.0e-12, 'ftol':1.0e-12, 'maxfun':1000000, 'maxiter':1000000}

#if method == 'LM':
#    A = vecA_gaussian

# array to store optimization exit flags
exitflags = np.empty(Nbeta, dtype='int')

################################################################################
# Carry out the annealing
################################################################################
def min_lbfgs_scipy(XP0):
    """
    Minimize f starting from x0 using L-BFGS-B method in scipy.
    This method supports the use of bounds.
    Returns the minimizing state, the minimum function value, and the L-BFGS
    termination information.
    """
    # tape the objective function
    print('Taping action evaluation...')
    tstart = time.time()
    # define a random state vector for the trace
    xtrace = np.random.rand(D*N + NPest)
    adolc.trace_on(adolcID)
    # set the active independent variables
    ax = adolc.adouble(xtrace)
    adolc.independent(ax)
    # set the dependent variable (or vector of dependent variables)
    af = A(ax)
    adolc.dependent(af)
    adolc.trace_off()
    #taped = True
    print('Done!')
    print('Time = {0} s\n'.format(time.time()-tstart))

    # define A and grad_A for evaluation
    def scipy_A(XP):
        return adolc.function(adolcID, XP)

    def scipy_A_grad(XP):
        return adolc.gradient(adolcID, XP)

    def scipy_A_plusgrad(XP):
        return adolc.function(adolcID, XP), adolc.gradient(adolcID, XP)

    # start the optimization
    print("Beginning optimization...")
    tstart = time.time()
    res = opt.minimize(scipy_A_plusgrad, XP0, method='L-BFGS-B', jac=True,
                       options=opt_args, bounds=bounds)
    XPmin,status,Amin = res.x, res.status, res.fun

    print("Optimization complete!")
    print("Time = {0} s".format(time.time()-tstart))
    print("Exit flag = {0}".format(status))
    print("Exit message: {0}".format(res.message))
    print("Iterations = {0}".format(res.nit))
    print("Obj. function value = {0}\n".format(Amin))
    return XPmin, Amin, status

tstart = time.time()
for i in beta_array:
    print('------------------------------')
    print('Step %d of %d'%(betaidx+1,len(beta_array)))
    print('alpha = %f, beta = %f'%(alpha,beta))
    print('')

    if method == 'L-BFGS-B':
        if betaidx == 0:
            XPmin, Amin, exitflag = min_lbfgs_scipy(minpaths[0])
        else:
            XPmin, Amin, exitflag = min_lbfgs_scipy(minpaths[betaidx-1])

    if NPest > 0:
        if isinstance(XPmin[0], adolc._adolc.adouble):
            P[Pidx] = np.array([XPmin[-NPest + i].val for i in xrange(NPest)])
        else:
            P[Pidx] = np.copy(XPmin[-NPest:])

    # store A_min and the minimizing path
    #A_array[betaidx] = Amin
    #me_array[betaidx] = me_gaussian(np.array(XPmin[:N*D]), P)
    #fe_array[betaidx] = fe_gaussian(np.array(XPmin[:N*D]), P)
    minpaths[betaidx] = np.array(XPmin)

    # increase RF
    if betaidx < len(beta_array) - 1:
        betaidx += 1
        beta = beta_array[betaidx]
        RF = RF0 * alpha**beta

print("N = %d annealing finished in %f s."%(N, time.time() - tstart))
