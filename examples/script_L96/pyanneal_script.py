import numpy as np
import adolc
import time
import scipy.optimize as opt
import sys

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
Lidx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
L = len(Lidx)

# load observed data, and set initial path guess
data = np.load('../../noisy_samples_D20_dt0p025_N161_sm0p5/sec%d/l96_D20_dt0p025_N161_sm0p5_sec%d_mem%d.npy'%(secID, secID, memID))
times = data[:, 0]
t = times
t0 = times[0]
tf = times[-1]
dt = times[1] - times[0]
N = len(times)
nstart = 0

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

################################################################################

NP = len(P)
NPest = len(Pidx)

t = t[nstart:(nstart + N)]
Y = Y[nstart:(nstart + N)]
if stim is not None:
    stim = stim[nstart:(nstart + N)]

################################################################################
# Define the Gaussian action
################################################################################
def disc_SimpsonHermite(x, p):
    if stim is not None:
        #sn = stim[:-2:2]
        #smid = stim[1:-1:2]
        #snp1 = stim[2::2]
        pn = (p, stim[:-2:2])
        pmid = (p, stim[1:-1:2])
        pnp1 = (p, stim[2::2])
    else:
        pn = p
        pmid = p
        pnp1 = p

    #xn = x[:-2:2]
    #xmid = x[1:-1:2]
    #xnp1 = x[2::2]
    #
    #tn = t[:-2:2]
    #tmid = t[1:-1:2]
    #tnp1 = t[2::2]

    #fn = f(tn, xn, pn)
    #fmid = f(tmid, xmid, pmid)
    #fnp1 = f(tnp1, xnp1, pnp1)

    fn = f(t[:-2:2], x[:-2:2], pn)
    fmid = f(t[1:-1:2], x[1:-1:2], pmid)
    fnp1 = f(t[2::2], x[2::2], pnp1)

    #disc_vec = np.zeros((N - 1, D), dtype="object")
    disc_vec = np.zeros((N - 1, D), dtype=x.dtype)
    disc_vec[:-1:2] = (fn + 4.0*fmid + fnp1) * (2.0*dt)/6.0
    disc_vec[1::2] = (x[:-2:2] + x[2::2])/2.0 + (fn - fnp1) * (2.0*dt)/8.0

    return disc_vec

disc = disc_SimpsonHermite

# Time series of error vectors.
def me_gaussian_TS_vec(x, p):
    """
    Time series of measurement error vectors, NOT times RM.
    """
    if x.ndim == 1:
        x = np.reshape(x, (N, D))
    diff = x[:, Lidx] - Y

    return diff

def fe_gaussian_TS_vec(x, p):
    """
    Time series of model error vectors, NOT times RF.
    """
    if x.ndim == 1:
        x = np.reshape(x, (N, D))

    if disc.im_func.__name__ == "disc_SimpsonHermite":
        disc_vec = disc(x, p)
        #diff = np.zeros((N - 1, D), dtype="object")
        diff = np.zeros((N - 1, D), dtype=x.dtype)
        diff[:-1:2] = x[2::2] - x[:-2:2] - disc_vec[:-1:2]
        diff[1::2] = x[1::2] - disc_vec[1::2]
    else:
        diff = x[1:] - x[:-1] - disc(x, p)

    return diff

# Time series of squared errors, times RM or RF.
def me_gaussian_TS(x, p):
    """
    Time series of squared measurement errors, times RM.
    """
    diff = me_gaussian_TS_vec(x, p)

    if type(RM) == np.ndarray:
        if RM.shape == (N, L):
            err = RM * diff * diff
        elif RM.shape == (L, L):
            for diffn in diff:
                err[i] = np.dot(diffn, np.dot(RM, diffn))
        elif RM.shape == (N, L, L):
            for diffn,RMn in zip(diff, _RM):
                err[i] = np.dot(diffn, np.dot(RMn, diffn))
        else:
            print("ERROR: RM is in an invalid shape.")
    else:
        err = RM * diff * diff

    return err

def fe_gaussian_TS(x, p):
    """
    Time series of squared model errors, times RF.
    """
    diff = fe_gaussian_TS_vec(x, p)

    if type(RF) == np.ndarray:
        if RF.shape == (D,):
            err = np.zeros(N - 1, dtype=diff.dtype)
            for i in xrange(N - 1):
                err[i] = np.sum(RF * diff[i] * diff[i])

        elif RF.shape == (N - 1, D):
            err = RF * diff * diff

        elif RF.shape == (D, D):
            err = np.zeros(N - 1, dtype=diff.dtype)
            for i in xrange(N - 1):
                err[i] = np.dot(diff[i], np.dot(RF, diff[i]))

        elif RF.shape == (N - 1, D, D):
            err = np.zeros(N - 1, dtype=diff.dtype)
            for i in xrange(N - 1):
                err[i] = np.dot(diff[i], np.dot(RF[i], diff[i]))

        else:
            print("ERROR: RF is in an invalid shape.")

    else:
        err = RF * diff * diff

    return err

# Gaussian action terms for matrix Rf and Rm
def me_gaussian(x, p):
    """
    Gaussian measurement error.
    """
    err = me_gaussian_TS(x, p)
    return np.sum(err) / (L * N)

def fe_gaussian(x, p):
    """
    Gaussian model error.
    """
    err = fe_gaussian_TS(x, p)
    return np.sum(err) / (D * (N - 1))

# Gaussian action
def A_gaussian(XP):
    """
    Gaussian action.
    """
    if NPest == 0:
        x = np.reshape(XP, (N, D))
        p = P
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

    # Traded the statements below in favor of calling stim directly in
    # the discretization functions.
    #if stim is not None:
    #    p = (p, stim)

    # evaluate the action
    me = me_gaussian(x, p)
    fe = fe_gaussian(x, p)
    return me + fe

A = A_gaussian

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
RF = RF0 * alpha**beta

# array to store minimum action values
A_array = np.zeros(Nbeta, dtype='float')
me_array = np.zeros(Nbeta, dtype='float')
fe_array = np.zeros(Nbeta, dtype='float')

# Store optimization bounds. Will only be used if the chosen
# optimization routine supports it.
#if bounds is None:
#    self.bounds = bounds
#else:
#    self.bounds = np.array(bounds)

method = 'L-BFGS-B'

if method == 'LM':
    A = vecA_gaussian

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
    #if taped == False:
    #    tape_A()

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

    # start the optimization
    print("Beginning optimization...")
    tstart = time.time()
    res = opt.minimize(A, XP0, method='L-BFGS-B', jac=scipy_A_grad,
                       options=opt_args, bounds=bounds)
    XPmin,status,Amin = res.x, res.status, res.fun

    print("Optimization complete!")
    print("Time = {0} s".format(time.time()-tstart))
    print("Exit flag = {0}".format(status))
    print("Exit message: {0}".format(res.message))
    print("Iterations = {0}".format(res.nit))
    print("Obj. function value = {0}\n".format(Amin))
    return XPmin, Amin, status

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
    A_array[betaidx] = Amin
    me_array[betaidx] = me_gaussian(np.array(XPmin[:N*D]), P)
    fe_array[betaidx] = fe_gaussian(np.array(XPmin[:N*D]), P)
    minpaths[betaidx] = np.array(XPmin)

    # increase RF
    if betaidx < len(beta_array) - 1:
        betaidx += 1
        beta = beta_array[betaidx]
        RF = RF0 * alpha**beta
