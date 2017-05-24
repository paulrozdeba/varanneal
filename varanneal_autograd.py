"""
Carry out variational annealing.

Variational annealing is a numerical continuation algorithm for estimating
states and parameters in a dynamical model using a variational method of
data assimilation.  This algorithm was proposed by () in [].

annealvar uses automatic differentiation (AD) to calculate derivatives of the 
action defined in [].

This module contains the class definition for the Annealer object, which is
an instantiation of a variational annealing calculation.  The user sets a model
and a data trajectory through the object isntance, and for the annealing
procedure may choose:
  1. R_m, the initial value for R_f, and the exponential "ladder" parameterized
     by alpha and beta.
  2. The time-discretization routine used for f in the model error term of the
     action.
  3. The measured components of the system, the parameters to be estimated, and
     bounds on all the states and parameters if the chosen optimization
     routine supports them.
  4. The optimization routine (current choices include L-BFGS-B, nonlinear
     conjugate gradient (NCG), Levenberg-Marquardt (LM), and a truncated
     Newton algorithm (TNC)).
"""

import autograd.numpy as np
import autograd as ag
import time
import scipy.optimize as opt

class Annealer:
    def __init__(self):
        """
        Constructor for the Annealer class.
        """
        self.taped = []
        self.annealing_initialized = False

    def set_model(self, f, p, D):
        """
        Set the D-dimensional dynamical model for the estimated system.
        The model must take arguments in the following order:
            t, x, p
        or, if there is a time-dependent stimulus for f (nonautonomous term):
            t, x, (p, stim)
        where x and stim are at the "current" time t.  Thus, x should be a
        D-dimensional vector, and stim similarly a D_stim-dimensional vector.

        Also set static values for the model parameters.
        Later, you can set which parameters are to be estimated during the 
        annealing when the anneal functions are called.
        """
        self.f = f
        self.P = p
        self.D = D

        self.NP = len(p)

    def set_data_fromfile(self, data_file, dt, stim_file=None):
        """
        Load data & stimulus time series from file.
        If data is a text file, must be in multi-column format with L+1 columns:
            t  y_1  y_2  ...  y_L
        If a .npy archive, should contain an N X (L+1) array with times in the
        zeroth element of each entry.
        Column/array formats should also be in the form t  s_1  s_2 ...
        """
        if data_file.endswith('npy'):
            data = np.load(data_file)
        else:
            data = np.loadtxt(data_file)
        self.Y = data[:, 1:]

        if stim_file.endswith('npy'):
            s = np.load(stim_file)
        else:
            s = np.loadtxt(stim_file)
        self.stim = s[:, 1:]

        self.dt = dt

    def set_data(self, data, dt, stim=None, t=None, nstart=0, N=None):
        """
        Directly pass in data and stim arrays
        If you pass in t, it's assumed y/stim does not contain time.  Otherwise,
        it has to contain time in the zeroth element of each sample.
        """
        if N is None:
            self.N = data.shape[0]
        else:
            self.N = N

        if t is None:
            self.t = data[nstart:(nstart + self.N), 0]
            self.Y = data[nstart:(nstart + self.N), 1:]
            if stim is not None:
                self.stim = stim[nstart:(nstart + self.N), 1:]
            else:
                self.stim = None
        else:
            self.t = t[nstart:(nstart + self.N)]
            self.Y = data[nstart:(nstart + self.N)]
            if stim is not None:
                self.stim = stim[nstart:(nstart + self.N)]
            else:
                self.stim = None

        self.dt = dt

    ############################################################################
    # Gaussian action
    ############################################################################
    def A_gaussian(self, XP):
        """
        Calculate the Gaussian action all in one go.
        """
        merr = self.me_gaussian(XP[:self.N*self.D])
        ferr = self.fe_gaussian(XP)
        return merr + ferr

    def me_gaussian(self, X):
        """
        Gaussian measurement error.
        """
        x = np.reshape(X, (self.N, self.D))
        diff = x[:, self.Lidx] - self.Y

        if type(self.RM) == np.ndarray:
            # Contract RM with error
            if self.RM.shape == (self.N, self.L):
                merr = np.einsum('ij, ij, ij', diff, self.RM, diff)
            elif self.RM.shape == (self.N, self.L, self.L):
                merr = np.einsum('ij, ijk, ik', diff, self.RM, diff)
            else:
                print("ERROR: RM is in an invalid shape.")
        else:
            merr = self.RM * np.einsum('ij, ij', diff, diff)

        return merr / (self.L * self.N)

    def fe_gaussian(self, XP):
        """
        Gaussian model error.
        """
        # Extract state and parameters from XP.
        if self.NPest == 0:
            x = np.reshape(XP, (self.N, self.D))
            p = self.P
        elif self.NPest == self.NP:
            x = np.reshape(XP[:-self.NP], (self.N, self.D))
            p = XP[-self.NP:]
        else:
            x = np.reshape(XP[:-self.NPest], (self.N, self.D))
            p = []
            j = self.NPest
            for i in xrange(self.NP):
                if i in self.Pidx:
                    p.append(XP[-j])
                    j -= 1
                else:
                    p.append(self.P[i])

        # Start calculating the model error.
        if self.disc.im_func.__name__ == "disc_SimpsonHermite":
            #disc_vec = self.disc(x, p)
            disc_vec1, disc_vec2 = self.disc(x, p)
            diff1 = x[2::2] - x[:-2:2] - disc_vec1
            diff2 = x[1::2] - disc_vec2
        else:
            diff = x[1:] - x[:-1] - self.disc(x, p)

        if type(self.RF) == np.ndarray:
            # Contract RF with the model error time series terms
            if self.RF.shape == (self.N - 1, self.D):
                if self.disc.im_func.__name__ == "disc_SimpsonHermite":
                    ferr1 = np.einsum('ij, ij, ij', diff1, self.RF[::2], diff1)
                    ferr2 = np.einsum('ij, ij, ij', diff2, self.RF[1::2], diff2)
                    ferr = ferr1 + ferr2
                else:
                    ferr = np.einsum('ij, ij, ij', diff, self.RF, diff)

            elif self.RF.shape == (self.N - 1, self.D, self.D):
                if self.disc.im_func.__name__ == "disc_SimpsonHermite":
                    ferr1 = np.einsum('ij, ijk, ik', diff1, self.RF[::2], diff1)
                    ferr2 = np.einsum('ij, ijk, ik', diff2, self.RF[1::2], diff2)
                    ferr = ferr1 + ferr2
                else:
                    ferr = np.einsum('ij, ijk, ik', diff, self.RF, diff)

            else:
                print("ERROR: RF is in an invalid shape.")

        else:
            if self.disc.im_func.__name__ == "disc_SimpsonHermite":
                ferr1 = self.RF * np.einsum('ij, ij', diff1, diff1)
                ferr2 = self.RF * np.einsum('ij, ij', diff2, diff2)
                ferr = ferr1 + ferr2
            else:
                ferr = self.RF * np.einsum('ij, ij', diff, diff)

        return ferr / (self.D * (self.N - 1))

    def vecA_gaussian(self, XP):
        """
        Vector-like terms of the Gaussian action.
        This is here primarily for Levenberg-Marquardt.
        """
        # Extract state and parameters from XP
        if self.NPest == 0:
            x = np.reshape(XP, (self.N, self.D))
            p = self.P
        elif self.NPest == self.NP:
            x = np.reshape(XP[:-self.NP], (self.N, self.D))
            p = XP[-self.NP:]
        else:
            x = np.reshape(XP[:-self.NPest], (self.N, self.D))
            p = []
            j = self.NPest
            for i in xrange(self.NP):
                if i in self.Pidx:
                    p.append(XP[-j])
                    j -= 1
                else:
                    p.append(self.P[i])

        # Evaluate the vector-like terms of the action.
        # Measurement error
        diff = x[:, self.Lidx] - self.Y

        if type(self.RM) == np.ndarray:
            # Contract RM with error
            if self.RM.shape == (self.N, self.L):
                merr = self.RM * diff
            elif self.RM.shape == (self.N, self.L, self.L):
                merr = np.einsum('...ij, ...j', self.RM, diff)
            else:
                print("ERROR: RM is in an invalid shape.")
        else:
            merr = self.RM * diff

        # Model error
        if self.disc.im_func.__name__ == "disc_SimpsonHermite":
            #disc_vec = self.disc(x, p)
            disc_vec1, disc_vec2 = self.disc(x, p)
            diff1 = x[2::2] - x[:-2:2] - disc_vec1
            diff2 = x[1::2] - disc_vec2
        else:
            diff = x[1:] - x[:-1] - self.disc(x, p)

        if type(self.RF) == np.ndarray:
            # Contract RF with the model error time series terms
            if self.RF.shape == (self.N - 1, self.D):
                if self.disc.im_func.__name__ == "disc_SimpsonHermite":
                    ferr1 = self.RF[::2] * diff1
                    ferr2 = self.RF[1::2] * diff2
                    ferr = np.append(ferr1, ferr2)
                else:
                    ferr = self.RF * diff

            elif self.RF.shape == (self.N - 1, self.D, self.D):
                if self.disc.im_func.__name__ == "disc_SimpsonHermite":
                    ferr1 = np.einsum('...ij, ...j', self.RF[::2], diff1)
                    ferr2 = np.einsum('...ij, ...j', self.RF[1::2], diff2)
                    ferr = np.append(ferr1, ferr2)
                else:
                    ferr = np.einsum('...ij, ...j', self.RF, diff)

            else:
                print("ERROR: RF is in an invalid shape.")

        else:
            if self.disc.im_func.__name__ == "disc_SimpsonHermite":
                ferr1 = self.RF * diff1
                ferr2 = self.RF * diff2
                ferr = np.append(ferr1, ferr2)
            else:
                ferr = self.RF * diff

        return np.append(merr/(self.N * self.L), ferr/((self.N - 1) * self.D))

    ############################################################################
    # Discretization routines
    ############################################################################
    def disc_trapezoid(self, x, p):
        """
        Time discretization for the action using the trapezoid rule.
        """
        if self.stim is not None:
            pn = (p, self.stim[:-1])
            pnp1 = (p, self.stim[1:])
        else:
            pn = p
            pnp1 = p

        fn = self.f(self.t[:-1], x[:-1], pn)
        fnp1 = self.f(self.t[1:], x[1:], pnp1)

        return self.dt * (fn + fnp1) / 2.0

    def disc_rk4(self, x, p):
        """
        RK4 time discretization for the action.
        """
        xn = x[:-1]
        tn = np.tile(self.t[:-1], (self.D, 1)).T
        k1 = self.f(tn, xn, p)
        k2 = self.f(tn + self.dt/2.0, xn + k1*self.dt/2.0, p)
        k3 = self.f(tn + self.dt/2.0, xn + k2*self.dt/2.0, p)
        k4 = self.f(tn + self.dt, xn + k3*self.dt, p)
        return self.dt * (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

    def disc_SimpsonHermite(self, x, p):
        """
        Simpson-Hermite time discretization for the action.
        This discretization applies Simpson's rule to all the even-index time
        points, and a Hermite polynomial interpolation for the odd-index points
        in between.
        """
        if self.stim is not None:
            pn = (p, self.stim[:-2:2])
            pmid = (p, self.stim[1:-1:2])
            pnp1 = (p, self.stim[2::2])
        else:
            pn = p
            pmid = p
            pnp1 = p

        fn = self.f(self.t[:-2:2], x[:-2:2], pn)
        fmid = self.f(self.t[1:-1:2], x[1:-1:2], pmid)
        fnp1 = self.f(self.t[2::2], x[2::2], pnp1)

        #disc_vec = np.zeros((self.N - 1, self.D), dtype="object")
        #disc_vec = np.zeros((self.N - 1, self.D), dtype=x.dtype)
        #disc_vec[:-1:2] = (fn + 4.0*fmid + fnp1) * (2.0*self.dt)/6.0
        #disc_vec[1::2] = (x[:-2:2] + x[2::2])/2.0 + (fn - fnp1) * (2.0*self.dt)/8.0
        disc_vec1 = (fn + 4.0*fmid + fnp1) * (2.0*self.dt)/6.0
        disc_vec2 = (x[:-2:2] + x[2::2])/2.0 + (fn - fnp1) * (2.0*self.dt)/8.0

        return disc_vec1, disc_vec2

    ############################################################################
    # Annealing functions
    ############################################################################
    def anneal(self, XP0, alpha, beta_array, RM, RF0, Lidx, Pidx, 
               init_to_data=True, action='A_gaussian', disc='trapezoid',
               method='L-BFGS-B', bounds=None, opt_args=None):
        """
        Convenience function to carry out a full annealing run over all values
        of beta in beta_array.
        """
        # initialize the annealing procedure, if not already done
        if self.annealing_initialized == False:
            self.anneal_init(XP0, alpha, beta_array, RM, RF0, Lidx, Pidx,
                             init_to_data, action, disc, method, bounds,
                             opt_args)
        for i in beta_array:
            print('------------------------------')
            print('Step %d of %d'%(self.betaidx+1, len(self.beta_array)))
            print('beta = %d, RF = %.8e'%(self.beta, self.RF))
            print('')
            self.anneal_step()

    def anneal_init(self, XP0, alpha, beta_array, RM, RF0, Lidx, Pidx, 
                    init_to_data=True, action='A_gaussian', disc='trapezoid',
                    method='L-BFGS-B', bounds=None, opt_args=None):
        """
        Initialize the annealing procedure.
        """
        if method not in ('L-BFGS-B', 'NCG', 'LM', 'TNC'):
            print("ERROR: Optimization routine not recognized. Annealing not initialized.")
            return 1
        else:
            self.method = method

        # Store optimization bounds. Will only be used if the chosen
        # optimization routine supports it.
        if bounds is None:
            self.bounds = bounds
        else:
            self.bounds = np.array(bounds)

        # get optimization extra arguments
        self.opt_args = opt_args

        # get indices of measured components of f
        self.Lidx = Lidx
        self.L = len(Lidx)
        # get indices of parameters to be estimated by annealing
        self.Pidx = Pidx
        self.NPest = len(Pidx)

        # properly set up the bounds arrays
        if bounds is not None:
            bounds_full = []
            state_b = bounds[:self.D]
            for i in range(self.N):
                for j in range(self.D):
                    bounds_full.append(state_b[j])
            for i in range(self.NPest):
                bounds_full.append(bounds[self.D + i])
        else:
            bounds_full = None

        # Reshape RM and RF so that they span the whole time series.  This is
        # done because in the action evaluation, it is more efficient to let
        # numpy handle multiplication over time rather than using python loops.
        if type(RM) == np.ndarray:
            if RM.shape == (self.L,):
                self.RM = np.resize(RM, (self.N, self.L))
            elif RM.shape == (self.L, self.L):
                self.RM = np.resize(RM, (self.N, self.L, self.L))
            elif RM.shape == (self.N, self.L) or RM.shape == np.resize(self.N, self.L, self.L):
                self.RM = RM
            else:
                print("ERROR: RM has an invalid shape. Exiting.")
                exit(1)

        else:
            self.RM = RM

        if type(RF0) == np.ndarray:
            if RF0.shape == (self.D,):
                self.RF0 = np.resize(RF0, (self.N - 1, self.D))
            elif RF0.shape == (self.D, self.D):
                self.RF0 = np.resize(RF0, (self.N - 1, self.D, self.D))
            elif RF0.shape == (self.N - 1, self.D) or RF0.shape == (self.N - 1, self.D, self.D):
                self.RF0 = RF0
            else:
                print("ERROR: RF0 has an invalid shape. Exiting.")
                exit(1)

        else:
            self.RF0 = RF0

        # set up beta array in RF = RF0 * alpha**beta
        self.alpha = alpha
        self.beta_array = beta_array
        self.betaidx = 0
        self.beta = self.beta_array[self.betaidx]
        self.Nbeta = len(self.beta_array)

        # set current RF
        if RF0 is not None:
            self.RF0 = RF0
        self.RF = self.RF0 * self.alpha**self.beta

        # set the desired action
        if self.method == 'LM':
            # Levenberg-Marquardt requires a "vector action"
            self.A = self.vecA_gaussian
        elif type(action) == str:
            exec 'self.A = self.%s'%(action)
        else:
            # Assumption: user has passed a function pointer
            self.A = action

        # set the discretization
        exec 'self.disc = self.disc_%s'%(disc,)

        # array to store minimizing paths
        self.minpaths = np.zeros((self.Nbeta, len(XP0)), dtype='float')
        if init_to_data == True:
            X0r = np.reshape(XP0[:self.N*self.D], (self.N, self.D))
            X0r[:, self.Lidx] = self.Y[:]
            if self.NPest > 0:
                P0 = XP0[-self.NPest:]
                XP0 = np.append(X0r.flatten(), P0)
            else:
                XP0 = X0r.flatten()
        self.minpaths[0] = XP0

        # array to store optimization results
        self.A_array = np.zeros(self.Nbeta, dtype='float')
        self.me_array = np.zeros(self.Nbeta, dtype='float')
        self.fe_array = np.zeros(self.Nbeta, dtype='float')
        self.exitflags = np.empty(self.Nbeta, dtype='int')

        # Initialization successful, we're at the beta = beta_0 step now.
        self.initalized = True

    def anneal_step(self):
        """
        Perform a single annealing step. The cost function is minimized starting
        from the previous minimum (or the initial guess, if this is the first
        step). Then, RF is increased to prepare for the next annealing step.
        """
        # minimize A using the chosen method
        if self.method == 'L-BFGS-B':
            if self.betaidx == 0:
                XPmin, Amin, exitflag = self.min_lbfgs_scipy(self.minpaths[0])
            else:
                XPmin, Amin, exitflag = self.min_lbfgs_scipy(self.minpaths[self.betaidx-1])

        elif self.method == 'NCG':
            if self.betaidx == 0:
                XPmin, Amin, exitflag = self.min_cg_scipy(self.minpaths[0])
            else:
                XPmin, Amin, exitflag = self.min_cg_scipy(self.minpaths[self.betaidx-1])

        elif self.method == 'TNC':
            if self.betaidx == 0:
                XPmin, Amin, exitflag = self.min_tnc_scipy(self.minpaths[0])
            else:
                XPmin, Amin, exitflag = self.min_tnc_scipy(self.minpaths[self.betaidx-1])

        elif self.method == 'LM':
            if self.betaidx == 0:
                XPmin, Amin, exitflag = self.min_lm_scipy(self.minpaths[0])
            else:
                XPmin, Amin, exitflag = self.min_lm_scipy(self.minpaths[self.betaidx-1])

        else:
            print("ERROR: Optimization routine not implemented or recognized.")

        # update optimal parameter values
        if self.NPest > 0:
            self.P[self.Pidx] = XPmin[-self.NPest:]

        # store A_min and the minimizing path
        self.A_array[self.betaidx] = Amin
        self.me_array[self.betaidx] = self.me_gaussian(np.array(XPmin[:self.N*self.D]))
        self.fe_array[self.betaidx] = self.fe_gaussian(np.array(XPmin))
        self.minpaths[self.betaidx] = np.array(XPmin)

        # increase RF
        if self.betaidx < len(self.beta_array) - 1:
            self.betaidx += 1
            self.beta = self.beta_array[self.betaidx]
            self.RF = self.RF0 * self.alpha**self.beta

        # set flags indicating that A needs to be retaped, and that we're no
        # longer at the beginning of the annealing procedure
        self.taped = []
        if self.annealing_initialized:
            # Indicate no longer at beta_0
            self.initialized = False

    ################################################################################
    # Routines to save annealing results.
    ################################################################################
    def save_paths(self, filename):
        """
        Save minimizing paths (not including parameters).
        """
        if self.NPest == 0:
            savearray = np.reshape(self.minpaths[:], (self.Nbeta, self.N, self.D))
        else:
            savearray = np.reshape(self.minpaths[:, :-self.NPest], (self.Nbeta, self.N, self.D))

        # append time
        tsave = np.reshape(self.t, (self.N, 1))
        tsave = np.resize(tsave, (self.Nbeta, self.N, 1))
        savearray = np.dstack((tsave, savearray))

        if filename.endswith('.npy'):
            np.save(filename, savearray)
        else:
            np.savetxt(filename, savearray)

    def save_params(self, filename):
        """
        Save minimum action parameter values.
        """
        if self.NPest == 0:
            print("WARNING: You did not estimate any parameters.  Writing fixed " \
                  + "parameter values to file anyway.")

        # write fixed parameters to array
        savearray = np.resize(self.P, (self.Nbeta, self.NP))
        # write estimated parameters to array
        if self.NPest > 0:
            est_param_array = np.reshape(self.minpaths[:, -self.NPest:], (self.Nbeta, self.NPest))
            savearray[:, self.Pidx] = est_param_array

        if filename.endswith('.npy'):
            np.save(filename, savearray)
        else:
            np.savetxt(filename, savearray)

    def save_action_errors(self, filename, cmpt=0):
        """
        Save beta values, action, and errors (with/without RM and RF) to file.
        cmpt sets which component of RF0 to normalize by.
        """
        savearray = np.zeros((self.Nbeta, 5))
        savearray[:, 0] = self.beta_array
        savearray[:, 1] = self.A_array
        savearray[:, 2] = self.me_array
        savearray[:, 3] = self.fe_array

        # Save model error / RF
        if type(self.RF) == np.ndarray:
            if self.RF0.shape == (self.N - 1, self.D):
                savearray[:, 4] = self.fe_array / (self.RF0[0, 0] * self.alpha**self.beta_array)
            elif self.RF0.shape == (self.N - 1, self.D, self.D):
                savearray[:, 4] = self.fe_array / (self.RF0[0, 0, 0] * self.alpha**self.beta_array)
            else:
                print("RF shape currently not supported for saving.")
                return 1
        else:
            savearray[:, 4] = self.fe_array / (self.RF0 * self.alpha**self.beta_array)

        if filename.endswith('.npy'):
            np.save(filename, savearray)
        else:
            np.savetxt(filename, savearray)

    def save_as_minAone(self, savedir='', savefile=None, pathinit=0):
        """
        Save the result of this annealing in minAone data file style.
        """
        if savedir.endswith('/') == False:
            savedir += '/'
        if savefile is None:
            savefile = savedir + 'D%d_M%d_PATH%d.dat'%(self.D, self.L, pathinit)
        else:
            savefile = savedir + savefile
        betaR = self.beta_array.reshape((self.Nbeta,1))
        exitR = self.exitflags.reshape((self.Nbeta,1))
        AR = self.A_array.reshape((self.Nbeta,1))
        savearray = np.hstack((betaR, exitR, AR, self.minpaths))
        np.savetxt(savefile, savearray)

    ############################################################################
    # AD taping & derivatives
    ############################################################################
    def init_gradA(self):
        """
        Initialize the gradient of the action."
        """
        if 'gradA' in self.taped:
            print("Warning: Re-taping the gradient.")
        print("Initializing AD gradient...")
        tstart = time.time()
        self.gradA_taped = ag.grad(self.A)
        self.taped.append('gradA')
        print("Done!")
        print("Time = %f s\n"%(time.time() - tstart,))
    
    def init_A_gradA(self):
        """
        Initialize the objective-and-gradient function.
        """
        if 'A_gradA' in self.taped:
            print("Warning: Re-taping A + gradient.")
        print("Initializing AD objective-and-gradient...")
        tstart = time.time()
        self.A_gradA_taped = ag.value_and_grad(self.A)
        self.taped.append('A_gradA')
        print("Done!")
        print("Time = %f s\n"%(time.time() - tstart,))

    def init_jacA(self):
        """
        Initialize the jacobian.
        """
        if 'jacA' in self.taped:
            print("Warning: Re-taping the Jacobian.")
        print("Initializing AD Jacobian...")
        tstart = time.time()
        self.jacA_taped = ag.jacobian(self.A)
        self.taped.append('jacA')
        print("Done!")
        print("Time = %f s\n"%(time.time() - tstart,))

    def A_jacA_taped(self, XP):
        """
        A + jacobian.
        """
        if 'jacA' not in self.taped:
            self.init_jacA()
        return self.A(XP), self.jacA_taped(XP)

    ################################################################################
    # Minimization functions
    ################################################################################
    def min_lbfgs_scipy(self, XP0):
        """
        Minimize f starting from x0 using L-BFGS-B method in scipy.
        This method supports the use of bounds.
        Returns the minimizing state, the minimum function value, and the L-BFGS
        termination information.
        """
        if 'A_gradA' not in self.taped:
            self.init_A_gradA()

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.minimize(self.A_gradA_taped, XP0, method='L-BFGS-B', jac=True,
                           options=self.opt_args, bounds=self.bounds)
        XPmin,status,Amin = res.x, res.status, res.fun

        print("Optimization complete!")
        print("Time = {0} s".format(time.time()-tstart))
        print("Exit flag = {0}".format(status))
        print("Exit message: {0}".format(res.message))
        print("Iterations = {0}".format(res.nit))
        print("Obj. function value = {0}\n".format(Amin))
        return XPmin, Amin, status

    def min_cg_scipy(self, XP0):
        """
        Minimize f starting from x0 using nonlinear CG method in scipy.
        Returns the minimizing state, the minimum function value, and the CG
        termination information.
        """
        if 'A_gradA' not in self.taped:
            self.init_A_gradA()

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.minimize(self.A_gradA, XP0, method='CG', jac=True,
                           options=self.opt_args)
        XPmin,status,Amin = res.x, res.status, res.fun

        print("Optimization complete!")
        print("Time = {0} s".format(time.time()-tstart))
        print("Exit flag = {0}".format(status))
        print("Exit message: {0}".format(res.message))
        print("Iterations = {0}".format(res.nit))
        print("Obj. function value = {0}\n".format(Amin))
        return XPmin, Amin, status

    def min_tnc_scipy(self, XP0):
        """
        Minimize f starting from x0 using Newton-CG method in scipy.
        Returns the minimizing state, the minimum function value, and the CG
        termination information.
        """
        if 'A_gradA' not in self.taped:
            self.init_A_gradA()

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.minimize(self.A_gradA, XP0, method='TNC', jac=True,
                           options=self.opt_args, bounds=self.bounds)
        XPmin,status,Amin = res.x, res.status, res.fun

        print("Optimization complete!")
        print("Time = {0} s".format(time.time()-tstart))
        print("Exit flag = {0}".format(status))
        print("Exit message: {0}".format(res.message))
        print("Iterations = {0}".format(res.nit))
        print("Obj. function value = {0}\n".format(Amin))
        return XPmin, Amin, status

    def min_lm_scipy(self, XP0):
        """
        Minimize f starting from x0 using Levenberg-Marquardt in scipy.
        Returns the minimizing state, the minimum function value, and the CG
        termination information.
        """
        if 'A_jacA' not in self.taped:
            self.init_A_jacA()

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.root(self.A_jacA_taped, XP0, method='lm', jac=True,
                       options=self.opt_args)

        XPmin,status,Amin = res.x, res.status, res.fun

        print("Optimization complete!")
        print("Time = {0} s".format(time.time()-tstart))
        print("Exit flag = {0}".format(status))
        print("Exit message: {0}".format(res.message))
        print("Iterations = {0}".format(res.nit))
        print("Obj. function value = {0}\n".format(Amin))
        return XPmin, Amin, status
