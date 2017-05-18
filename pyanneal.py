"""
Carry out path space annealing.
"""

import numpy as np
import adolc
import time
import scipy.optimize as opt
try:
    import xalglib
except:
    pass

class TwinExperiment:
    def __init__(self, f, dt, D, Lidx, RM, RF0,
                 stim_file=None, stim=None, data_file=None, Y=None, t=None,
                 N=None, nstart=0, P=(), Pidx=(), adolcID=0):
        self.f = f
        self.dt = dt
        self.D = D
        self.Lidx = Lidx
        self.L = len(Lidx)
        self.RM = RM
        self.RF0 = RF0
        self.P = P
        self.Pidx = Pidx
        self.NP = len(P)
        self.NPest = len(Pidx)
        self.adolcID = adolcID

        # load data
        if data_file is None:
            self.Y = Y
        else:
            self.load_data(data_file)

        # load stim
        if stim_file is None:
            self.stim = stim
        else:
            self.load_stim(stim_file)

        #self.nstart = nstart  # first time index to use from data

        # extract data from nstart to nstart + N
        if N is None:
            self.N = self.Y.shape[0]
        else:
            self.N = N

        self.t = t[nstart:(nstart + N)]
        self.Y = self.Y[nstart:(nstart + N)]
        if self.stim is not None:
            self.stim = self.stim[nstart:(nstart + N)]

        # Reshape RM and RF so that they span the whole time series.  This is
        # done because in the action evaluation, it is more efficient to let
        # numpy handle multiplication over time rather than using python loops.
        if type(self.RM) == np.ndarray:
            if self.RM.shape == (self.L,):
                self.RM = np.resize(self.RM, (self.N, self.L))
            elif self.RM.shape == (self.L, self.L):
                self.RM = np.resize(self.RM, (self.N, self.L, self.L))
            elif self.RM.shape == (self.N, self.L) or \
                 self.RM.shape == (self.N, self.L, self.L):
                pass
            else:
                print("ERROR: RM has an invalid shape. Exiting.")
                exit(1)

        if type(self.RF0) == np.ndarray:
            if self.RF0.shape == (self.D,):
                self.RF0 = np.resize(self.RF0, (self.N - 1, self.D))
            elif self.RF0.shape == (self.D, self.D):
                self.RF0 = np.resize(self.RF0, (self.N - 1, self.D, self.D))
            elif self.RF0.shape == (self.N - 1, self.D) or \
                 self.RF0.shape == (self.N - 1, self.D, self.D):
                pass
            else:
                print("ERROR: RF0 has an invalid shape. Exiting.")
                exit(1)

        # other stuff
        self.taped = False
        self.initalized = False

        # arguments for optimization routine
        #self.opt_args = None

    def load_data(self, data_file):
        """
        Load twin experiment data from file. If a text file, must be in
        multi-column format with L+1 columns:
            t  xmeas_1  xmeas_2  ...  xmeas_L
        If a .npy archive, should contain an N X (L+1) array with times in the
        first element of each entry.
        """
        if data_file.endswith('npy'):
            data = np.load(data_file)
        else:
            data = np.loadtxt(data_file)
        self.Y = data[:, 1:]

    def load_stim(self, stim_file):
        """
        Load stimulus for experimental data from file.
        """
        if stim_file.endswith('npy'):
            s = np.load(stim_file)
        else:
            s = np.loadtxt(stim_file)
        self.stim = s[:, 1:]

    ############################################################################
    # Gaussian action
    ############################################################################
    def A_gaussian_pieced(self, XP):
        """
        Gaussian action.
        """        
        if self.NPest == 0:
            x = np.reshape(XP, (self.N, self.D))
            p = self.P
        elif self.NP == self.NPest:
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

        # evaluate the action
        me = self.me_gaussian(x, p)
        fe = self.fe_gaussian(x, p)
        return me + fe

    # Gaussian action terms for matrix Rf and Rm
    def me_gaussian(self, x, p):
        """
        Gaussian measurement error.
        """
        err = self.me_gaussian_TS(x, p)
        return np.sum(err) / (self.L * self.N)

    def fe_gaussian(self, x, p):
        """
        Gaussian model error.
        """
        err = self.fe_gaussian_TS(x, p)
        return np.sum(err) / (self.D * (self.N - 1))

    # Time series of squared errors, times RM or RF.
    def me_gaussian_TS(self, x, p):
        """
        Time series of squared measurement errors, times RM.
        """
        diff = self.me_gaussian_TS_vec(x, p)

        if type(self.RM) == np.ndarray:
            if self.RM.shape == (self.N, self.L):
                err = self.RM * diff * diff
            elif self.RM.shape == (self.L, self.L):
                for diffn in diff:
                    err[i] = np.dot(diffn, np.dot(self.RM, diffn))
            elif self.RM.shape == (self.N, self.L, self.L):
                for diffn,RMn in zip(diff, self._RM):
                    err[i] = np.dot(diffn, np.dot(RMn, diffn))
            else:
                print("ERROR: RM is in an invalid shape.")
        else:
            err = self.RM * diff * diff

        return err

    def fe_gaussian_TS(self, x, p):
        """
        Time series of squared model errors, times RF.
        """
        diff = self.fe_gaussian_TS_vec(x, p)

        if type(self.RF) == np.ndarray:
            if self.RF.shape == (self.D,):
                err = np.zeros(self.N - 1, dtype=diff.dtype)
                for i in xrange(self.N - 1):
                    err[i] = np.sum(self.RF * diff[i] * diff[i])

            elif self.RF.shape == (self.N - 1, self.D):
                err = self.RF * diff * diff

            elif self.RF.shape == (self.D, self.D):
                err = np.zeros(self.N - 1, dtype=diff.dtype)
                for i in xrange(self.N - 1):
                    err[i] = np.dot(diff[i], np.dot(self.RF, diff[i]))

            elif self.RF.shape == (self.N - 1, self.D, self.D):
                err = np.zeros(self.N - 1, dtype=diff.dtype)
                for i in xrange(self.N - 1):
                    err[i] = np.dot(diff[i], np.dot(RF[i], diff[i]))

            else:
                print("ERROR: RF is in an invalid shape.")

        else:
            err = self.RF * diff * diff

        return err

    # Time series of error vectors.
    def me_gaussian_TS_vec(self, x, p):
        """
        Time series of measurement error vectors, NOT times RM.
        """
        if x.ndim == 1:
            x = np.reshape(x, (self.N, self.D))
        diff = x[:, self.Lidx] - self.Y

        return diff

    def fe_gaussian_TS_vec(self, x, p):
        """
        Time series of model error vectors, NOT times RF.
        """
        if x.ndim == 1:
            x = np.reshape(x, (self.N, self.D))

        if self.disc.im_func.__name__ == "disc_SimpsonHermite":
            disc_vec = self.disc(x, p)
            #diff = np.zeros((self.N - 1, self.D), dtype="object")
            diff = np.zeros((self.N - 1, self.D), dtype=x.dtype)
            diff[:-1:2] = x[2::2] - x[:-2:2] - disc_vec[:-1:2]
            diff[1::2] = x[1::2] - disc_vec[1::2]
        else:
            diff = x[1:] - x[:-1] - self.disc(x, p)

        return diff

    def vecA_gaussian(self, XP):
        """
        Vector-like terms of the Gaussian action.
        This only works with scalar RM and RF!!!
        """
        if self.NPest == 0:
            x = np.reshape(XP, (self.N, self.D))
            p = self.P
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

        # evaluate the vector-like terms of the action
        me_vec = np.sqrt(self.RM / (self.L * self.N)) \
                 * self.me_gaussian_TS_vec(x, p)
        fe_vec = np.sqrt(self.RF / (self.D * (self.N - 1))) \
                 * self.fe_gaussian_TS_vec(x, p)

        return np.append(me_vec, fe_vec)

    def A_gaussian(self, XP):
        """
        Calculate the Gaussian action all in one go.
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

        # Measurement error
        diff = x[:, self.Lidx] - self.Y

        if type(self.RM) == np.ndarray:
            # Contract RM with error
            if self.RM.shape == (self.N, self.L):
                merr = np.sum(self.RM * diff * diff)
            elif self.RM.shape == (self.N, self.L, self.L):
                merr = 0.0
                for i in xrange(self.N):
                    merr = merr + np.dot(diff[i], np.dot(self.RM[i], diff[i]))
            else:
                print("ERROR: RM is in an invalid shape.")
        else:
            merr = self.RM * np.sum(diff * diff)

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
                    ferr1 = np.sum(self.RF[::2] * diff1 * diff1)
                    ferr2 = np.sum(self.RF[1::2] * diff2 * diff2)
                    ferr = ferr1 + ferr2
                else:
                    ferr = np.sum(self.RF * diff * diff)

            elif self.RF.shape == (self.N - 1, self.D, self.D):
                if self.disc.im_func.__name__ == "disc_SimpsonHermite":
                    ferr1 = 0.0
                    ferr2 = 0.0
                    for i in xrange((self.N - 1) / 2):
                        ferr1 = ferr1 + np.dot(diff1[i], np.dot(self.RF[2*i], diff1[i]))
                        ferr2 = ferr2 + np.dot(diff2[i], np.dot(self.RF[2*i+1], diff2[i]))
                    ferr = ferr1 + ferr2
                else:
                    ferr = 0.0
                    for i in xrange(self.N - 1):
                        ferr = ferr + np.dot(diff[i], np.dot(self.RF[i], diff))

            else:
                print("ERROR: RF is in an invalid shape.")

        else:
            if self.disc.im_func.__name__ == "disc_SimpsonHermite":
                ferr1 = self.RF * np.sum(diff1 * diff1)
                ferr2 = self.RF * np.sum(diff2 * diff2)
                ferr = ferr1 + ferr2
            else:
                ferr = self.RF * np.sum(diff * diff)

        return merr/(self.L*self.N) + ferr/(self.D*(self.N-1))

    ############################################################################
    # Discretization routines
    ############################################################################
    def disc_trapezoid(self, x, p):
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
        xn = x[:-1]
        tn = np.tile(self.t[:-1], (self.D, 1)).T
        k1 = self.f(tn, xn, p)
        k2 = self.f(tn + self.dt/2.0, xn + k1*self.dt/2.0, p)
        k3 = self.f(tn + self.dt/2.0, xn + k2*self.dt/2.0, p)
        k4 = self.f(tn + self.dt, xn + k3*self.dt, p)
        return self.dt * (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

    def disc_SimpsonHermite(self, x, p):
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
    def anneal_init(self, XP0, alpha, beta_array, RF0=None, bounds=None,
                    init_to_data=True, method='L-BFGS', disc='trapezoid',
                    action='A_gaussian'):
        """
        Initialize the annealing procedure.
        """
        if method not in ('L-BFGS', 'NCG', 'LM', 'LM_FGH', 'L-BFGS-B', 'TNC'):
            print("ERROR: Optimization routine not implemented or recognized.")
            return 1
        else:
            self.method = method

        self.initalized = True  # indicates we're at the first annealing step

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

        # set up bet array in RF = RF0 * alpha**beta
        self.alpha = alpha
        self.beta_array = beta_array
        self.betaidx = 0
        self.beta = self.beta_array[self.betaidx]
        self.Nbeta = len(self.beta_array)

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

        # set current RF
        if RF0 is not None:
            self.RF0 = RF0
        self.RF = self.RF0 * self.alpha**self.beta

        # array to store minimum action values
        self.A_array = np.zeros(self.Nbeta, dtype='float')
        self.me_array = np.zeros(self.Nbeta, dtype='float')
        self.fe_array = np.zeros(self.Nbeta, dtype='float')

        # Store optimization bounds. Will only be used if the chosen
        # optimization routine supports it.
        if bounds is None:
            self.bounds = bounds
        else:
            self.bounds = np.array(bounds)

        # array to store optimization exit flags
        self.exitflags = np.empty(self.Nbeta, dtype='int')

    def set_disc(self, disc):
        exec 'self.disc = self.disc_%s'%(disc,)

    def anneal_step(self):
        """
        Perform a single annealing step. The cost function is minimized starting
        from the previous minimum (or the initial guess, if this is the first
        step). Then, RF is increased to prepare for the next annealing step.
        """
        # minimize A using the chosen method
        #if self.method == 'L-BFGS':
        #    if self.betaidx == 0:
        #        XPmin, Amin, rep = self.min_lbfgs(self.minpaths[0])
        #    else:
        #        XPmin, Amin, rep = self.min_lbfgs(self.minpaths[self.betaidx-1])
        #    self.exitflags[self.betaidx] = rep.terminationtype

        #elif self.method == 'NCG':
        #    if self.betaidx == 0:
        #        XPmin, Amin, rep = self.min_ncg(self.minpaths[0])
        #    else:
        #        XPmin, Amin, rep = self.min_ncg(self.minpaths[self.betaidx-1])
        #    self.exitflags[self.betaidx] = rep.terminationtype

        #elif self.method == 'LM':
        #    if self.betaidx == 0:
        #        XPmin, Amin, rep = self.min_lm(self.minpaths[0])
        #    else:
        #        XPmin, Amin, rep = self.min_lm(self.minpaths[self.betaidx-1])
        #    self.exitinfo.append[rep]
        #    self.exitflags[self.betaidx] = rep.terminationtype
        #
        #elif self.method == 'LM_FGH':
        #    if self.betaidx == 0:
        #        XPmin, Amin, rep = self.min_lm_FGH(self.minpaths[0])
        #    else:
        #        XPmin, Amin, rep = self.min_lm_FGH(self.minpaths[self.betaidx-1])
        #    self.exitinfo.append[rep]
        #    self.exitflags[self.betaidx] = rep.terminationtype

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

        else:
            print("ERROR: Optimization routine not implemented or recognized.")

        # update optimal parameter values
        if self.NPest > 0:
            if isinstance(XPmin[0], adolc._adolc.adouble):
                self.P[self.Pidx] = np.array([XPmin[-self.NPest + i].val for i in xrange(self.NPest)])
            else:
                self.P[self.Pidx] = np.copy(XPmin[-self.NPest:])

        # store A_min and the minimizing path
        self.A_array[self.betaidx] = Amin
        self.me_array[self.betaidx] = self.me_gaussian(np.array(XPmin[:self.N*self.D]), self.P)
        self.fe_array[self.betaidx] = self.fe_gaussian(np.array(XPmin[:self.N*self.D]), self.P)
        self.minpaths[self.betaidx] = np.array(XPmin)

        # increase RF
        if self.betaidx < len(self.beta_array) - 1:
            self.betaidx += 1
            self.beta = self.beta_array[self.betaidx]
            self.RF = self.RF0 * self.alpha**self.beta

        # set flags indicating that A needs to be retaped, and that we're no
        # longer at the beginning of the annealing procedure
        self.taped = False
        self.initalized = False

    def anneal(self, XP0, alpha, beta_array, RF0=None, bounds=None,
               init_to_data=True, method='L-BFGS', disc='trapezoid',
               opt_args=None):
        """
        Convenience function to carry out a full annealing run over all values
        of beta in beta_array.
        """
        # get optimization extra arguments
        self.opt_args = opt_args

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

        # initialize the annealing procedure, if not already done
        if self.initalized == False:
            self.anneal_init(XP0, alpha, beta_array, RF0, bounds_full, init_to_data, method, disc)
        for i in beta_array:
            print('------------------------------')
            print('Step %d of %d'%(self.betaidx+1,len(self.beta_array)))
            print('alpha = %f, beta = %f'%(self.alpha,self.beta))
            print('')
            self.anneal_step()

    def save_as_minAone(self, savedir='', savefile=None):
        """
        Save the result of this annealing in minAone data file style.
        """
        if savedir.endswith('/') == False:
            savedir += '/'
        if savefile is None:
            savefile = savedir + 'D%d_M%d_PATH%d.dat'%(self.D, self.L, self.adolcID)
        else:
            savefile = savedir + savefile
        betaR = self.beta_array.reshape((self.Nbeta,1))
        exitR = self.exitflags.reshape((self.Nbeta,1))
        AR = self.A_array.reshape((self.Nbeta,1))
        savearray = np.hstack((betaR, exitR, AR, self.minpaths))
        np.savetxt(savefile, savearray)

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

    ############################################################################
    # AD and minimization functions
    ############################################################################
    #def alglib_lbfgs_A(self, XP, grad_A, param=None):
    #    """
    #    ALGLIB-acceptable action for the L-BFGS algorithm.
    #    Returns A, but sets grad by reference.
    #    """
    #    grad_A[:] = adolc.gradient(self.adolcID, XP)
    #    return adolc.function(self.adolcID, XP)
    #
    #def alglib_lm_vecA(self, XP, fi, param=None):
    #    """
    #    ALGLIB-acceptable function which sets the individual "vector-like"
    #    terms of the action, for Levenburg-Marquardt.
    #    """
    #    fi[:] = adolc.function(self.adolcID, XP)
    #
    #def alglib_lm_vecA_jac(self, XP, fi, jac, param=None):
    #    """
    #    ALGLIB-acceptable function which sets the Jacobian of the individual
    #    "vector-like" terms of the action, for Levenburg-Marquardt.
    #    """
    #    fi[:] = adolc.function(self.adolcID, XP)
    #    jac[:] = adolc.jacobian(self.adolcID, XP).tolist()
    #
    #def alglib_lm_FGH_A(self, XP, param=None):
    #    """
    #    ALGLIB-acceptable action for the Levenburg-Marquardt algorithm.
    #    """
    #    return adolc.function(self.adolcID, XP)
    #
    #def alglib_lm_A_FGH_grad(self, XP, grad_A, param=None):
    #    """
    #    ALGLIB-acceptable action gradient for the Levenburg-Marquardt algorithm.
    #    Sets gradient by reference.
    #    """
    #    grad_A[:] = adolc.gradient(self.adolcID, XP)
    #    return adolc.function(self.adolcID, XP)
    #
    #def alglib_lm_A_FGH_hess(self, XP, grad_A, hess_A, param=None):
    #    """
    #    ALGLIB-acceptable action hessian for the Levenburg-Marquardt algorithm.
    #    Sets gradient and hessian by reference.
    #    """
    #    grad_A[:] = adolc.gradient(self.adolcID, XP)
    #    hess_A[:] = adolc.hessian(self.adolcID, XP).tolist()
    #    return adolc.function(self.adolcID, XP)

    def scipy_A(self, XP):
        return adolc.function(self.adolcID, XP)
    
    def scipy_A_grad(self, XP):
        return adolc.gradient(self.adolcID, XP)

    def scipy_A_plusgrad(self, XP):
        return adolc.function(self.adolcID, XP), adolc.gradient(self.adolcID, XP)

    def hessian_eval(self, XP):
        return adolc.hessian(self.adolcID, XP)

    def tape_A(self):
        """
        Tape the objective function.
        """
        print('Taping action evaluation...')
        tstart = time.time()
        # define a random state vector for the trace
        xtrace = np.random.rand(self.D*self.N + self.NPest)
        adolc.trace_on(self.adolcID)
        # set the active independent variables
        ax = adolc.adouble(xtrace)
        adolc.independent(ax)
        # set the dependent variable (or vector of dependent variables)
        af = self.A(ax)
        adolc.dependent(af)
        adolc.trace_off()
        self.taped = True
        print('Done!')
        print('Time = {0} s\n'.format(time.time()-tstart))

    #def min_lbfgs(self, XP0, epsg=1e-8, epsf=1e-8, epsx=1e-8, maxits=10000):
    #    """
    #    Minimize f starting from x0 using L-BFGS.
    #    Returns the minimizing state, the minimum function value, and the L-BFGS
    #    termination information.
    #    """
    #    if self.taped == False:
    #        self.tape_A()
    #
    #    # start the optimization
    #    print('Beginning optimization...')
    #    tstart = time.time()
    #    # initialize the L-BFGS optimization
    #    state = xalglib.minlbfgscreate(5, list(XP0.flatten()))
    #    xalglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits)
    #    # run the optimization
    #    xalglib.minlbfgsoptimize_g(state, self.alglib_lbfgs_A)
    #    # store the result of the optimization
    #    XPmin,rep = xalglib.minlbfgsresults(state)
    #    Amin = self.A(XPmin)
    #
    #    print('Optimization complete!')
    #    print('Time = {0} s'.format(time.time()-tstart))
    #    print('Exit flag = {0}'.format(rep.terminationtype))
    #    print('Iterations = {0}'.format(rep.iterationscount))
    #    print('Obj. function value = {0}\n'.format(Amin))
    #    return XPmin, Amin, rep
    #
    #def min_ncg(self, XP0, epsg=1e-8, epsf=1e-8, epsx=1e-8, maxits=10000):
    #    """
    #    Minimize f starting from x0 using NCG.
    #    Returns the minimizing state, the minimum function value, and the L-BFGS
    #    termination information.
    #    """
    #    if self.taped == False:
    #        self.tape_A()
    #
    #    # start the optimization
    #    print('Beginning optimization...')
    #    tstart = time.time()
    #    # initialize the L-BFGS optimization
    #    state = xalglib.mincgcreate(list(XP0.flatten()))
    #    xalglib.mincgsetcond(state, epsg, epsf, epsx, maxits)
    #    # run the optimization
    #    xalglib.mincgoptimize_g(state, self.alglib_lbfgs_A)
    #    # store the result of the optimization
    #    XPmin,rep = xalglib.mincgresults(state)
    #    Amin = self.A(XPmin)
    #
    #    print('Optimization complete!')
    #    print('Time = {0} s'.format(time.time()-tstart))
    #    print('Exit flag = {0}'.format(rep.terminationtype))
    #    print('Iterations = {0}'.format(rep.iterationscount))
    #    print('Obj. function value = {0}\n'.format(Amin))
    #    return XPmin, Amin, rep
    #
    #def min_lm(self, XP0, epsg=1e-8, epsf=1e-8, epsx=1e-8, maxits=10000):
    #    """
    #    Minimize the action starting from XP0, using the Levenburg-Marquardt method.
    #    This method supports the use of bounds.
    #    The vector-like structure used in the definition of the action is
    #    exploited here.
    #    """
    #    if self.taped == False:
    #        self.tape_A()
    #
    #    # start the optimization
    #    print('Beginning optimization...')
    #    tstart = time.time()
    #    # initialize the L-BFGS optimization
    #    Nf = (self.N - 1) * self.D + self.N * self.L
    #    Nx = self.N * self.D + self.NPest
    #    state = xalglib.minlmcreatevj(Nx , Nf, list(XP0.flatten()))
    #    xalglib.minlmsetcond(state, epsg, epsf, epsx, maxits)
    #    # set optimization bounds
    #    if self.bounds is not None:
    #        bndl, bndu = self.bounds[:,0], self.bounds[:,1]
    #        xalglib.minlmsetbc(state, bndl, bndu)
    #    # run the optimization
    #    xalglib.minlmoptimize_vj(state, self.alglib_lm_vecA, self.alglib_lm_vecA_jac)
    #    # store the result of the optimization
    #    XPmin, rep = xalglib.minlmresults(state)
    #    Amin = self.A(XPmin)
    #
    #    print('Optimization complete!')
    #    print('Time = {0} s'.format(time.time()-tstart))
    #    print('Exit flag = {0}'.format(rep.terminationtype))
    #    print('Iterations = {0}'.format(rep.iterationscount))
    #    print('Obj. function value = {0}\n'.format(Amin))
    #    return XPmin, Amin, rep
    #
    #def min_lm_FGH(self, XP0, epsg=1e-8, epsf=1e-8, epsx=1e-8, maxits=10000):
    #    """
    #    Minimize the action starting from XP0, using the Levenburg-Marquardt method.
    #    This method supports the use of bounds.
    #    This is for a general action function, NOT using the vector-like
    #    substructure in its definition.
    #    """
    #    if self.taped == False:
    #        self.tape_A()
    #
    #    # start the optimization
    #    print("Beginning optimization...")
    #    tstart = time.time()
    #    # initialize the LM algorithm
    #    state = xalglib.minlmcreatefgh(list(XP0.flatten()))
    #    xalglib.minlmsetcond(state, epsg, epsf, epsx, maxits)
    #    # set optimization bounds
    #    if self.bounds is not None:
    #        bndl,bndu = self.bounds[:,0], self.bounds[:,1]
    #        xalglib.minlmsetbc(state, bndl, bndu)
    #    # run the optimization
    #    xalglib.minlmoptimize_fgh(state, self.alglib_LM_A, self.alglib_LM_A_grad,
    #                              self.alglib_LM_A_hess)
    #    # store the result
    #    XPmin,rep = xalglib.minlmresults(state)
    #    Amin = self.A(XPmin)
    #
    #    print('Optimization complete!')
    #    print('Time = {0} s'.format(time.time()-tstart))
    #    print('Exit flag = {0}'.format(rep.terminationtype))
    #    print('Iterations = {0}'.format(rep.iterationscount))
    #    print('Obj. function value = {0}\n'.format(Amin))
    #    return XPmin, Amin, rep

    def min_lbfgs_scipy(self, XP0):
        """
        Minimize f starting from x0 using L-BFGS-B method in scipy.
        This method supports the use of bounds.
        Returns the minimizing state, the minimum function value, and the L-BFGS
        termination information.
        """
        if self.taped == False:
            self.tape_A()

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.minimize(self.scipy_A_plusgrad, XP0, method='L-BFGS-B', jac=True,
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
        if self.taped == False:
            self.tape_A()

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.minimize(self.scipy_A_plusgrad, XP0, method='CG', jac=True,
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
        if self.taped == False:
            self.tape_A()

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.minimize(self.scipy_A_plusgrad, XP0, method='TNC', jac=True,
                           options=self.opt_args, bounds=self.bounds)
        XPmin,status,Amin = res.x, res.status, res.fun

        print("Optimization complete!")
        print("Time = {0} s".format(time.time()-tstart))
        print("Exit flag = {0}".format(status))
        print("Exit message: {0}".format(res.message))
        print("Iterations = {0}".format(res.nit))
        print("Obj. function value = {0}\n".format(Amin))
        return XPmin, Amin, status

    ############################################################################
    # Class properties
    ############################################################################
