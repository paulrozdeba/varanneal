"""
Carry out path space annealing.
"""

import numpy as np
import xalglib, adolc
import time
import scipy.optimize as opt

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
        self.t = t
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

        if N is None:
            self.N = self.Y.shape[0]
        else:
            self.N = N

        self.nstart = nstart  # first time index to use from data

        # load stim
        if stim_file is None:
            self.stim = stim
        else:
            self.load_stim(stim_file)

        # other stuff
        self.A = self.A_gaussian
        self.taped = False
        self.initalized = False

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
        self.Y = data[:,1:]

    #def load_stim(self, stim_file):
    #    """
    #    Load stimulus for experimental data from file.
    #    """

    ############################################################################
    # Gaussian action
    ############################################################################
    def A_gaussian(self, XP):
        """
        Gaussian action.
        """
        if self.NPest == 0:
            x = np.reshape(XP, (self.N, self.D))
            p = self.P
        else:
            x = np.reshape(XP[:-self.NPest], (self.N, self.D))
            if isinstance(XP[0], adolc._adolc.adouble):
                #self.P[self.Pidx] = np.array([XP[-self.NPest + i].val for i in range(self.NPest)])
                p = []
                j = self.NPest
                for i in range(self.NP):
                    if i in self.Pidx:
                        p.append(XP[-j])
                        j -= 1
                    else:
                        p.append(self.P[i])
            else:
                #self.P[self.Pidx] = XP[-self.NPest:]
                #p = np.copy(self.P)
                p = np.copy(XP[-self.NPest:])

        # evaluate the action
        me = self.me_gaussian(x, p)
        #if self.disc.im_func.__name__ == "disc_SimpsonHermite":
        #    fe = self.fe_SimpsonHermite(x, p)
        #else:
        fe = self.fe_gaussian(x, p)
        return me + fe

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
            if isinstance(XP[0], adolc._adolc.adouble):
                #self.P[self.Pidx] = np.array([XP[-self.NPest + i].val for i in range(self.NPest)])
                p = []
                j = self.NPest
                for i in range(self.NP):
                    if i in self.Pidx:
                        p.append(XP[-j])
                        j -= 1
                    else:
                        p.append(self.P[i])
            else:
                #self.P[self.Pidx] = XP[-self.NPest:]
                #p = np.copy(self.P)
                p = np.copy(XP[-self.NPest:])

        # evaluate the vector-like terms of the action
        me_vec = np.sqrt(self.RM / (self.L * self.N)) \
                 * self.me_gaussian_TS_vec(x, p)

        if self.disc.im_func.__name__ == "disc_SimpsonHermite":
            fe_vec = np.sqrt(self.RF / (self.D * (self.N - 1))) \
                 * self.fe_SimpsonHermite_TS_vec(x, p)
        else:
            fe_vec = np.sqrt(self.RF / (self.D * (self.N - 1))) \
                 * self.fe_gaussian_TS_vec(x, p)

        return np.append(me_vec, fe_vec)

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
            diff = np.zeros((self.N - 1, self.D), dtype="object")
            diff[:-1:2] = x[2::2] - x[:-2:2] - disc_vec[:-1:2]
            diff[1::2] = x[1::2] - disc_vec[1::2]
        else:
            diff = x[1:] - x[:-1] - self.disc(x, p)

        return diff

    #def fe_SimpsonHermite_TS_vec(self, x, p):
    #    """
    #    Time series of model error vectors, NOT times RF.
    #    """
    #    if x.ndim == 1:
    #        x = np.reshape(x, (self.N, self.D))
    #    xn = x[:-2:2]
    #    xmid = x[1:-1:2]
    #    xnp1 = x[2::2]
    #
    #    tn = self.t[:-2:2]
    #    tmid = self.t[1:-1:2]
    #    tnp1 = self.t[2::2]
    #
    #    fn = self.f(xn, tn, p)
    #    fmid = self.f(xmid, tmid, p)
    #    fnp1 = self.f(xnp1, tnp1, p)
    #
    #    diff = np.empty((self.N - 1, self.D), dtype="object")
    #
    #    diff[:-1:2] = xnp1 - xn - (fn + 4.0*fmid + fnp1) * (2.0*self.dt)/6.0
    #    diff[1::2] = xmid - (xn + xnp1)/2.0 - (fn - fnp1) * (2.0*self.dt)/8.0
    #
    #    return diff

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
            if self.RF.shape == (self.N - 1, self.D):
                err = self.RF * diff * diff
            elif self.RF.shape == (self.D, self.D):
                for diffn in diff:
                    err[i] = np.dot(diff, self.RF, diff)
            elif self.RF.shape == (self.N - 1, self.D, self.D):
                for diffn,RFn in zip(diff, self.RF):
                    err[i] = np.dot(diffn, np.dot(RFn, diffn))
            else:
                print("ERROR: RF is in an invalid shape.")
        else:
            err = self.RF * diff * diff

        return err

    #def fe_SimpsonHermite_TS(self, x, p):
    #    """
    #    Time series of squared model errors, times RF.
    #    """
    #    diff = self.fe_SimpsonHermite_TS_vec(x, p)
    #
    #    if type(self.RF) == np.ndarray:
    #        if self.RF.shape == (self.N - 1, self.D):
    #            err = self.RF * diff * diff
    #        elif self.RF.shape == (self.D, self.D):
    #            for diffn in diff:
    #                err[i] = np.dot(diff, self.RF, diff)
    #        elif self.RF.shape == (self.N - 1, self.D, self.D):
    #            for diffn,RFn in zip(diff, self.RF):
    #                err[i] = np.dot(diffn, np.dot(RFn, diffn))
    #        else:
    #            print("ERROR: RF is in an invalid shape.")
    #    else:
    #        err = self.RF * diff * diff
    #
    #    return err

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

    #def fe_SimpsonHermite(self, x, p):
    #    """
    #    Simpson-Hermite model error.
    #    """
    #    err = self.fe_SimpsonHermite_TS(x, p)
    #    return np.sum(err) / (self.D * (self.N - 1))

    ############################################################################
    # Discretization routines
    ############################################################################
    def disc_trapezoid(self, x, p):
        fn = self.f(x[:-1], self.t[:-1], p)
        fnp1 = self.f(x[1:], self.t[1:], p)
        return self.dt * (fn + fnp1) / 2.0

    def disc_rk4(self, x, p):
        xn = x[:-1]
        tn = np.tile(self.t[:-1], (self.D, 1)).T
        k1 = self.f(xn, tn, p)
        k2 = self.f(xn + k1*self.dt/2.0, tn + self.dt/2.0, p)
        k3 = self.f(xn + k2*self.dt/2.0, tn + self.dt/2.0, p)
        k4 = self.f(xn + k3*self.dt, tn + self.dt, p)
        return self.dt * (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

    def disc_SimpsonHermite(self, x, p):
        xn = x[:-2:2]
        xmid = x[1:-1:2]
        xnp1 = x[2::2]

        tn = self.t[:-2:2]
        tmid = self.t[1:-1:2]
        tnp1 = self.t[2::2]

        fn = self.f(xn, tn, p)
        fmid = self.f(xmid, tmid, p)
        fnp1 = self.f(xnp1, tnp1, p)

        disc_vec = np.empty((self.N - 1, self.D), dtype="object")
        disc_vec[:-1:2] = (fn + 4.0*fmid + fnp1) * (2.0*self.dt)/6.0
        disc_vec[1::2] = (xn + xnp1)/2.0 + (fn - fnp1) * (2.0*self.dt)/8.0

        return disc_vec

    ############################################################################
    # Annealing functions
    ############################################################################
    def anneal_init(self, XP0, alpha, beta_array, RF0=None, bounds=None,
                    init_to_data=True, method='L-BFGS', disc='trapezoid'):
        """
        Initialize the annealing procedure.
        """
        if method not in ('L-BFGS', 'NCG', 'LM', 'LM_FGH', 'L-BFGS-B'):
            print("ERROR: Optimization routine not implemented or recognized.")
            return 1

        self.initalized = True  # indicates we're at the first annealing step

        exec 'self.disc = self.disc_%s'%(disc,)

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

        self.method = method

        if self.method == 'LM':
            self.A = self.vecA_gaussian

        # array to store optimization exit flags
        self.exitflags = np.empty(self.Nbeta, dtype='int')

    def anneal_step(self):
        """
        Perform a single annealing step. The cost function is minimized starting
        from the previous minimum (or the initial guess, if this is the first
        step). Then, RF is increased to prepare for the next annealing step.
        """
        # minimize A using the chosen method
        if self.method == 'L-BFGS':
            if self.betaidx == 0:
                XPmin, Amin, rep = self.min_lbfgs(self.minpaths[0])
            else:
                XPmin, Amin, rep = self.min_lbfgs(self.minpaths[self.betaidx-1])
            self.exitflags[self.betaidx] = rep.terminationtype

        elif self.method == 'NCG':
            if self.betaidx == 0:
                XPmin, Amin, rep = self.min_ncg(self.minpaths[0])
            else:
                XPmin, Amin, rep = self.min_ncg(self.minpaths[self.betaidx-1])
            self.exitflags[self.betaidx] = rep.terminationtype

        elif self.method == 'LM':
            if self.betaidx == 0:
                XPmin, Amin, rep = self.min_lm(self.minpaths[0])
            else:
                XPmin, Amin, rep = self.min_lm(self.minpaths[self.betaidx-1])
            self.exitinfo.append[rep]
            self.exitflags[self.betaidx] = rep.terminationtype

        elif self.method == 'LM_FGH':
            if self.betaidx == 0:
                XPmin, Amin, rep = self.min_lm_FGH(self.minpaths[0])
            else:
                XPmin, Amin, rep = self.min_lm_FGH(self.minpaths[self.betaidx-1])
            self.exitinfo.append[rep]
            self.exitflags[self.betaidx] = rep.terminationtype

        elif self.method == 'L-BFGS-B':
            if self.betaidx == 0:
                XPmin, Amin, exitflag = self.min_lbfgs_scipy(self.minpaths[0])
            else:
                XPmin, Amin, exitflag = self.min_lbfgs_scipy(self.minpaths[self.betaidx-1])

        else:
            print("ERROR: Optimization routine not implemented or recognized.")

        # update optimal parameter values
        if self.NPest > 0:
            if isinstance(XPmin[0], adolc._adolc.adouble):
                self.P[self.Pidx] = np.array([XPmin[-self.NPest + i].val for i in range(self.NPest)])
            else:
                self.P[self.Pidx] = np.copy(XPmin[-self.NPest:])

        # store A_min and the minimizing path
        self.A_array[self.betaidx] = Amin
        self.me_array[self.betaidx] = self.me_gaussian(np.array(XPmin[:self.N*self.D]), self.P)
        #if self.disc.im_func.__name__ == "disc_SimpsonHermite":
        #    self.fe_array[self.betaidx] \
        #        = self.fe_SimpsonHermite(np.array(XPmin[:self.N*self.D]), self.P)
        #else:
        self.fe_array[self.betaidx] = self.fe_gaussian(np.array(XPmin[:self.N*self.D]), self.P)
        #if self.betaidx == 0:
        #    self.minpaths[0] = np.array(XPmin)
        #else:
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
               init_to_data=True, method='L-BFGS', disc='trapezoid'):
        """
        Convenience function to carry out a full annealing run over all values
        of beta in beta_array.
        """
        # initialize the annealing procedure, if not already done
        if self.initalized == False:
            self.anneal_init(XP0, alpha, beta_array, RF0, bounds, init_to_data, method, disc)
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
        savearray = np.reshape(self.minpaths[:, :-self.NPest], (self.Nbeta, self.N, self.D))
        
        if filename.endswith('.npy'):
            np.save(savefile, savearray)
        else:
            np.savetxt(savefile, savearray)

    def save_params(self, filename):
        """
        Save minimum action parameter values.
        """
        # write fixed parameters to array
        savearray = np.resize(self.P, (self.Nbeta, self.NP))
        # write estimated parameters to array
        est_param_array = np.reshape(self.minpaths[:, -self.NPest:], (self.Nbeta, self.NPest))
        savearray[:, self.Pidx] = est_param_array

        if filename.endswith('.npy'):
            np.save(savefile, savearray)
        else:
            np.savetxt(savefile, savearray)

    def save_action_errors(self, filename):
        """
        Save beta values, action, and errors (with/without RM and RF) to file.
        """
        savearray = np.zeros((self.Nbeta, 5))
        savearray[:, 0] = self.beta
        savearray[:, 1] = self.A_array
        savearray[:, 2] = self.me_array
        savearray[:, 3] = self.fe_array
        savearray[:, 5] = self.fe_array / (self.RF0 * self.alpha**self.beta)

        if filename.endswith('.npy'):
            np.save(savefile, savearray)
        else:
            np.savetxt(savefile, savearray)

    ############################################################################
    # AD and minimization functions
    ############################################################################
    def alglib_lbfgs_A(self, XP, grad_A, param=None):
        """
        ALGLIB-acceptable action for the L-BFGS algorithm.
        Returns A, but sets grad by reference.
        """
        grad_A[:] = adolc.gradient(self.adolcID, XP)
        return adolc.function(self.adolcID, XP)

    def alglib_lm_vecA(self, XP, fi, param=None):
        """
        ALGLIB-acceptable function which sets the individual "vector-like"
        terms of the action, for Levenburg-Marquardt.
        """
        fi[:] = adolc.function(self.adolcID, XP)

    def alglib_lm_vecA_jac(self, XP, fi, jac, param=None):
        """
        ALGLIB-acceptable function which sets the Jacobian of the individual
        "vector-like" terms of the action, for Levenburg-Marquardt.
        """
        fi[:] = adolc.function(self.adolcID, XP)
        jac[:] = adolc.jacobian(self.adolcID, XP).tolist()

    def alglib_lm_FGH_A(self, XP, param=None):
        """
        ALGLIB-acceptable action for the Levenburg-Marquardt algorithm.
        """
        return adolc.function(self.adolcID, XP)

    def alglib_lm_A_FGH_grad(self, XP, grad_A, param=None):
        """
        ALGLIB-acceptable action gradient for the Levenburg-Marquardt algorithm.
        Sets gradient by reference.
        """
        grad_A[:] = adolc.gradient(self.adolcID, XP)
        return adolc.function(self.adolcID, XP)

    def alglib_lm_A_FGH_hess(self, XP, grad_A, hess_A, param=None):
        """
        ALGLIB-acceptable action hessian for the Levenburg-Marquardt algorithm.
        Sets gradient and hessian by reference.
        """
        grad_A[:] = adolc.gradient(self.adolcID, XP)
        hess_A[:] = adolc.hessian(self.adolcID, XP).tolist()
        return adolc.function(self.adolcID, XP)

    def scipy_A_grad(self, XP):
        return adolc.gradient(self.adolcID, XP)

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
        print('Time = {0} s\n'.format(time.time()-tstart))

    def min_lbfgs(self, XP0, epsg=1e-8, epsf=1e-8, epsx=1e-8, maxits=10000):
        """
        Minimize f starting from x0 using L-BFGS.
        Returns the minimizing state, the minimum function value, and the L-BFGS
        termination information.
        """
        if self.taped == False:
            self.tape_A()

        # start the optimization
        print('Beginning optimization...')
        tstart = time.time()
        # initialize the L-BFGS optimization
        state = xalglib.minlbfgscreate(5, list(XP0.flatten()))
        xalglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits)
        # run the optimization
        xalglib.minlbfgsoptimize_g(state, self.alglib_lbfgs_A)
        # store the result of the optimization
        XPmin,rep = xalglib.minlbfgsresults(state)
        Amin = self.A(XPmin)

        print('Optimization complete!')
        print('Time = {0} s'.format(time.time()-tstart))
        print('Exit flag = {0}'.format(rep.terminationtype))
        print('Iterations = {0}'.format(rep.iterationscount))
        print('Obj. function value = {0}\n'.format(Amin))
        return XPmin, Amin, rep

    def min_ncg(self, XP0, epsg=1e-8, epsf=1e-8, epsx=1e-8, maxits=10000):
        """
        Minimize f starting from x0 using NCG.
        Returns the minimizing state, the minimum function value, and the L-BFGS
        termination information.
        """
        if self.taped == False:
            self.tape_A()

        # start the optimization
        print('Beginning optimization...')
        tstart = time.time()
        # initialize the L-BFGS optimization
        state = xalglib.mincgcreate(list(XP0.flatten()))
        xalglib.mincgsetcond(state, epsg, epsf, epsx, maxits)
        # run the optimization
        xalglib.mincgoptimize_g(state, self.alglib_lbfgs_A)
        # store the result of the optimization
        XPmin,rep = xalglib.mincgresults(state)
        Amin = self.A(XPmin)

        print('Optimization complete!')
        print('Time = {0} s'.format(time.time()-tstart))
        print('Exit flag = {0}'.format(rep.terminationtype))
        print('Iterations = {0}'.format(rep.iterationscount))
        print('Obj. function value = {0}\n'.format(Amin))
        return XPmin, Amin, rep

    def min_lm(self, XP0, epsg=1e-8, epsf=1e-8, epsx=1e-8, maxits=10000):
        """
        Minimize the action starting from XP0, using the Levenburg-Marquardt method.
        This method supports the use of bounds.
        The vector-like structure used in the definition of the action is
        exploited here.
        """
        if self.taped == False:
            self.tape_A()

        # start the optimization
        print('Beginning optimization...')
        tstart = time.time()
        # initialize the L-BFGS optimization
        Nf = (self.N - 1) * self.D + self.N * self.L
        Nx = self.N * self.D + self.NPest
        state = xalglib.minlmcreatevj(Nx , Nf, list(XP0.flatten()))
        xalglib.minlmsetcond(state, epsg, epsf, epsx, maxits)
        # set optimization bounds
        if self.bounds is not None:
            bndl, bndu = self.bounds[:,0], self.bounds[:,1]
            xalglib.minlmsetbc(state, bndl, bndu)
        # run the optimization
        xalglib.minlmoptimize_vj(state, self.alglib_lm_vecA, self.alglib_lm_vecA_jac)
        # store the result of the optimization
        XPmin, rep = xalglib.minlmresults(state)
        Amin = self.A(XPmin)

        print('Optimization complete!')
        print('Time = {0} s'.format(time.time()-tstart))
        print('Exit flag = {0}'.format(rep.terminationtype))
        print('Iterations = {0}'.format(rep.iterationscount))
        print('Obj. function value = {0}\n'.format(Amin))
        return XPmin, Amin, rep

    def min_lm_FGH(self, XP0, epsg=1e-8, epsf=1e-8, epsx=1e-8, maxits=10000):
        """
        Minimize the action starting from XP0, using the Levenburg-Marquardt method.
        This method supports the use of bounds.
        This is for a general action function, NOT using the vector-like
        substructure in its definition.
        """
        if self.taped == False:
            self.tape_A()

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        # initialize the LM algorithm
        state = xalglib.minlmcreatefgh(list(XP0.flatten()))
        xalglib.minlmsetcond(state, epsg, epsf, epsx, maxits)
        # set optimization bounds
        if self.bounds is not None:
            bndl,bndu = self.bounds[:,0], self.bounds[:,1]
            xalglib.minlmsetbc(state, bndl, bndu)
        # run the optimization
        xalglib.minlmoptimize_fgh(state, self.alglib_LM_A, self.alglib_LM_A_grad,
                                  self.alglib_LM_A_hess)
        # store the result
        XPmin,rep = xalglib.minlmresults(state)
        Amin = self.A(XPmin)

        print('Optimization complete!')
        print('Time = {0} s'.format(time.time()-tstart))
        print('Exit flag = {0}'.format(rep.terminationtype))
        print('Iterations = {0}'.format(rep.iterationscount))
        print('Obj. function value = {0}\n'.format(Amin))
        return XPmin, Amin, rep

    def min_lbfgs_scipy(self, XP0):
        """
        """
        if self.taped == False:
            self.tape_A()

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.minimize(self.A, XP0, method='L-BFGS-B', jac=self.scipy_A_grad)
        XPmin,status,Amin = res.x, res.status, res.fun

        print('Optimization complete!')
        print('Time = {0} s'.format(time.time()-tstart))
        print('Exit flag = {0}'.format(status))
        print('Iterations = {0}'.format(res.nit))
        print('Obj. function value = {0}\n'.format(Amin))
        return XPmin, Amin, status

    ############################################################################
    # Class properties
    ############################################################################
