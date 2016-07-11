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
                 P=(), Pidx=(), adolcID=0):
        self.f = f
        self.dt = dt
        self._D = D
        self._Lidx = Lidx
        self._L = len(Lidx)
        self.RM = RM
        self.RF0 = RF0
        self.t = t

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

        self._N = self.Y.shape[0]
        self.P = P
        self.Pidx = Pidx
        self._NP = len(P[0])
        self._NPest = len(Pidx)
        self.adolcID = adolcID

        # other stuff
        self.A = self.A_gaussian
        self._taped = False
        self._initialized = False

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

    def load_stim(self, stim_file):
        """
        Load stimulus for experimental data from file.
        """

    ############################################################################
    # Gaussian action
    ############################################################################
    def A_gaussian(self, XP):
        """
        Gaussian action.
        """
        if self._NPest == 0:
            X = np.reshape(XP, (self._N, self._D))
        else:
            X = np.reshape(XP[:-self._NPest], (self._N, self._D))
            self.P[self.Pidx] = XP[-self._NPest:]

        # evaluate the action
        me = self.me_gaussian(X)
        fe = self.fe_gaussian(X)
        return me + fe

    # Gaussian action terms for matrix Rf and Rm
    def me_gaussian(self, X):
        """
        Gaussian measurement error.
        """
        if X.ndim == 1:
            X = np.reshape(X, (self._N, self._D))
        err = 0.0
        diff = X[:,self._Lidx] - self.Y

        if type(self.RM) == np.ndarray:
            if self.RM.shape == (self._N, self._L):
                err = np.sum(self.RM * diff * diff)
            elif self.RM.shape == (self._L, self._L):
                for diffn in diff:
                    err += np.dot(diffn, np.dot(self.RM, diffn))
            elif self.RM.shape == (self._N, self._L, self._L):
                for diffn,RMn in zip(diff, self._RM):
                    err += np.dot(diffn, np.dot(RMn, diffn))
            else:
                print("ERROR: RM is in an invalid shape.")
        else:
            err = self.RM * np.sum(diff * diff)

        return err / (self._L * self._N)

    def fe_gaussian(self, X):
        """
        Gaussian model error.
        """
        if X.ndim == 1:
            X = np.reshape(X, (self._N, self._D))
            if self._NPest > 0:
                self.P[self.Pidx] = X[-self._NPest:]
        err = 0.0
        diff = X[1:] - X[:-1] - self.disc(X)

        if type(self.RF) == np.ndarray:
            if self.RF.shape == (self._N - 1, self._D):
                err = np.sum(self.RF * diff * diff)
            elif self.RF.shape == (self._D, self._D):
                for diffn in diff:
                    err += np.dot(diff, self.RF, diff)
            elif self.RF.shape == (self._N - 1, self._D, self._D):
                for diffn,RFn in zip(diff, self.RF):
                    err += np.dot(diffn, np.dot(RFn, diffn))
            else:
                print("ERROR: RF is in an invalid shape.")
        else:
            err = self.RF * np.sum(diff * diff)

        return err / (self._D * (self._N - 1))

    # Error time series functions.
    def me_gaussian_TS(self, X):
        """
        Returns a time series of (normalized) measurement errors.
        """
        if X.ndim == 1:
            X = np.reshape(X, (self._N, self._D))
        err = np.zeros((self._N, self._L))
        diff = X[:,self._Lidx] - self.Y

        if type(self.RM) == np.ndarray:
            if self.RM.shape == (self._N, self._L):
                err = self.RM * diff * diff
            elif self.RM.shape == (self._L, self._L):
                for diffn in diff:
                    err[i] = np.dot(diffn, np.dot(self.RM, diffn))
            elif self.RM.shape == (self._N, self._L, self._L):
                for diffn,RMn in zip(diff, self._RM):
                    err += np.dot(diffn, np.dot(RMn, diffn))
            else:
                print("ERROR: RM is in an invalid shape.")
        else:
            err = self.RM * np.sum(diff * diff)

        return err / (self._L * self._N)
            

    ############################################################################
    # Discretization routines
    ############################################################################
    def disc_trapezoid(self, X):
        fn = self.f(X[:-1], self.t[:-1], self.P)
        fnp1 = self.f(X[1:], self.t[1:], self.P)
        return self.dt * (fn + fnp1) / 2.0

#    def disc_rk2(self, X, dt):
#        Xn, tn = X[:-1], self.t[:-1]
#        k1 = self.f(Xn, t, self.P)
#        k2 = self.f(Xn + k1/2.0, tn + dt/2.0, self.P)
#        return k2

    def disc_rk4(self, X):
        Xn = X[:-1]
        tn = np.tile(self.t[:-1], (self._D,1)).T
        k1 = self.f(Xn, tn, self.P)
        k2 = self.f(Xn + (self.dt*k1)/2.0, tn + self.dt/2.0, self.P)
        k3 = self.f(Xn + (self.dt*k2)/2.0, tn + self.dt/2.0, self.P)
        k4 = self.f(Xn + self.dt*k3, tn + self.dt, self.P)
        return self.dt * (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

    ############################################################################
    # Annealing functions
    ############################################################################
    def anneal_init(self, XP0, alpha, beta_array, RF0=None, bounds=None,
                    init_to_data=True, method='L-BFGS', disc='trapezoid'):
        """
        Initialize the annealing procedure.
        """
        if method not in ('L-BFGS', 'LM', 'L-BFGS-B'):
            print("ERROR: Optimization routine not implemented or recognized.")
            return 1

        self._initialized = True  # indicates we're at the first annealing step

        exec 'self.disc = self.disc_%s'%(disc,)

        self.alpha = alpha
        self.beta_array = beta_array
        self.betaidx = 0
        self._beta = self.beta_array[self.betaidx]
        self.Nbeta = len(self.beta_array)

        # array to store minimizing paths
        self.minpaths = np.zeros((self.Nbeta, len(XP0)), dtype='float')
        if init_to_data == True:
            X0r = np.reshape(XP0[:self._N*self._D], (self._N, self._D))
            X0r[:,self.Lidx] = self.Y[:]
            XP0 = X0r.flatten()
            if self._NPest > 0:
                P0 = XP0[-self._NP:]
                XP0 = np.append(XP0, P0)
        self.minpaths[0] = XP0

        # set current RF
        if RF0 is not None:
            self.RF0 = RF0
        self._RF = self.RF0 * self.alpha**self._beta

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
            XPmin, Amin, rep = self.min_lbfgs(self.minpaths[self.betaidx])
            self.exitflags[self.betaidx] = rep.terminationtype
        elif self.method == 'LM':
            XPmin, Amin, rep = self.min_lm(self.minpaths[self.betaidx])
            self.exitinfo.append[rep]
            self.exitflags[self.betaidx] = rep.terminationtype
        elif self.method == 'L-BFGS-B':
            XPmin, Amin, exitflag = self.min_lbfgs_scipy(self.minpaths[self.betaidx])
        else:
            print("ERROR: Optimization routine not implemented or recognized.")

        # store A_min and the minimizing path
        self.A_array[self.betaidx] = Amin
        self.me_array[self.betaidx] = self.me_gaussian(np.array(XPmin))
        self.fe_array[self.betaidx] = self.fe_gaussian(np.array(XPmin))
        self.minpaths[self.betaidx] = np.array(XPmin)

        # increase RF
        if self.betaidx < len(self.beta_array) - 1:
            self.betaidx += 1
            self._beta = self.beta_array[self.betaidx]
            self._RF = self.RF0 * self.alpha**self._beta

        # set flags indicating that A needs to be retaped, and that we're no
        # longer at the beginning of the annealing procedure
        self._taped = False
        self._initialized = False

    def anneal(self, XP0, alpha, beta_array, RF0=None, bounds=None,
               init_to_data=True, method='L-BFGS', disc='trapezoid'):
        """
        Convenience function to carry out a full annealing run over all values
        of beta in beta_array.
        """
        # initialize the annealing procedure, if not already done
        if self._initialized == False:
            self.anneal_init(XP0, alpha, beta_array, RF0, bounds, init_to_data, method, disc)
        for i in beta_array:
            print('------------------------------')
            print('Step %d of %d'%(self.betaidx+1,len(self.beta_array)))
            print('alpha = %f, beta = %f'%(self.alpha,self._beta))
            print('')
            self.anneal_step()

    def save_as_minAone(self, savedir='', savefile=None):
        """
        Save the result of this annealing in minAone data file style.
        """
        if savedir.endswith('/') == False:
            savedir += '/'
        if savefile is None:
            savefile = savedir + 'D%d_M%d_PATH%d.dat'%(self._D, self._L, self.adolcID)
        else:
            savefile = savedir + savefile
        betaR = self.beta_array.reshape((self.Nbeta,1))
        exitR = self.exitflags.reshape((self.Nbeta,1))
        AR = self.A_array.reshape((self.Nbeta,1))
        savearray = np.hstack((betaR, exitR, AR, self.minpaths))
        np.savetxt(savefile, savearray)

    def save_paths(self, savedir='', savefile=None, filetype='npy'):
        """
        Save path vectors over the annealing.
        """
        if savedir.endswith('/') == False:
            savedir += '/'
        if savefile is None:
            if filetype == 'npy':
                savefile = savedir + 'trial%D_paths.npy'%(self.adolcID)
            elif filetype in ('dat', 'text'):
                savefile = savedir + 'trial%D_paths.dat'%(self.adolcID)
            else:
                print("ERROR: File type unknown.")
                return 1
        else:
            if filetype == 'npy' and savefile.endswith('.npy') == False:
                print("CAUTION: File extension changed to .npy")
                savefile = savefile[:-4]
                savefile += '.npy'
            savefile = savedir + savefile

        betaR = self.beta_array.reshape((self.Nbeta,1))
        savearray = np.hstack((betaR, self.minpaths))
        if filetype == 'npy':
            np.save(savefile, savearray)
        else:
            np.savetxt(savefile, savearray)

    def save_action(self, savedir='./', savefile=None, filetype='npy'):
        """
        Save action values over the annealing.
        """
        if savedir.endswith('/') == False:
            savedir += '/'
        if savefile is None:
            if filetype == 'npy':
                savefile = savedir + 'trial%D_action.npy'%(self.adolcID)
            elif filetype in ('dat', 'text'):
                savefile = savedir + 'trial%D_action.dat'%(self.adolcID)
            else:
                print("ERROR: File type unknown.")
                return 1
        else:
            if filetype == 'npy' and savefile.endswith('.npy') == False:
                print("CAUTION: File extension changed to .npy")
                savefile = savefile[:-4]
                savefile += '.npy'
            savefile = savedir + savefile

        betaR = self.beta_array.reshape((self.Nbeta,1))
        AR = self.A_array.reshape((self.Nbeta,1))
        savearray = np.hstack((betaR, AR))
        if filetype == 'npy':
            np.save(savefile, savearray)
        else:
            np.savetxt(savefile, savearray)

    def save_action_errors(self, savefile):
        """
        Save the action, as well as the time series of errors (with and without
        RM/RF included).
        """
        AR = self.A_array.reshape((self.Nbeta, 1))
        

    def save_modelerr(self, savedir='', savefile=None, filetype='npy'):
        """
        Save model error over the annealing.
        """
        if savedir.endswith('/') == False:
            savedir += '/'
        if savefile is None:
            if filetype == 'npy':
                savefile = savedir + 'trial%D_modelerr.npy'%(self.adolcID)
            elif filetype in ('dat', 'text'):
                savefile = savedir + 'trial%D_modelerr.dat'%(self.adolcID)
            else:
                print("ERROR: File type unknown.")
                return 1
        else:
            if filetype == 'npy' and savefile.endswith('.npy') == False:
                print("CAUTION: File extension changed to .npy")
                savefile = savefile[:-4]
                savefile += '.npy'
            savefile = savedir + savefile

        betaR = self.beta_array.reshape((self.Nbeta,1))
        AR = self.A_array.reshape((self.Nbeta,1))
        savearray = np.hstack((betaR, AR))
        if filetype == 'npy':
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

    def alglib_LM_A(self, XP, param=None):
        """
        ALGLIB-acceptable action for the Levenburg-Marquardt algorithm.
        """
        return adolc.function(self.adolcID, XP)

    def alglib_LM_A_grad(self, XP, grad_A, param=None):
        """
        ALGLIB-acceptable action gradient for the Levenburg-Marquardt algorithm.
        Sets gradient by reference.
        """
        grad_A[:] = adolc.gradient(self.adolcID, XP)
        return adolc.function(self.adolcID, XP)

    def alglib_LM_A_hess(self, XP, grad_A, hess_A, param=None):
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
        xtrace = np.random.rand(self._D*self._N + self._NPest)
        adolc.trace_on(self.adolcID)
        # set the active independent variables
        ax = adolc.adouble(xtrace)
        adolc.independent(ax)
        # set the dependent variable (A)
        af = self.A(ax)
        adolc.dependent(af)
        adolc.trace_off()
        self._taped = True
        print('Time = {0} s\n'.format(time.time()-tstart))

    def min_lbfgs(self, XP0, epsg=1e-8, epsf=1e-8, epsx=1e-8, maxits=10000):
        """
        Minimize f starting from x0 using L-BFGS.
        Returns the minimizing state, the minimum function value, and the L-BFGS
        termination information.
        """
        if self._taped == False:
            self.tape_A()

        # start the optimization
        print('Beginning optimization...')
        tstart = time.time()
        # initialize the L-BFGS optimization
        state = xalglib.minlbfgscreate(1, list(XP0.flatten()))
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

    def min_lm(self, XP0, epsg=1e-8, epsf=1e-8, epsx=1e-8, maxits=10000):
        """
        Minimize the action starting from XP0, using the Levenburg-Marquardt method.
        This method supports the use of bounds.
        """
        if self._taped == False:
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
        if self._taped == False:
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
    @property
    def Lidx(self):
        return self._Lidx
    @Lidx.setter
    def Lidx(self, Lidx):
        self._Lidx = Lidx
        self._L = len(Lidx)

    @property
    def L(self):
        return self._L
    @L.setter
    def L(self, L):
        print("Can\'t set L independently. Reset Lidx instead.")

    @property
    def N(self):
        return self._N
    @N.setter
    def N(self, N):
        print("Can\'t set N independently.  Automatically set with new data.")

    @property
    def NPest(self):
        return self._NPest
    @NPest.setter
    def NPest(self, NPest):
        print("Can\'t set NPest independently. Reset Pidx instead.")

    @property
    def RF(self):
        return self._RF
    @RF.setter
    def RF(self, RF):
        print("Resetting RF manually not allowed.")

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, beta):
        print("Resetting beta manually not allowed.")
