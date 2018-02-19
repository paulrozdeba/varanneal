"""
Paul Rozdeba (prozdeba@physics.ucsd.edu)
Department of Physics
University of California, San Diego
May 23, 2017

VarAnneal

Carry out the variational annealing algorithm (VA) for estimating unobserved
dynamical model states and parameters from time series data.

VA is a form of variational data assimilation that uses numerical continuation
to regularize the variational cost function, or "action", in a controlled way.
VA was first proposed by Jack C. Quinn in his Ph.D. thesis (2010) [1], and is
described by J. Ye et al. (2015) in detail in [2].

This code uses automatic differentiation to evaluate derivatives of the
action for optimization as implemented in ADOL-C, wrapped in Python code in a
package called PYADOLC (installation required for usage of VarAnneal).
PYADOLC is available at https://github.com/b45ch1/pyadolc.

To run the annealing algorithm using this code, instantiate an Annealer object
in your code using this module.  This object allows you to load in observation
data, set a model for the system, initial guesses for the states and parameters,
etc.  To get a good sense of how to use the code, follow along with the examples
included with this package, and the user guide (coming soon).

References:
[1] J.C. Quinn, "A path integral approach to data assimilation in stochastic
    nonlinear systems."  Ph.D. thesis in physics, UC San Diego (2010).
    Available at: https://escholarship.org/uc/item/0bm253qk

[2] J. Ye et al., "Improved variational methods in statistical data assimilation."
    Nonlin. Proc. in Geophys., 22, 205-213 (2015).
"""

import numpy as np
import adolc
import time
import sys
import default_actions
from _autodiffmin import ADmin
from _autodiffipopt import ADipopt

class Annealer(object):
    """
    Annealer is the main object type for performing variational data
    assimilation using VA.  It inherits the function minimization routines
    from ADmin, which uses automatic differentiation.
    """
    def __init__(self):
        """
        Constructor for the Annealer class.
        - Set ipopt_flag = True if you plan to use IPOPT for optimization.
        """
        self.A_taped_flag = False
        self.annealing_initialized = False
        self.dt_model = None
        self.ipopt_flag = False  # assume no IPOPT by default
        self.minimizer = None

    #def set_action(self, action):
    #    """
    #    Set the action function.
    #    action should be a callable class instantiation; see default_actions
    #    for some examples.  It must be properly instantiated and ready to go
    #    BEFORE being loaded in here.
    #    """
    #    self.A = action

    def set_model(self, f, D, dt_model=None):
        """
        Set the D-dimensional dynamical model for the estimated system.
        The model function, f, must take arguments in the following order:
            t, x, p
        or, if there is a time-dependent stimulus for f (nonautonomous term):
            t, x, (p, stim)
        where x and stim are at the "current" time t.  Thus, x should be a
        D-dimensional vector, and stim similarly a D_stim-dimensional vector.

        For now, dt_model can optionally be defined here or in anneal_init.
        In future versions of the code, dt_model will probably be removed
        from the inputs of anneal_init and required to be passed in here.
        """
        self.f = f
        self.D = D
        if dt_model is not None:
            self.dt_model = dt_model

    def set_data_fromfile(self, data_file, stim_file=None, nstart=0, N=None):
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

        self.t_data = data[:, 0]
        self.dt_data = self.t_data[1] - self.t_data[0]
        self.Y = data[:, 1:]

        if stim_file.endswith('npy'):
            s = np.load(stim_file)
        else:
            s = np.loadtxt(stim_file)
        self.stim = s[:, 1:]

        self.dt_data = dt_data

    def set_data(self, data, stim=None, t=None, nstart=0, N=None):
        """
        Directly pass in data and stim arrays
        If you pass in t, it's assumed y/stim does not contain time.  Otherwise,
        it has to contain time in the zeroth element of each sample.
        """
        if N is None:
            self.N_data = data.shape[0]
        else:
            self.N_data = N

        if t is None:
            self.t_data = data[nstart:(nstart + self.N_data), 0]
            self.dt_data = self.t_data[1] - self.t_data[0]
            self.Y = data[nstart:(nstart + self.N_data), 1:]
            if stim is not None:
                self.stim = stim[nstart:(nstart + self.N_data), 1:]
            else:
                self.stim = None
        else:
            self.t_data = t[nstart:(nstart + self.N_data)]
            self.dt_data = self.t_data[1] - self.t_data[0]
            self.Y = data[nstart:(nstart + self.N_data)]
            if stim is not None:
                self.stim = stim[nstart:(nstart + self.N_data)]
            else:
                self.stim = None

    def ipopt_init(self, g, x_L, x_U, g_L, g_U):
        """
        Initialize IPOPT hyperparameters and other settings.
        g - Equations of constraint
        x_L, x_U - Lower and upper bounds for solutions.
        g_L, g_U - Lower and upper bounds for constraints.
        """
        self.ipopt_flag = True  # Using IPOPT
        self.g_taped_flag = False  # Constraints not yet taped
        self.lgr_taped_flag = False  # Lagrangian not yet taped

        self.ipopt_g = g
        self.ipopt_x_L = x_L
        self.ipopt_x_U = x_U
        self.ipopt_g_L = g_L
        self.ipopt_g_U = g_U

    ############################################################################
    # Annealing functions
    ############################################################################
    def anneal(self, X0, P0, alpha, beta_array, RM, RF0, Lidx, Pidx, dt_model=None,
               init_to_data=True, action='A_gaussian', disc='trapezoid', 
               method='L-BFGS-B', bounds=None, opt_args=None, adolcID=0,
               track_paths=None, track_params=None, track_action_errors=None):
        """
        Convenience function to carry out a full annealing run over all values
        of beta in beta_array.
        """
        # Initialize the annealing procedure, if not already done.
        if self.annealing_initialized == False:
            self.anneal_init(X0, P0, alpha, beta_array, RM, RF0, Lidx, Pidx, dt_model,
                             init_to_data, action, disc, method, bounds,
                             opt_args, adolcID)

        # Loop through all beta values for annealing.
        for i in beta_array:
            print('------------------------------')
            print('Step %d of %d'%(self.betaidx+1, len(self.beta_array)))
            # Print RF
            if type(self.RF) == np.ndarray:
                if self.RF.shape == (self.N_model - 1, self.D):
                    print('beta = %d, RF[n=0, i=0] = %.8e'%(self.beta, self.RF[0, 0]))
                elif self.RF.shape == (self.N_model - 1, self.D, self.D):
                    print('beta = %d, RF[n=0, i=0, j=0] = %.8e'%(self.beta, self.RF[0, 0, 0]))
                else:
                    print("Error: RF has an invalid shape. You really shouldn't be here...")
                    sys.exit(1)
            else:
                print('beta = %d, RF = %.8e'%(self.beta, self.RF))
            print('')

            self.anneal_step()

            # Track progress by saving to file after every step, if specified
            self.track_progress(track_paths, track_params, track_action_errors)

    def track_progress(self, track_paths=False, track_params=False, track_action_errors=False):
        """
        Track progress.
        """
        if track_paths is not None:
            try:
                dtype = track_paths['dtype']
            except:
                dtype = np.float64
            try:
                fmt = track_paths['fmt']
            except:
                fmt = "%.8e"
            self.save_paths(track_paths['filename'], dtype, fmt)

        if track_params is not None:
            try:
                dtype = track_params['dtype']
            except:
                dtype = np.float64
            try:
                fmt = track_params['fmt']
            except:
                fmt = "%.8e"
            self.save_params(track_params['filename'], dtype, fmt)

        if track_action_errors is not None:
            try:
                cmpt = track_action_errors['cmpt']
            except:
                cmpt = 0
            try:
                dtype = track_action_errors['dtype']
            except:
                dtype = np.float64
            try:
                fmt = track_action_errors['fmt']
            except:
                fmt = "%.8e"
            self.save_action_errors(track_action_errors['filename'], cmpt, dtype, fmt)

    def anneal_init(self, X0, P0, alpha, beta_array, RM, RF0, Lidx, Pidx, dt_model=None,
                    init_to_data=True, action='A_gaussian', disc='trapezoid',
                    method='L-BFGS-B', bounds=None, opt_args=None, adolcID=0):
        """
        Initialize the annealing procedure.
        """
        # set up beta array in RF = RF0 * alpha**beta
        self.alpha = alpha
        self.beta_array = np.array(beta_array, dtype=np.uint16)
        self.Nbeta = len(self.beta_array)

        #if action == 'A_gaussian':
        # Separate dt_data and dt_model not supported yet if there is an external stimulus.
        if dt_model is not None and dt_model != self.dt_data and self.stim is not None:
            print("Error! Separate dt_data and dt_model currently not supported with an " +\
                  "external stimulus. Exiting.")
            sys.exit(1)
        else:
            if dt_model is None and self.dt_model is None:
                self.dt_model = self.dt_data
                self.N_model = self.N_data
                self.merr_nskip = 1
                self.t_model = np.copy(self.t_data)
            else:
                self.dt_model = dt_model
                self.merr_nskip = int(self.dt_data / self.dt_model)
                self.N_model = (self.N_data - 1) * self.merr_nskip + 1
                self.t_model = np.linspace(self.t_data[0], self.t_data[-1], self.N_model)

        # set up parameters and determine if static or time series
        self.P = P0
        if P0.ndim == 1:
            # Static parameters, so p is a single vector.
            self.NP = len(P0)
        else:
            # Time-dependent parameters, so p is a time series of N values.
            self.NP = P0.shape[1]

        # get indices of parameters to be estimated by annealing
        self.Pidx = Pidx
        self.NPest = len(Pidx)

        # get indices of measured components of f
        self.Lidx = Lidx
        self.L = len(Lidx)

        # initialize observed state components to data if desired
        if init_to_data == True:
            X0[::self.merr_nskip, self.Lidx] = self.Y[:]

        # Reshape RM and RF so that they span the whole time series, if they
        # are passed in as vectors or matrices. This is done because in the
        # action evaluation, it is more efficient to let numpy handle
        # multiplication over time rather than using python loops.
        # If RM or RF is already passed in as a time series, move on!
        if type(RM) == list:
            RM = np.array(RM)
        if type(RM) == np.ndarray:
            if RM.shape == (self.L,):
                self.RM = np.resize(RM, (self.N_data, self.L))
            elif RM.shape == (self.L, self.L):
                self.RM = np.resize(RM, (self.N_data, self.L, self.L))
            elif RM.shape in [(self.N_data, self.L), (self.N_data, self.L, self.L)]:
                self.RM = RM
            else:
                print("ERROR: RM has an invalid shape. Exiting.")
                sys.exit(1)
        else:
            self.RM = RM

        if type(RF0) == list:
            RF0 = np.array(RF0)
        if type(RF0) == np.ndarray:
            if RF0.shape == (self.D,):
                self.RF0 = np.resize(RF0, (self.N_model - 1, self.D))
            elif RF0.shape == (self.D, self.D):
                self.RF0 = np.resize(RF0, (self.N_model - 1, self.D, self.D))
            elif RF0.shape in [(self.N_model - 1, self.D), (self.N_model - 1, self.D, self.D)]:
                self.RF0 = RF0
            else:
                print("ERROR: RF0 has an invalid shape. Exiting.")
                sys.exit(1)
        else:
            self.RF0 = RF0

        # set initial RF
        self.betaidx = 0
        self.beta = self.beta_array[self.betaidx]
        self.RF = self.RF0 * self.alpha**self.beta

        self.A = default_actions.GaussianAction(self.N_model, self.D, self.merr_nskip,
                len(Lidx), Lidx, self.Y, RM, self.N_data, P0, len(P0), len(Pidx),
                disc, RF0, self.stim, self.f, self.t_model, self.dt_model)

        if method not in ('L-BFGS-B', 'NCG', 'LM', 'TNC', 'IPOPT'):
            print("ERROR: Optimization routine not recognized. Annealing not initialized.")
            return None
        else:
            self.method = method

        # get optimization extra arguments
        self.opt_args = opt_args

        # Store optimization bounds. Will only be used if the chosen
        # optimization routine supports it.
        if bounds is not None:
            self.bounds = []
            state_b = bounds[:self.D]
            param_b = bounds[self.D:]
            # set bounds on states for all N time points
            for n in xrange(self.N_model):
                for i in xrange(self.D):
                    self.bounds.append(state_b[i])
            # set bounds on parameters
            if self.P.ndim == 1:
                # parameters are static
                for i in xrange(self.NPest):
                    self.bounds.append(param_b[i])
            else:
                # parameters are time-dependent
                #if self.disc.im_func.__name__ in ["disc_euler", "disc_forwardmap"]:
                if self.disc in ['euler', 'forwardmap']:
                    nmax = N_model - 1
                else:
                    nmax = N_model
                for n in xrange(self.nmax):
                    for i in xrange(self.NPest):
                        self.bounds.append(param_b[i])
        else:
            self.bounds = None

        # array to store minimizing paths
        if P0.ndim == 1:
            self.minpaths = np.zeros((self.Nbeta, self.N_model*self.D + self.NP), dtype=np.float64)
        else:
            #if self.disc.im_func.__name__ in ["disc_euler", "disc_forwardmap"]:
            if self.disc in ['euler', 'forwardmap']:
                nmax_p = self.N_model - 1
            else:
                nmax_p = self.N_model
            self.minpaths = np.zeros((self.Nbeta, self.N_model*self.D + nmax_p*self.NP), 
                                      dtype=np.float64)

        # Flatten X0 and P0 into extended XP0 path vector
        #if self.NPest > 0:
        #    if P0.ndim == 1:
        #        XP0 = np.append(X0.flatten(), P0)
        #    else:
        #        XP0 = np.append(X0.flatten(), P0.flatten())
        #else:
        #    XP0 = X0.flatten()
        if P0.ndim == 1:
            XP0 = np.append(X0.flatten(), P0)
        else:
            XP0 = np.append(X0.flatten(), P0.flatten())

        self.minpaths[0] = XP0

        # array to store optimization results
        self.A_array = np.zeros(self.Nbeta, dtype=np.float64)
        self.me_array = np.zeros(self.Nbeta, dtype=np.float64)
        self.fe_array = np.zeros(self.Nbeta, dtype=np.float64)
        self.exitflags = np.empty(self.Nbeta, dtype=np.int8)

        # set the adolcID
        self.adolcID = adolcID

        # Finally, initialize an ADmin or ADipopt instance
        if not self.ipopt_flag:
            self.minimizer = ADmin(self.A, self.opt_args, self.bounds, self.adolcID)
        else:
            self.minimizer = ADipopt(self.A, self.adolcID)

        # Initialization successful, we're at the beta = beta_0 step now.
        self.initalized = True

    def anneal_step(self):
        """
        Perform a single annealing step. The cost function is minimized starting
        from the previous minimum (or the initial guess, if this is the first
        step). Then, RF is increased to prepare for the next annealing step.
        """
        # Set IPOPT constraints
        if self.ipopt_flag:
            bi = self.betaidx
            self.minimizer.set_ipopt_constraints(self.ipopt_g,
                                                 self.ipopt_x_L[bi], self.ipopt_x_U[bi],
                                                 self.ipopt_g_L[bi], self.ipopt_g_U[bi])

        if self.method in ['L-BFGS-B', 'NCG', 'TNC', 'IPOPT']:#, LM]:
            # Decide which path to use at init. If we're on the first step of
            # annealing, use the seed path specified earlier (before any
            # optimization has been performed). Otherwise just use the minimizing
            # path stored previously.
            if self.betaidx == 0:
                mpidx = 0
            else:
                mpidx = self.betaidx - 1
            # Now load this path into XP0, the optimization seed.
            if self.NPest == 0:
                XP0 = np.copy(self.minpaths[mpidx][:self.N_model*self.D])
            elif self.NPest == self.NP:
                XP0 = np.copy(self.minpaths[mpidx])
            else:
                X0 = self.minpaths[mpidx][:self.N_model*self.D]
                P0 = self.minpaths[mpidx][self.N_model*self.D:][self.Pidx]
                XP0 = np.append(X0, P0)

            # Now minimize using the routine of choice.
            if self.method == 'L-BFGS-B':
                #XPmin, Amin, exitflag, self.A_taped_flag = \
                XPmin, Amin, exitflag = \
                    self.minimizer.min_lbfgs_scipy(XP0, self.gen_xtrace(), self.A_taped_flag)
            elif self.method == 'NCG':
                #XPmin, Amin, exitflag, self.A_taped_flag = \
                XPmin, Amin, exitflag = \
                    self.minimizer.min_cg_scipy(XP0, self.gen_xtrace(), self.A_taped_flag)
            elif self.method == 'TNC':
                #XPmin, Amin, exitflag, self.A_taped_flag = \
                XPmin, Amin, exitflag = \
                    self.minimizer.min_tnc_scipy(XP0, self.gen_xtrace(), self.A_taped_flag)
            elif self.method == 'IPOPT':
                #XPmin, Amin, exitflag, self.A_taped_flag, self.g_taped_flag, self.lgr_taped_flag = self.minimizer.min_ipopt(XP0, self.N_model*self.D + self.NPest, self.gen_xtrace(), self.gen_lgr_mult_trace(), self.gen_obj_fac_trace(), self.A_taped_flag, self.g_taped_flag, self.lgr_taped_flag)
                XPmin, Amin, exitflag = self.minimizer.min_ipopt(XP0, self.N_model*self.D + self.NPest, self.gen_xtrace(), self.gen_lgr_mult_trace(), self.gen_obj_fac_trace(), self.A_taped_flag, self.g_taped_flag, self.lgr_taped_flag)
            #elif self.method == 'LM':
            #    XPmin, Amin, exitflag = self.min_lm_scipy(XP0)
            else:
                print("You really shouldn't be here.  Exiting.")
                sys.exit(1)
        else:
            print("ERROR: Optimization routine not implemented or recognized.")
            sys.exit(1)

        # update optimal parameter values
        if self.NPest > 0:
            if self.P.ndim == 1:  # Time series of parameters
                if isinstance(XPmin[0], adolc._adolc.adouble):
                    self.P[self.Pidx] = np.array([XPmin[-self.NPest + i].val \
                                                  for i in xrange(self.NPest)])
                else:
                    self.P[self.Pidx] = np.copy(XPmin[-self.NPest:])
            else:  # Static parameters
                #if self.disc.im_func.__name__ in ["disc_euler", "disc_forwardmap"]:
                if self.disc in ['euler', 'forwardmap']:
                    nmax = self.N_model - 1
                else:
                    nmax = self.N_model
                for n in xrange(nmax):
                    if isinstance(XPmin[0], adolc._adolc.adouble):
                        nidx = nmax - n - 1
                        self.P[n, self.Pidx] = np.array([XPmin[-nidx*self.NPest + i].val \
                                                         for i in xrange(self.NPest)])
                    else:
                        pi1 = nmax*self.D + n*self.NPest
                        pi2 = nmax*self.D + (n+1)*self.NPest
                        self.P[n, self.Pidx] = np.copy(XPmin[pi1:pi2])

            # Update parameter values in the action container object.
            self.A.P = self.P

        # store A_min and the minimizing path
        self.A_array[self.betaidx] = Amin
        self.me_array[self.betaidx] = self.A.meas_err(np.array(XPmin[:self.N_model*self.D]))
        self.fe_array[self.betaidx] = self.A.model_err(np.array(XPmin))
        #self.me_array[self.betaidx] = self.me_gaussian(np.array(XPmin[:self.N_model*self.D]))
        #self.fe_array[self.betaidx] = self.fe_gaussian(np.array(XPmin))
        self.minpaths[self.betaidx] = np.array(np.append(XPmin[:self.N_model*self.D], self.P))

        # increase RF
        if self.betaidx < len(self.beta_array) - 1:
            self.betaidx += 1
            self.beta = self.beta_array[self.betaidx]
            self.RF = self.RF0 * self.alpha**self.beta
            self.A.RF = self.RF
            # RF increased ==> need to retape before next step
            self.A_taped_flag = False
            if self.ipopt_flag:
                self.g_taped_flag = False
                self.lgr_taped_flag = False

        # set flag indicating that we're no longer at the beginning of the annealing procedure
        if self.annealing_initialized:
            # Indicate no longer at beta_0
            self.initialized = False

    ################################################################################
    # Routines to save annealing results.
    ################################################################################
    def save_paths(self, filename, dtype=np.float64, fmt="%.8e"):
        """
        Save minimizing paths (not including parameters).
        """
        savearray = np.reshape(self.minpaths[:, :self.N_model*self.D], \
                               (self.Nbeta, self.N_model, self.D))

        # append time
        tsave = np.reshape(self.t_model, (self.N_model, 1))
        tsave = np.resize(tsave, (self.Nbeta, self.N_model, 1))
        savearray = np.dstack((tsave, savearray))

        if filename.endswith('.npy'):
            np.save(filename, savearray.astype(dtype))
        else:
            np.savetxt(filename, savearray, fmt=fmt)

    def save_params(self, filename, dtype=np.float64, fmt="%.8e"):
        """
        Save minimum action parameter values.
        """
        if self.NPest == 0:
            print("WARNING: You did not estimate any parameters.  Writing fixed " \
                  + "parameter values to file anyway.")

        # write fixed parameters to array
        if self.P.ndim == 1:
            savearray = np.resize(self.P, (self.Nbeta, self.NP))
        else:
            #if self.disc.im_func.__name__ in ["disc_euler", "disc_forwardmap"]:
            if self.disc in ['euler', 'forwardmap']:
                savearray = np.resize(self.P, (self.Nbeta, self.N_model - 1, self.NP))
            else:
                savearray = np.resize(self.P, (self.Nbeta, self.N_model, self.NP))
        # write estimated parameters to array
        if self.NPest > 0:
            if self.P.ndim == 1:
                est_param_array = self.minpaths[:, self.N_model*self.D:]
                savearray[:, self.Pidx] = est_param_array
            else:
                #if self.disc.im_func.__name__ in ["disc_euler", "disc_forwardmap"]:
                if self.disc in ['euler', 'forwardmap']:
                    est_param_array = np.reshape(self.minpaths[:, self.N_model*self.D:],
                                                 (self.Nbeta, self.N_model - 1, self.NPest))
                    savearray[:, :, self.Pidx] = est_param_array
                else:
                    est_param_array = np.reshape(self.minpaths[:, self.N_model*self.D:],
                                                 (self.Nbeta, self.N_model, self.NPest))
                    savearray[:, :, self.Pidx] = est_param_array

        if filename.endswith('.npy'):
            np.save(filename, savearray.astype(dtype))
        else:
            np.savetxt(filename, savearray, fmt=fmt)

    def save_action_errors(self, filename, cmpt=0, dtype=np.float64, fmt="%.8e"):
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
            if self.RF0.shape == (self.N_model - 1, self.D):
                savearray[:, 4] = self.fe_array / (self.RF0[0, 0] * self.alpha**self.beta_array)
            elif self.RF0.shape == (self.N_model - 1, self.D, self.D):
                savearray[:, 4] = self.fe_array / (self.RF0[0, 0, 0] * self.alpha**self.beta_array)
            else:
                print("RF shape currently not supported for saving.")
                return 1
        else:
            savearray[:, 4] = self.fe_array / (self.RF0 * self.alpha**self.beta_array)

        if filename.endswith('.npy'):
            np.save(filename, savearray.astype(dtype))
        else:
            np.savetxt(filename, savearray, fmt=fmt)

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

    ############################################################################
    # AD taping & derivatives
    ############################################################################
    def gen_xtrace(self):
        """
        Define a random state vector for the AD trace.
        """
        if self.P.ndim == 1:
            xtrace = np.random.rand(self.N_model*self.D + self.NPest)
        else:
            #if self.disc.im_func.__name__ in ["disc_euler", "disc_forwardmap"]:
            if self.disc in ['euler', 'forwardmap']:
                xtrace = np.random.rand(self.N_model*self.D + (self.N_model-1)*self.NPest)
            else:
                xtrace = np.random.rand(self.N_model*(self.D + self.NPest))
        return xtrace

    def gen_lgr_mult_trace(self):
        """
        Define random values for lagrange multipliers for ADOLC trace.
        """
        lgr_mult_trace = np.random.rand(self.minimizer.ncon)
        return lgr_mult_trace

    def gen_obj_fac_trace(self):
        """
        Define a random value for the objective function factor value for tracing.
        """
        obj_fac_trace = np.random.rand()
        return obj_fac_trace
