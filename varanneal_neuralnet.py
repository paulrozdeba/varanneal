"""
Paul Rozdeba (prozdeba@physics.ucsd.edu)
Department of Physics
University of California, San Diego
May 23, 2017

NeuralNetVA

Carry out variational annealing (VA) to train a neural network model.  Network
states and activation function parameters are estimated from training data
consisting of many input/output pair examples.

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
import scipy.optimize as opt
from common import ADmin

class Annealer(ADmin):
    """
    Annealer is the main object type for performing variational data
    assimilation using VA.  It inherits the function minimization routines
    from ADmin, which uses automatic differentiation.
    """
    def __init__(self):
        """
        Constructor for the Annealer class.
        """
        self.taped = False
        self.annealing_initialized = False
        self.M = 0  # number of training examples
        self.structure = None

    def set_structure(self, structure):
        """
        Set up the network structure.
        structure - a length-N numpy array for an N-layer network, where each
        entry is the number of neurons in that layer.
        """
        self.structure = structure
        self.N = len(structure)
        if self.M > 0:
            self.NDnet = np.sum(structure)
            self.NDens = self.NDnet * self.M

    def set_activation(self, f):
        """
        Set activation for a neuron.
        f - activation function
        """
        self.f = f

    def set_input_data(self, data):
        """
        Set data for the input layer.
        """
        if data.ndim == 1:
            self.data_in = np.array([data])
        else:
            self.data_in = data

        if self.M == 0:
            self.M = data.shape[0]
        if self.structure is not None:
            self.NDnet = np.sum(self.structure)
            self.NDens = self.NDnet * self.M

    def set_output_data(self, data):
        """
        Set data for the output layer.
        """
        if data.ndim == 1:
            self.data_out = np.array([data])
        else:
            self.data_out = data

        if self.M == 0:
            self.M = data.shape[0]
        if self.structure is not None:
            self.NDnet = np.sum(self.structure)
            self.NDens = self.NDnet * self.M

    ############################################################################
    # Gaussian action
    ############################################################################
    def A_gaussian(self, XP):
        """
        Calculate the Gaussian action all in one go.
        """
        return self.me_gaussian(XP) + self.fe_gaussian(XP)

    def me_gaussian(self, XP):
        """
        Gaussian measurement error.
        """
        X = XP[:self.NDens]

        # Extract input and output layer states corresponding to the first example
        xmeas_in = X[:self.structure[0]][self.Lidx[0]]
        xmeas_out = X[(self.NDnet - self.structure[-1]):self.NDnet][self.Lidx[1]]

        # Calculate measurement error for the first example
        diff_in = xmeas_in - self.data_in[0]
        diff_out = xmeas_out - self.data_out[0]

        if type(self.RM) == np.ndarray:
            if self.RM.shape == (2,):
                merr = self.RM[0] * np.sum(diff_in*diff_in)
                merr = merr + self.RM[1] * np.sum(diff_out*diff_out)
            else:
                # Assume otherwise that you put in an array which looks like
                # [[RMin_1, RMin_2, ..., RMin_Lin], [RMout_1, RMout_2, ..., RMout_Lout]]
                merr = np.dot(diff_in, np.dot(self.RM[0], diff_in))
                merr = merr + np.dot(diff_out, np.dot(self.RM[1], diff_out))
            #else:
            #    print("ERROR: RM has an invalid shape. Exiting.")
            #    exit(1)
        else:
            merr = self.RM * (np.sum(diff_in*diff_in) + np.sum(diff_out*diff_out))

        # Loop through the remaining M-1 examples and add to the measurement error
        for m in xrange(1, self.M):
            # Get input and output layer states for example m+1
            i0 = m * self.NDnet
            i1 = (m+1) * self.NDnet
            xmeas_in = X[i0:i0 + self.structure[0]][self.Lidx[0]]
            xmeas_out = X[i1 - self.structure[-1]:i1][self.Lidx[1]]

            # Calculate error vectors and measurement error contribution
            diff_in = xmeas_in - self.data_in[m]
            diff_out = xmeas_out - self.data_out[m]

            if type(self.RM) == np.ndarray:
                if self.RM.shape == (2,):
                    merr = merr + self.RM[0] * np.sum(diff_in*diff_in)
                    merr = merr + self.RM[1] * np.sum(diff_out*diff_out)
                else:
                    # Assume otherwise that you put in an array which looks like
                    # [[RMin_1, RMin_2, ..., RMin_Lin], [RMout_1, RMout_2, ..., RMout_Lout]]
                    merr = merr + np.dot(diff_in, np.dot(self.RM[0], diff_in))
                    merr = merr + np.dot(diff_out, np.dot(self.RM[1], diff_out))
                #else:
                #    print("ERROR: RM has an invalid shape. Exiting.")
                #    exit(1)
            else:
                merr = merr + self.RM * (np.sum(diff_in*diff_in) + np.sum(diff_out*diff_out))

        return merr / np.float64(self.Ltot * self.M)

    def fe_gaussian(self, XP):
        """
        Gaussian model error.
        """
        # Extract state and parameters from XP.
        if self.NPest == 0:
            x = XP
            p = self.P
        elif self.NPest == self.NP:
            x = XP[:self.NDens]
            p = XP[self.NDens:]
        else:
            x = XP[:self.NDens]
            p = np.array(self.P, dtype=XP.dtype)
            p[self.Pidx] = XP[self.NDens:]
        p = np.array(p)

        # Start calculating the model error.
        # First get out the paramters, which are the same for EVERY example.
        W = []
        b = []
        W_i0 = 0
        W_if = self.structure[0]*self.structure[1]
        b_i0 = W_if
        b_if = b_i0 + self.structure[1]
        for n in xrange(self.N - 1):
            W.append(np.reshape(p[W_i0:W_if], (self.structure[n+1], self.structure[n])))
            b.append(p[b_i0:b_if])
            if n < self.N - 2:
                W_i0 = b_if
                W_if = W_i0 + self.structure[n+1]*self.structure[n+2]
                b_i0 = W_if
                b_if = b_i0 + self.structure[n+2]

        # Now loop over each copy of the network for each example
        for m in xrange(self.M):
            # Get entire network state for example m+1
            Xm = x[m*self.NDnet:(m+1)*self.NDnet]
            diff = np.zeros(np.sum(self.structure[1:]), dtype=x.dtype)
            diff_i0 = 0
            diff_if = self.structure[1]
            xn_i0 = 0
            xn_if = self.structure[0]
            xnp1_i0 = self.structure[0]
            xnp1_if = self.structure[0] + self.structure[1]
            #W_i0 = 0
            #W_if = self.structure[0]*self.structure[1]
            #b_i0 = W_if
            #b_if = b_i0 + self.structure[1]

            for n in xrange(self.N-1):
                xn = Xm[xn_i0:xn_if]
                xnp1 = Xm[xnp1_i0:xnp1_if]
                #W = np.reshape(p[W_i0:W_if], (self.structure[n+1], self.structure[n]))
                #b = p[b_i0:b_if]
                diff[diff_i0:diff_if] = xnp1 - self.disc(xn, W[n], b[n])

                if n < self.N - 2:
                    diff_i0 = diff_if
                    diff_if = diff_i0 + self.structure[n+2]
                    xn_i0 = xn_if
                    xn_if = xn_i0 + self.structure[n+1]
                    xnp1_i0 = xnp1_if
                    xnp1_if = xnp1_i0 + self.structure[n+2]
                    W_i0 = b_if
                    W_if = W_i0 + self.structure[n+1]*self.structure[n+2]
                    b_i0 = W_if
                    b_if = b_i0 + self.structure[n+2]
            if m == 0:
                ferr = self.RF * np.sum(diff * diff)
            else:
                ferr = ferr + self.RF * np.sum(diff * diff)
        #print diff
        #exit(0)
        #if isinstance(diff[0], adolc._adolc.adub):
        #    pass
        #else:
        #    print diff
        #    exit(0)
        #ferr = self.RF * np.sum(diff * diff)
        return ferr / np.float64((self.NDnet - self.structure[0]) * self.M)

    ############################################################################
    # Discretization routines
    ############################################################################
    def disc_forwardmap(self, x, W, b):
        """
        "Discretization" when f is a forward mapping, not an ODE.
        """
        return self.f(x, W, b)

    ############################################################################
    # Annealing functions
    ############################################################################
    def anneal(self, X0, P0, alpha, beta_array, RM, RF0, Pidx, Lidx=None,
               init_to_data=True, action='A_gaussian', disc='forwardmap', 
               method='L-BFGS-B', bounds=None, opt_args=None, adolcID=0):
        """
        Convenience function to carry out a full annealing run over all values
        of beta in beta_array.
        """
        # initialize the annealing procedure, if not already done
        if self.annealing_initialized == False:
            self.anneal_init(X0, P0, alpha, beta_array, RM, RF0, Pidx, Lidx,
                             init_to_data, action, disc, method, bounds,
                             opt_args, adolcID)
        for i in beta_array:
            print('------------------------------')
            print('Step %d of %d'%(self.betaidx+1, len(self.beta_array)))
            print('beta = %d, RF = %.8e'%(self.beta, self.RF))
            print('')
            self.anneal_step()

    def anneal_init(self, X0, P0, alpha, beta_array, RM, RF0, Pidx, Lidx=None,
                    init_to_data=True, action='A_gaussian', disc='forwardmap', 
                    method='L-BFGS-B', bounds=None, opt_args=None, adolcID=0):
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
            self.bounds = None
        else:
            self.bounds = np.array(bounds)

        # get optimization extra arguments
        self.opt_args = opt_args

        # set up parameters and determine if static or time series
        self.P = P0
        self.NP = len(P0)
        #if P0.ndim == 1:
        #    # Static parameters, so p is a single vector.
        #    self.NP = len(P0)
        #else:
        #    # Time-dependent parameters, so p is a time series of N values.
        #    self.NP = P0.shape[1]

        # get indices of parameters to be estimated by annealing
        self.Pidx = Pidx
        self.NPest = len(Pidx)

        # get indices of measured components
        if Lidx is None:
            self.Lidx = [np.linspace(0, self.structure[0]-1, self.structure[0]),
                         np.linspace(0, self.structure[-1]-1, self.structure[-1])]
        else:
            self.Lidx = Lidx

        self.L = [len(self.Lidx[0]), len(self.Lidx[1])]
        self.Ltot = self.L[0] + self.L[1]

        # Simply set RM here; measurement error function should decide how to handle it
        self.RM = RM

        ## get indices of measured components of f
        #self.Lidx = Lidx
        #self.L = len(Lidx)
        #
        ## properly set up the bounds arrays
        #if bounds is not None:
        #    bounds_full = []
        #    state_b = bounds[:self.D]
        #    param_b = bounds[self.D:]
        #    # set bounds on states for all N time points
        #    for n in xrange(self.N):
        #        for i in xrange(self.D):
        #            bounds_full.append(state_b[i])
        #    # set bounds on parameters
        #    if self.P.ndim == 1:
        #        # parameters are static
        #        for i in xrange(self.NPest):
        #            bounds_full.append(param_b[i])
        #    else:
        #        # parameters are time-dependent
        #        for n in xrange(self.N):
        #            for i in xrange(self.NPest):
        #                bounds_full.append(param_b[i])
        #else:
        #    bounds_full = None
        #
        ## Reshape RM and RF so that they span the whole time series.  This is
        ## done because in the action evaluation, it is more efficient to let
        ## numpy handle multiplication over time rather than using python loops.
        #if type(RM) == np.ndarray:
        #    if RM.shape == (self.L,):
        #        self.RM = np.resize(RM, (self.N, self.L))
        #    elif RM.shape == (self.L, self.L):
        #        self.RM = np.resize(RM, (self.N, self.L, self.L))
        #    elif RM.shape == (self.N, self.L) or RM.shape == np.resize(self.N, self.L, self.L):
        #        self.RM = RM
        #    else:
        #        print("ERROR: RM has an invalid shape. Exiting.")
        #        exit(1)
        #
        #else:
        #    self.RM = RM
        #
        #if type(RF0) == np.ndarray:
        #    if RF0.shape == (self.D,):
        #        self.RF0 = np.resize(RF0, (self.N - 1, self.D))
        #    elif RF0.shape == (self.D, self.D):
        #        self.RF0 = np.resize(RF0, (self.N - 1, self.D, self.D))
        #    elif RF0.shape == (self.N - 1, self.D) or RF0.shape == (self.N - 1, self.D, self.D):
        #        self.RF0 = RF0
        #    else:
        #        print("ERROR: RF0 has an invalid shape. Exiting.")
        #        exit(1)
        #
        #else:
        #    self.RF0 = RF0

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
        self.minpaths = np.zeros((self.Nbeta, self.NDens + self.NP), dtype=np.float64)
        #if P0.ndim == 1:
        #    self.minpaths = np.zeros((self.Nbeta, self.N*self.D + self.NP), dtype=np.float64)
        #else:
        #    self.minpaths = np.zeros((self.Nbeta, self.N*(self.D + self.NP)), dtype=np.float64)

        # initialize observed state components to data if desired
        if init_to_data == True:
            for m in xrange(self.M):
                i0 = m * self.NDnet
                i1 = i0 + self.structure[0]
                i2 = (m+1) * self.NDnet - self.structure[-1]
                i3 = (m+1) * self.NDnet
                X0[i0:i1][self.Lidx[0]] = self.data_in[m]
                X0[i2:i3][self.Lidx[1]] = self.data_out[m]

        #if self.NPest > 0:
        #    if P0.ndim == 1:
        #        XP0 = np.append(X0.flatten(), P0)
        #    else:
        #        XP0 = np.append(X0.flatten(), P0.flatten())
        #else:
        #    XP0 = X0.flatten()

        XP0 = np.append(X0, P0)
        self.minpaths[0] = XP0

        # array to store optimization results
        self.A_array = np.zeros(self.Nbeta, dtype=np.float64)
        self.me_array = np.zeros(self.Nbeta, dtype=np.float64)
        self.fe_array = np.zeros(self.Nbeta, dtype=np.float64)
        self.exitflags = np.empty(self.Nbeta, dtype=np.int8)

        # set the adolcID
        self.adolcID = adolcID

        # Initialization successful, we're at the beta = beta_0 step now.
        self.initalized = True

    def anneal_step(self):
        """
        Perform a single annealing step. The cost function is minimized starting
        from the previous minimum (or the initial guess, if this is the first
        step). Then, RF is increased to prepare for the next annealing step.
        """
        # minimize A using the chosen method
        if self.method in ['L-BFGS-B', 'NCG', 'TNC']:#, 'LM']:
            if self.betaidx == 0:
                if self.NP == self.NPest:
                    XP0 = np.copy(self.minpaths[0])
                else:
                    X0 = self.minpaths[0][:self.NDens]
                    P0 = self.minpaths[0][self.NDens:][self.Pidx]
                    XP0 = np.append(X0, P0)
            else:
                if self.NP == self.NPest:
                    XP0 = np.copy(self.minpaths[self.betaidx-1])
                else:
                    X0 = self.minpaths[self.betaidx-1][:self.NDens]
                    P0 = self.minpaths[self.betaidx-1][self.NDens:][self.Pidx]
                    XP0 = np.append(X0, P0)

            if self.method == 'L-BFGS-B':
                XPmin, Amin, exitflag = self.min_lbfgs_scipy(XP0, self.gen_xtrace())
            elif self.method == 'NCG':
                XPmin, Amin, exitflag = self.min_cg_scipy(XP0, self.gen_xtrace())
            elif self.method == 'TNC':
                XPmin, Amin, exitflag = self.min_tnc_scipy(XP0, self.gen_xtrace())
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
            #if isinstance(XPmin[0], adolc._adolc.adouble):
            if isinstance(XPmin[0], adolc._adolc.adub):
                Pest_vals = [XPmin[-self.NPest + i].val for i in xrange(self.NPest)]
                self.P[self.Pidx] = np.array(Pest_vals)
            else:
                self.P[self.Pidx] = np.copy(XPmin[-self.NPest:])

        # store A_min and the minimizing path
        self.A_array[self.betaidx] = Amin
        self.me_array[self.betaidx] = self.me_gaussian(np.array(XPmin))
        self.fe_array[self.betaidx] = self.fe_gaussian(np.array(XPmin))
        self.minpaths[self.betaidx] = np.array(np.append(XPmin[:self.NDens], self.P))

        # increase RF
        if self.betaidx < len(self.beta_array) - 1:
            self.betaidx += 1
            self.beta = self.beta_array[self.betaidx]
            self.RF = self.RF0 * self.alpha**self.beta

        # set flags indicating that A needs to be retaped, and that we're no
        # longer at the beginning of the annealing procedure
        self.taped = False
        if self.annealing_initialized == True:
            # Indicate no longer at initial beta value
            self.initialized = False

    ################################################################################
    # Routines to save annealing results.
    ################################################################################
    def save_states(self, filename):
        """
        Save minimizing neuron states (not including parameters).
        """
        #savearray = np.reshape(self.minpaths[:, :self.NDens], (self.Nbeta, self.NDens))

        savearray = []
        for m in xrange(self.M):
            savearray.append([])
            for b in xrange(self.Nbeta):
                savearray[m].append([])
                i0 = m * self.NDnet
                i1 = i0 + self.structure[0]
                for n in xrange(self.N):
                    savearray[m][b].append(self.minpaths[b, i0:i1])
                    i0 = i1
                    i1 = i0 + self.structure[n]
        savearray = np.array(savearray)

        if filename.endswith('.npy'):
            np.save(filename, savearray)
        else:
            np.savetxt(filename, savearray)

    def save_io(self, filename):
        """
        Save minimizing input/output neuron states (not including parameters).
        """
        #savearray = np.reshape(self.minpaths[:, :self.NDens], (self.Nbeta, self.NDens))

        savearray = []
        for m in xrange(self.M):
            savearray.append([])
            for b in xrange(self.Nbeta):
                savearray[m].append([])
                i0 = m * self.NDnet
                i1 = i0 + self.structure[0]
                savearray[m][b].append(self.minpaths[b, i0:i1])
                i0 = (m+1) * self.NDnet - self.structure[-1]
                i1 = (m+1) * self.NDnet
                savearray[m][b].append(self.minpaths[b, i0:i1])
        savearray = np.array(savearray)

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
            est_param_array = self.minpaths[:, self.NDnet:]
            savearray[:, self.Pidx] = est_param_array

        if filename.endswith('.npy'):
            np.save(filename, savearray)
        else:
            np.savetxt(filename, savearray)

    def save_Wb(self, W_filename, b_filename):
        """
        Save W and b in separate files.
        """
        # write fixed parameters to array
        W = []
        b = []
        for i in xrange(self.Nbeta):
            W.append([])
            b.append([])
            p_i0 = self.NDens
            p_i1 = self.NDens + self.structure[0]*self.structure[1] + self.structure[1]
            for n in xrange(self.N-1):
                pn = self.minpaths[i, p_i0:p_i1]
                W[i].append(np.reshape(pn[:self.structure[n]*self.structure[n+1]],
                                       (self.structure[n+1], self.structure[n])))
                b[i].append(pn[-self.structure[n+1]:])

                if n < self.N - 2:
                    p_i0 = p_i1
                    p_i1 += self.structure[n+1]*self.structure[n+2] + self.structure[n+2]

        np.save(W_filename, W)
        np.save(b_filename, b)

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
        savearray[:, 4] = self.fe_array / (self.RF0 * self.alpha**self.beta_array)
        #if type(self.RF) == np.ndarray:
        #    if self.RF0.shape == (self.N - 1, self.D):
        #        savearray[:, 4] = self.fe_array / (self.RF0[0, 0] * self.alpha**self.beta_array)
        #    elif self.RF0.shape == (self.N - 1, self.D, self.D):
        #        savearray[:, 4] = self.fe_array / (self.RF0[0, 0, 0] * self.alpha**self.beta_array)
        #    else:
        #        print("RF shape currently not supported for saving.")
        #        return 1
        #else:
        #    savearray[:, 4] = self.fe_array / (self.RF0 * self.alpha**self.beta_array)

        if filename.endswith('.npy'):
            np.save(filename, savearray)
        else:
            np.savetxt(filename, savearray)

    ############################################################################
    # AD taping & derivatives
    ############################################################################
    def gen_xtrace(self):
        """
        Define a random state vector for the AD trace.
        """
        xtrace = np.random.rand(self.NDens + self.NPest)
        return xtrace
