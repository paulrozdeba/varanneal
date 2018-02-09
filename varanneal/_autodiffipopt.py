"""
Paul Rozdeba (prozdeba@physics.ucsd.edu)
Department of Physics
University of California, San Diego
May 23, 2017

Functions and base class definitions common to all system types using 
variational annealing.
"""

import numpy as np
import adolc
import time
import scipy.sparse as sps

class ADipopt(object):
    """
    ADmin is an object type for using AD ad implemented in ADOL-C to minimize
    arbitrary scalar functions, i.e. functions f s.t. f: R^N --> R.
    """
    def __init__(self, sparseprob=False):
        """
        These routines are the same for all system types and their variables
        are set in the Annealer objects which inherit ADmin when IPOPT is used.
        Here, only ask the user if the problem is sparse using the flag
        sparseprob.
        *** NOTE: The code doesn't handle non-sparse problems yet!  Even if
        the problem is dense and the user feeds in sparseprob = False,
        this fact is ignored for the time being and the program proceeds as if
        it is sparse.
        """
        #self.sparseprob = sparseprob
        self.sparseprob = True  # ignore non-sparse indication

    ############################################################################
    # Ipopt-relevant stuff
    ############################################################################
    def set_ipopt_constraints(self, g, x_L, x_U, g_L, g_U):
        """
        Set constraint functions if using IPOPT.
        """
        self.ipopt_g = g
        self.ipopt_x_L = x_L
        self.ipopt_x_U = x_U
        self.ipopt_g_L = g_L
        self.ipopt_g_U = g_U
        self.ncon = len(g)

    ############################################################################
    # AD taping & derivatives
    ############################################################################
    def tape_A(self, xtrace):
        """
        Tape the objective function.
        """
        print('Taping action evaluation...')
        tstart = time.time()

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

    def tape_g(self, xtrace):
        """
        Trace the constraint functions (used with IPOPT).
        """
        print("Taping constraint functions...")
        tstart = time.time()

        adolc.trace_on(self.adolcID + 1)
        ax = adolc.adouble(xtrace)
        adolc.independent(ax)
        af = self.ipopt_g(ax)
        adolc.dependent(af)
        adolc.trace_off()
        self.ipopt_g_taped = True
        print("Done!")
        print("Time = {0} s\n".format(time.time() - tstart))

        if self.sparseprob:
            self.ipopt_g_taped = False  # in case sparsity detection fails
            print("Determining sparsity structure of constraint Jacobian...")
            tstart = time.time()

            options = np.array([1, 1, 0, 0], dtype=int)
            result = adolc.colpack.sparse_jac_no_repeat(self.adolcID+1, x, options)

            self.g_nnz = result[0]
            self.g_rind = np.asarray(result[1], dtype=int)
            self.g_cind = np.asarray(result[2], dtype=int)
            self.g_values = np.asarray(result[3], dtype=int)

            print("Done!")
            print("Time = {0} s\n".format(time.time() - tstart))
            self.ipopt_g_taped = True

    def tape_lgr(self, xtrace, lgr_mult_trace, obj_fac_trace, nconst):
        """
        Tape the objective function with Lagrange multiplier terms (used with IPOPT).
        """
        print("Taping the Lagrangian...")
        tstart = time.time()

        adolc.trace_on(self.adolcID + 2)
        ax = adolc.adouble(xtrace)
        algr_mult = adolc.adouble(lgr_mult_trace)
        aobj_factor = adolc.adouble(obj_fac_trace)
        adolc.independent(ax)
        adolc.independent(algr_mult)
        adolc.independent(aobj_factor)
        af = self.ipopt_lgr(ax, algr_mult, aobj_factor)
        adolc.dependent(af)
        adolc.trace_off()
        self.ipopt_lgr_taped = True
        print("Done!")
        print("Time = {0} s\n".format(time.time() - tstart))

        if self.sparseprob:
            self.ipopt_lgr_taped = False  # in case sparsity detection fails
            print("Determining sparsity structure of Lagrangian Hessian...")
            tstart = time.time()

            options = np.array([0, 0], dtype=int)
            result = adolc.colpack.sparse_hess_no_repeat(self.h_adolcID+2, x, options)

            self.lgr_rind = np.asarray(result[1], dtype=int)
            self.lgr_cind = np.asarray(result[2], dtype=int)
            self.lgr_values = np.asarray(result[3], dtype=int)

            # Need only upper left part of Hessian.
            self.lgr_mask = np.where(self.lgr_cind < len(x))

            print("Done!")
            print("Time = {0} s\n".format(time.time() - tstart))
            self.ipopt_lgr_taped = True

    # Taped action callable functions
    def A_taped(self, XP):
        return adolc.function(self.adolcID, XP)
    
    def gradA_taped(self, XP):
        return adolc.gradient(self.adolcID, XP)

    def A_gradA_taped(self, XP):
        return adolc.function(self.adolcID, XP), adolc.gradient(self.adolcID, XP)

    def jacA_taped(self, XP):
        return adolc.jacobian(self.adolcID, XP)

    def A_jacaA_taped(self, XP):
        return adolc.function(self.adolcID, XP), adolc.jacobian(self.adolcID, XP)

    def hessianA_taped(self, XP):
        return adolc.hessian(self.adolcID, XP)

    # Constraint and Lagrangian callable functions
    def jacg_taped(self, XP, flag, user_data=None):
        if flag:
            return (self.g_rind, self.g_cind)
        else:
            result = adolc.colpack.sparse_jac_repeat(self.adolcID+1, XP, self.g_nnz, self.g_rind, self.g_cind, self.g_values)
            return result[3]

    def hesslgr_taped(self, XP, lgr_mult, obj_fac, flag, user_data=None):
        if flag:
            return (self.lgr_rind[self.lgr_mask], self.cind[self.lgr_mask])
        else:
            x = np.hstack((XP, lagrange, obj_factor))
            result = adolc.colpack.sparse_hess_repeat(self.adolcID+2, x, self.lgr_rind, self.lgr_cind, self.lgr_values)
            return result[3][self.lgr_mask]

    ################################################################################
    # Minimization functions
    ################################################################################
    def min_ipopt(self, XP0, xtrace=None, lgr_mult_trace=None, obj_fac_trace=None):
        """
        Minimize the action starting from XP0 using IPOPT.
        """
        if self.taped == False:
            self.tape_A(xtrace)
        if self.ipopt_g_taped == False:
            self.tape_g(xtrace)
        if self.ipopt_lgr_traced == False:
            self.tape_lgr(xtrace)

        # Start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h)
        res = nlp.solve(XP0)
        XPmin, Amin, status = res['x'], res['f'], 0

        print("Optimization complete!")
        print("Time = {0} s".format(time.time()-tstart))
        #print("Exit flag = {0}".format(status))
        #print("Exit message: {0}".format(res.message))
        #print("Iterations = {0}".format(res.nit))
        print("Obj. function value = {0}\n".format(Amin))
        return XPmin, Amin, status
