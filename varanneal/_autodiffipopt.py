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
import pyipopt
import scipy.sparse as sps

class Eval_jac_g(object):
    """
    Callable object for evaluation of constraint Jacobian(s).
    """
    def __init__(self, XP, adolcID=0):
        self.gID = adolcID + 1
        options = np.array([1, 1, 0, 0], dtype=int)
        result = adolc.colpack.sparse_jac_no_repeat(self.gID, XP, options)

        self.nnz  = result[0]     
        self.rind = np.asarray(result[1],dtype=int)
        self.cind = np.asarray(result[2],dtype=int)
        self.values = np.asarray(result[3],dtype=float)

    def __call__(self, XP, flag, user_data = None):
        if flag:
            return (self.rind, self.cind)
        else:
            result = adolc.colpack.sparse_jac_repeat(self.gID, XP, self.nnz, self.rind,
                self.cind, self.values)
            return result[3]

class Eval_hess_lgr(object):
    """
    Callable object for evaluation of Lagrangian Hessian.
    """
    def __init__(self, XP, lgr_mult, obj_fac, adolcID=0):
        self.hID = adolcID + 2
        Dpath = len(XP)
        options = np.array([0, 0], dtype=int)
        x = np.hstack([XP, lgr_mult, obj_fac])
        result = adolc.colpack.sparse_hess_no_repeat(self.hID, x, options)

        self.nnz = result[0]
        self.rind = np.asarray(result[1],dtype=int)
        self.cind = np.asarray(result[2],dtype=int)
        self.values = np.asarray(result[3],dtype=float)
        
        # need only upper left part of the Hessian
        self.mask = np.where(self.cind < Dpath)
        
    def __call__(self, XP, lgr_mult, obj_fac, flag, user_data=None):
        if flag:
            return (self.rind[self.mask], self.cind[self.mask])
        else:
            x = np.hstack([XP, lgr_mult, obj_fac])
            result = adolc.colpack.sparse_hess_repeat(self.hID, x, self.nnz, self.rind,
                self.cind, self.values)
            return result[3][self.mask]

class ADipopt(object):
    """
    ADmin is an object type for using AD ad implemented in ADOL-C to minimize
    arbitrary scalar functions, i.e. functions f s.t. f: R^N --> R.
    """
    def __init__(self, A, adolcID=0, sparseprob=False):
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
        self.A = A
        self.adolcID = adolcID
        #self.sparseprob = sparseprob
        self.sparseprob = True  # ignore non-sparse indication

    ############################################################################
    # Ipopt-relevant stuff
    ############################################################################
    def set_ipopt_constraints(self, g, x_L, x_U, g_L, g_U):
        """
        Set constraint functions if using IPOPT.
        """
        self.g = g
        self.x_L = x_L
        self.x_U = x_U
        self.g_L = g_L
        self.g_U = g_U
        #self.ncon = len(g)
        self.ncon = 1

        #if self.sparseprob:
        #    # Initialize sparse objects to null pointers; later when
        #    # constraints are taped, replace list elements on evaluation.
        #    self.g_nnz = [None] * self.ncon
        #    self.g_rind = [None] * self.ncon
        #    self.g_cind = [None] * self.ncon
        #    self.g_values = [None] * self.ncon

        if self.sparseprob:
            # Initialize sparse objects to null pointers; later when
            # constraints are taped, replace list elements on evaluation.
            self.g_nnz = None
            self.g_rind = None
            self.g_cind = None
            self.g_values = None

    def lgr(self, XP, lgr_mult, obj_fac, user_data=None):
        """
        "Lagrangian" = action + constraints
        """
        return obj_fac * self.A(XP) + np.dot(lgr_mult, self.g(XP))

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
        #self.A_taped_flag = True
        print('Done!')
        print('Time = {0} s\n'.format(time.time()-tstart))

    def tape_g(self, xtrace):
        """
        Trace the constraint functions (used with IPOPT).
        """
        print("Taping constraint function(s)...")
        tstart = time.time()

        #for i,gi in enumerate(self.g):
        #    adolc.trace_on(self.adolcID + 1 + i)
        #    ax = adolc.adouble(xtrace)
        #    adolc.independent(ax)
        #    af = gi(ax)
        #    adolc.dependent(af)
        #    adolc.trace_off()
        adolc.trace_on(self.adolcID + 1)
        ax = adolc.adouble(xtrace)
        adolc.independent(ax)
        af = self.g(ax)
        adolc.dependent(af)
        adolc.trace_off()
        #self.g_taped_flag = True
        print("Done!")
        print("Time = {0} s\n".format(time.time() - tstart))

        if self.sparseprob:
            #self.g_taped_flag = False  # in case sparsity analysis fails
            print("Determining sparsity structure of constraint Jacobian(s)...")
            tstart = time.time()
            self.jacg_taped_sparse = Eval_jac_g(xtrace, self.adolcID)
            print("Done!")
            print("Time = {0} s\n".format(time.time() - tstart))
            #self.g_taped_flag = True

    def tape_lgr(self, xtrace, lgr_mult_trace, obj_fac_trace, nconst=1):
        """
        Tape the objective function with Lagrange multiplier terms (used with IPOPT).
        """
        print("Taping the Lagrangian...")
        tstart = time.time()

        adolc.trace_on(self.adolcID + 2)
        ax = adolc.adouble(xtrace)
        algr_mult = adolc.adouble(lgr_mult_trace)
        aobj_fac = adolc.adouble(obj_fac_trace)
        adolc.independent(ax)
        adolc.independent(algr_mult)
        adolc.independent(aobj_fac)
        af = self.lgr(ax, algr_mult, aobj_fac)
        adolc.dependent(af)
        adolc.trace_off()
        #self.lgr_taped_flag = True
        print("Done!")
        print("Time = {0} s\n".format(time.time() - tstart))

        if self.sparseprob:
            #self.lgr_taped_flag = False  # in case sparsity analysis fails
            print("Determining sparsity structure of Lagrangian Hessian...")
            tstart = time.time()
            self.hesslgr_taped_sparse = Eval_hess_lgr(xtrace, lgr_mult_trace,
                                                   obj_fac_trace, self.adolcID)
            print("Done!")
            print("Time = {0} s\n".format(time.time() - tstart))
            #self.lgr_taped_flag = True

    # Taped action callable functions
    def A_taped(self, XP):
        return adolc.function(self.adolcID, XP)
    
    def gradA_taped(self, XP):
        return adolc.gradient(self.adolcID, XP)

    def A_gradA_taped(self, XP):
        return adolc.function(self.adolcID, XP), adolc.gradient(self.adolcID, XP)

    def jacA_taped(self, XP):
        return adolc.jacobian(self.adolcID, XP)

    def A_jacA_taped(self, XP):
        return adolc.function(self.adolcID, XP), adolc.jacobian(self.adolcID, XP)

    def hessianA_taped(self, XP):
        return adolc.hessian(self.adolcID, XP)

    # Constraint and Lagrangian callable functions
    def g_taped(self, XP):
        return adolc.function(self.adolcID + 1, XP)

    def jacg_taped(self, XP, flag, user_data=None):
        if flag:
            return (self.g_rind, self.g_cind)
        else:
            result = adolc.colpack.sparse_jac_repeat(self.adolcID+1, XP, self.g_nnz,
                                                     self.g_rind, self.g_cind, self.g_values)
    #        return result[3]
    #
    #def hesslgr_taped(self, XP, lgr_mult, obj_fac, flag, user_data=None):
    #    if flag:
    #        return (self.lgr_rind[self.lgr_mask], self.cind[self.lgr_mask])
    #    else:
    #        x = np.hstack((XP, lagrange, obj_factor))
    #        result = adolc.colpack.sparse_hess_repeat(self.adolcID+2, x,
    #                                                  self.lgr_rind, self.lgr_cind, self.lgr_values)
    #        return result[3][self.lgr_mask]

    def jacg_taped_dense(self, XP):
        return adolc.jacobian(self.adolcID + 1, XP)
    
    def hesslgr_taped_dense(self, XP):
        return adolc.hessian(self.adolcID + 2, XP)

    ################################################################################
    # Minimization functions
    ################################################################################
    def min_ipopt(self, XP0, nvar, xtrace=None, lgr_mult_trace=None, obj_fac_trace=None,
                  A_taped_flag=False, g_taped_flag=False, lgr_taped_flag=False):
        """
        Minimize the action starting from XP0 using IPOPT.
        """
        if not A_taped_flag:
            self.tape_A(xtrace)
        if not g_taped_flag:
            self.tape_g(xtrace)
        if not lgr_taped_flag:
            self.tape_lgr(xtrace, lgr_mult_trace, obj_fac_trace)

        # Start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        #if self.sparseprob:
        #    jacg_taped = self.jacg_taped_sparse
        #    hesslgr_taped = self.hesslgr_taped_sparse
        #else:
        #    jacg_taped = self.jacg_taped_dense
        #    hesslgr_taped = self.hesslgr_taped_dense
        jacg_taped = self.jacg_taped_sparse
        hesslgr_taped = self.hesslgr_taped_sparse
        nnzj = jacg_taped.nnz
        nnzh = hesslgr_taped.nnz
        nlp = pyipopt.create(nvar, self.x_L, self.x_U, 1, self.g_L, self.g_U,
                             nnzj, nnzh, self.A_taped, self.gradA_taped,
                             self.g_taped, jacg_taped, hesslgr_taped)
        res = nlp.solve(XP0)
        XPmin, Amin, status = res['x'], res['f'], 0

        print("Optimization complete!")
        print("Time = {0} s".format(time.time()-tstart))
        #print("Exit flag = {0}".format(status))
        #print("Exit message: {0}".format(res.message))
        #print("Iterations = {0}".format(res.nit))
        print("Obj. function value = {0}\n".format(Amin))
        return XPmin, Amin, status#, self.Ataped_flag, self.g_taped, self.lgr_taped
