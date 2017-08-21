"""
Paul Rozdeba (prozdeba@physics.ucsd.edu)
Department of Physics
University of California, San Diego
May 23, 2017

Functions and base class definitions common to all versions of systems using 
variational annealing.
"""

import numpy as np
import adolc
import scipy.optimize as opt

class ADmin(object):
    def __init__(self):
        """
        These routines are the same for all system types and their variables
        are set in the Annealer objects which inherit ADmin, so nothing special
        to do here really.
        """
        pass

    ############################################################################
    # AD taping & derivatives
    ############################################################################
    def tape_A(self, xtrace):
        """
        Tape the objective function.
        """
        print('Taping action evaluation...')
        tstart = time.time()

        # define a random state vector for the trace
        if self.systype == 'ode':
            if self.P.ndim == 1:
                xtrace = np.random.rand(self.N_model*self.D + self.NPest)
            else:
                if self.disc.im_func.__name__ in ["disc_euler", "disc_forwardmap"]:
                    xtrace = np.random.rand(self.N_model*self.D + (self.N_model-1)*self.NPest)
                else:
                    xtrace = np.random.rand(self.N_model*(self.D + self.NPest))
        elif self.systype == 'nnet':
            xtrace = np.random.rand(self.NDens + self.NPest)
        else:
            print("ERROR: Invalid systype. This shouldn't happen unless you " + \
                  "tried to manually set this yourself, which is a no no.")
            exit(1)

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

    ################################################################################
    # Minimization functions
    ################################################################################
    def min_lbfgs_scipy(self, XP0):
        """
        Minimize f starting from XP0 using L-BFGS-B method in scipy.
        This method supports the use of bounds.
        Returns the minimizing state, the minimum function value, and the L-BFGS
        termination information.
        """
        if self.taped == False:
            self.tape_A()

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
        Minimize f starting from XP0 using nonlinear CG method in scipy.
        Returns the minimizing state, the minimum function value, and the CG
        termination information.
        """
        if self.taped == False:
            self.tape_A()

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.minimize(self.A_gradA_taped, XP0, method='CG', jac=True,
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
        Minimize f starting from XP0 using Newton-CG method in scipy.
        Returns the minimizing state, the minimum function value, and the CG
        termination information.
        """
        if self.taped == False:
            self.tape_A()

        # start the optimization
        print("Beginning optimization...")
        tstart = time.time()
        res = opt.minimize(self.A_gradA_taped, XP0, method='TNC', jac=True,
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
        Minimize f starting from XP0 using Levenberg-Marquardt in scipy.
        Returns the minimizing state, the minimum function value, and the CG
        termination information.
        """
        if self.taped == False:
            self.tape_A()

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

    #def min_lm_scipy(self, XP0):
    #    """
    #    Minimize f starting from XP0 using Levenberg-Marquardt in scipy.
    #    Returns the minimizing state, the minimum function value, and the CG
    #    termination information.
    #    """
    #    if self.taped == False:
    #        self.tape_A()
    #
    #    # start the optimization
    #    print("Beginning optimization...")
    #    tstart = time.time()
    #    res = opt.root(self.A_jacA_taped, XP0, method='lm', jac=True,
    #                   options=self.opt_args)
    #
    #    XPmin,status,Amin = res.x, res.status, res.fun
    #
    #    print("Optimization complete!")
    #    print("Time = {0} s".format(time.time()-tstart))
    #    print("Exit flag = {0}".format(status))
    #    print("Exit message: {0}".format(res.message))
    #    print("Iterations = {0}".format(res.nit))
    #    print("Obj. function value = {0}\n".format(Amin))
    #    return XPmin, Amin, status
