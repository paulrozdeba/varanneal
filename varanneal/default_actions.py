"""
Set of common actions to use in variational DA.
"""

import numpy as np

class GaussianAction(object):
    """
    Action with Gaussian measurement and model errors.
    """
    def __init__(self, N_model, D, merr_nskip, L, Lidx, Y, RM, N_data,
                 P, NP, NPest, disc, RF, stim, f, t_model, dt_model):
        self.N_model = N_model
        self.D = D
        self.merr_nskip = merr_nskip
        self.L = L
        self.Lidx = Lidx
        self.Y = Y
        self.RM = RM
        self.N_data = N_data
        self.P = P
        self.NP = NP
        self.NPest = NPest
        self.disc_str = disc
        self.RF = RF
        self.stim = stim
        self.f = f
        self.t_model = t_model
        self.dt_model = dt_model

        exec "self.disc = self.disc_%s" % (self.disc_str)

        # All actions must have measurement and model error functions called
        # meas_err and model_err, respectively.
        self.meas_err = self.me_gaussian
        self.model_err = self.fe_gaussian

    ############################################################################
    # Function definition
    ############################################################################
    def __call__(self, XP):
        """
        Callable.
        """
        merr = self.me_gaussian(XP[:self.N_model * self.D])
        ferr = self.fe_gaussian(XP)
        return merr + ferr

    def me_gaussian(self, X):
        """
        Gaussian measurement error.
        """
        x = np.reshape(X, (self.N_model, self.D))
        diff = x[::self.merr_nskip, self.Lidx] - self.Y

        if type(self.RM) == np.ndarray:
            # Contract RM with error
            if self.RM.shape == (self.N_data, self.L):
                merr = np.sum(self.RM * diff * diff)
            elif self.RM.shape == (self.N_data, self.L, self.L):
                merr = 0.0
                for i in xrange(self.N_data):
                    merr = merr + np.dot(diff[i], np.dot(self.RM[i], diff[i]))
            else:
                print("ERROR: RM is in an invalid shape.")
        else:
            merr = self.RM * np.sum(diff * diff)

        return merr / (self.L * self.N_data)

    def fe_gaussian(self, XP):
        """
        Gaussian model error.
        """
        # Extract state and parameters from XP.
        if self.NPest == 0:
            x = np.reshape(XP, (self.N_model, self.D))
            p = self.P
        elif self.NPest == self.NP:
            x = np.reshape(XP[:self.N_model*self.D], (self.N_model, self.D))
            if self.P.ndim == 1:
                p = XP[self.N_model*self.D:]
            else:
                if self.disc.im_func.__name__ in ["disc_euler", "disc_forwardmap"]:
                    p = np.reshape(XP[self.N_model*self.D:], (self.N_model - 1, self.NPest))
                else:
                    p = np.reshape(XP[self.N_model*self.D:], (self.N_model, self.NPest))
        else:
            x = np.reshape(XP[:self.N_model*self.D], (self.N_model, self.D))
            p = np.array(self.P, dtype=XP.dtype)
            if self.P.ndim == 1:
                p[self.Pidx] = XP[self.N_model*self.D:]
            else:
                if self.disc.im_func.__name__ in ["disc_euler", "disc_forwardmap"]:
                    p[:, self.Pidx] = np.reshape(XP[self.N_model*self.D:],
                                                 (self.N_model-1, self.NPest))
                else:
                    p[:, self.Pidx] = np.reshape(XP[self.N_model*self.D:],
                                                 (self.N_model, self.NPest))

        # Start calculating the model error.
        # First compute time series of error terms.
        if self.disc.im_func.__name__ == "disc_SimpsonHermite":
            disc_vec1, disc_vec2 = self.disc(x, p)
            diff1 = x[2::2] - x[:-2:2] - disc_vec1
            diff2 = x[1::2] - disc_vec2
        elif self.disc.im_func.__name__ == 'disc_forwardmap':
            diff = x[1:] - self.disc(x, p)
        else:
            diff = x[1:] - x[:-1] - self.disc(x, p)

        # Contract errors quadratically with RF.
        if type(self.RF) == np.ndarray:
            if self.RF.shape == (self.N_model - 1, self.D):
                if self.disc.im_func.__name__ == "disc_SimpsonHermite":
                    ferr1 = np.sum(self.RF[::2] * diff1 * diff1)
                    ferr2 = np.sum(self.RF[1::2] * diff2 * diff2)
                    ferr = ferr1 + ferr2
                else:
                    ferr = np.sum(self.RF * diff * diff)

            elif self.RF.shape == (self.N_model - 1, self.D, self.D):
                if self.disc.im_func.__name__ == "disc_SimpsonHermite":
                    ferr1 = 0.0
                    ferr2 = 0.0
                    for i in xrange((self.N_model - 1) / 2):
                        ferr1 = ferr1 + np.dot(diff1[i], np.dot(self.RF[2*i], diff1[i]))
                        ferr2 = ferr2 + np.dot(diff2[i], np.dot(self.RF[2*i+1], diff2[i]))
                    ferr = ferr1 + ferr2
                else:
                    ferr = 0.0
                    for i in xrange(self.N_model - 1):
                        ferr = ferr + np.dot(diff[i], np.dot(self.RF[i], diff))

            else:
                print("ERROR: RF is in an invalid shape. Exiting.")
                sys.exit(1)

        else:
            if self.disc.im_func.__name__ == "disc_SimpsonHermite":
                ferr = self.RF * np.sum(diff1 * diff1 + diff2 * diff2)
            else:
                ferr = self.RF * np.sum(diff * diff)

        return ferr / (self.D * (self.N_model - 1))

    ############################################################################
    # Discretization routines
    ############################################################################
    def disc_euler(self, x, p):
        """
        Euler's method for time discretization of f.
        """
        if self.stim is None:
            if self.P.ndim == 1:
                pn = p
            else:
                pn = p[:-1]
        else:
            if self.P.ndim == 1:
                pn = (p, self.stim[:-1])
            else:
                pn = (p[:-1], self.stim[:-1])

        return self.dt_model * self.f(self.t_model[:-1], x[:-1], pn)

    def disc_trapezoid(self, x, p):
        """
        Time discretization for the action using the trapezoid rule.
        """
        if self.stim is None:
            if self.P.ndim == 1:
                pn = p
                pnp1 = p
            else:
                pn = p[:-1]
                pnp1 = p[1:]
        else:
            if self.P.ndim == 1:
                pn = (p, self.stim[:-1])
                pnp1 = (p, self.stim[1:])
            else:
                pn = (p[:-1], self.stim[:-1])
                pnp1 = (p[1:], self.stim[1:])

        fn = self.f(self.t_model[:-1], x[:-1], pn)
        fnp1 = self.f(self.t_model[1:], x[1:], pnp1)

        return self.dt_model * (fn + fnp1) / 2.0

    #Don't use RK4 yet, still trying to decide how to implement with a stimulus.
    #def disc_rk4(self, x, p):
    #    """
    #    RK4 time discretization for the action.
    #    """
    #    if self.stim is None:
    #        pn = p
    #        pmid = p
    #        pnp1 = p
    #    else:
    #        pn = (p, self.stim[:-2:2])
    #        pmid = (p, self.stim[1:-1:2])
    #        pnp1 = (p, self.stim[2::2])
    #
    #    xn = x[:-1]
    #    tn = np.tile(self.t[:-1], (self.D, 1)).T
    #    k1 = self.f(tn, xn, pn)
    #    k2 = self.f(tn + self.dt/2.0, xn + k1*self.dt/2.0, pmid)
    #    k3 = self.f(tn + self.dt/2.0, xn + k2*self.dt/2.0, pmid)
    #    k4 = self.f(tn + self.dt, xn + k3*self.dt, pnp1)
    #    return self.dt * (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

    def disc_SimpsonHermite(self, x, p):
        """
        Simpson-Hermite time discretization for the action.
        This discretization applies Simpson's rule to all the even-index time
        points, and a Hermite polynomial interpolation for the odd-index points
        in between.
        """
        if self.stim is None:
            if self.P.ndim == 1:
                pn = p
                pmid = p
                pnp1 = p
            else:
                pn = p[:-2:2]
                pmid = p[1:-1:2]
                pnp1 = p[2::2]
        else:
            if self.P.ndim == 1:
                pn = (p, self.stim[:-2:2])
                pmid = (p, self.stim[1:-1:2])
                pnp1 = (p, self.stim[2::2])
            else:
                pn = (p[:-2:2], self.stim[:-2:2])
                pmid = (p[1:-1:2], self.stim[1:-1:2])
                pnp1 = (p[2::2], self.stim[2::2])

        fn = self.f(self.t_model[:-2:2], x[:-2:2], pn)
        fmid = self.f(self.t_model[1:-1:2], x[1:-1:2], pmid)
        fnp1 = self.f(self.t_model[2::2], x[2::2], pnp1)

        disc_vec1 = (fn + 4.0*fmid + fnp1) * (2.0*self.dt_model)/6.0
        disc_vec2 = (x[:-2:2] + x[2::2])/2.0 + (fn - fnp1) * (2.0*self.dt_model)/8.0

        return disc_vec1, disc_vec2

    def disc_forwardmap(self, x, p):
        """
        "Discretization" when f is a forward mapping, not an ODE.
        """
        if self.stim is None:
            if self.P.ndim == 1:
                pn = p
            else:
                pn = p[:-1]
        else:
            if self.P.ndim == 1:
                pn = (p, self.stim[:-1])
            else:
                 pn = (p[:-1], self.stim[:-1])

        return self.f(self.t_model[:-1], x[:-1], pn)
