def A_gaussian(XP):
    """
    Gaussian action.
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

    # Traded the statements below in favor of calling self.stim directly in
    # the discretization functions.
    #if self.stim is not None:
    #    p = (p, self.stim)

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
