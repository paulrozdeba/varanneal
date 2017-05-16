import numpy as np
import adolc
import time
import scipy.optimize as opt

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

if N is None:
    self.t = t[nstart:]
    self.Y = self.Y[nstart:]
    self.N = self.Y.shape[0]
    if self.stim is not None:
        self.stim = self.stim[nstart:]
else:
    self.t = t[nstart:(nstart + N)]
    self.Y = self.Y[nstart:(nstart + N)]
    self.N = N
    if self.stim is not None:
        self.stim = self.stim[nstart:(nstart + N)]

# other stuff
self.A = self.A_gaussian
self.taped = False
self.initalized = False

# arguments for optimization routine
self.opt_args = None
