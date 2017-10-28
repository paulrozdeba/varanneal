# True parameter values
# Listed in the same order as defined in the NaKL ode

import numpy as np

ptrue = [120.0, 20.0, 0.3, 50.0, -77.0, -54.0,
         -40.0, 1.0/0.06667, 0.1, 0.4,
         -60.0, -1.0/0.06667, 1.0, 7.0,
         -55.0, 1.0/0.03333, 1.0, 5.0]

np.save("NaKL_trueparam_dt0p02_N6001.npy", np.array(ptrue))
