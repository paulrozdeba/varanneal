# VarAnneal
VarAnneal is a Python module that uses the variational annealing (VA) algorithm first proposed by J.C. Quinn<sup>[1]</sup>
and described in detail in Ye et. al.<sup>[2]</sup> to 
perform state and parameter estimation in partially observed dynamical systems.  This method requires optimization 
of a cost function called the "action" that balances measurement error (deviations of state estimates from observations) 
and model error (deviations of trajectories from the assumed dynamical behavior given a state and parameter estimate). 
Derivatives of the cost function are computed using automatic differentiation (AD) as implemented in 
[PYADOLC](https://github.com/paulrozdeba/pyadolc), a Python wrapper around [ADOL-C](https://projects.coin-or.org/ADOL-C).  (Note that for now I'm linking to my fork of PYADOLC, which updates the version of ColPack but requires one or two extra steps to install).

### Install
VarAnneal requires you have the following software installed on your computer:
1. Python 2 (tested on 2.7.9, probably will work on anything >= 2.7).
2. NumPy (tested on 1.12.0).
3. SciPy (tested on 0.18.1).
4. [PYADOLC](https://github.com/paulrozdeba/pyadolc)  
    -- The installation instructions for PYADOLC are good, and if you are on a UNIX system then installation is 
       fairly painless (I haven't tried installing this in Windows).  Installing PYADOLC means installing 
       ADOL-C, which also installs some readily available and commonly used libraries like boost and ColPack.  Again, 
       this is fairly automated, but may require a little tinkering.
    -- Installing ColPack, which is part of insalling PYADOLC, requires some packages in addition to what is listed 
       on the PYADOLC page: automake and autoconf.  You should be able to find these in your Linux distribution's 
       software repository, or on homebrew if you use Mac OS.

Finally, your own code needs to be able to find VarAnneal.  There is no formal install strategy as of yet, 
but simply placing varanneal.py in the same folder as your code, or preferably creating a link/alias to 
varanneal.py in the same folder, will suffice.  For example,
```bash
ln -s <varanneal_dir>/varanneal.py ./
```

### Usage
(This example loosely follows the code found in the examples folder in this repository.)  
Start by importing VarAnneal, as well as NumPy which we'll need too, then instantiate an Annealer object:
```python
import numpy as np
import varanneal

myannealer = varanneal.Annealer()
```
Now define a dynamical model for the observed system (here we're going to use Lorenz 96 as an example):
```python
def l96(t, x, k):
    return np.roll(x, 1, 1) * (np.roll(x, -1, 1) - np.roll(x, 2, 1)) - x + k
D = 20  # dimensionality of the Lorenz model

myannealer.set_model(l96, D)
```
Import the observed data.  This file should be plain-text file in the following format:

`t, y_1, y_2, ..., y_L`

or a Numpy .npy archive with shape (*N*, *L+1*) where the 0th element of each time step is the time, and the rest are 
the observed values of the L observed variables.  Use the built-in convenience function to do this:
```python
myannealer.set_data_fromfile("datafile.npy")
N_data = myannealer.N_data  # Number of data time points, we're going to use this in a bit
```
Your other option is to just pass myannealer a NumPy array containing the data directly, using myannealer.set_data.  
This is up to you and your coding preferences.  An example of how to use this other function is in the Lorenz96 
example in the examples folder.

Finally, we need to set a few other important quantities like the model indices of the observed variables; the 
indices of the estimated parameters (all other parameters remain fixed); the annealing hyperparameters 
(measurement and model error coefficients RM and RF, respectively); the "exponential ladder" for annealing RF; 
the desired optimization routine (and options for the routine); and last but not least the initial state and 
parameter guesses:
```python
Lidx = [0, 2, 4, 8, 10, 14, 16]  # measured variable indices
RM = 1.0 / (0.5**2)
RF0 = 4.0e-6  # initial RF value for annealing

# Now define the "exponential ladder" for anealing
alpha = 1.5
beta = np.linspace(0, 100, 101)

# Initial state and parameter guesses
# We're going to use the init_to_data option later to automatically set the observed variables to 
# their observed values in the data.
N_model = N_data  # Want to evaluate the model at the observation times
X0 = (20.0 * np.random.rand(N_model * D) - 10.0).reshape((N_model, D))

Pidx = [0]  # indices of estimated parameters
# The initial parameter guess can either be a list of values, or an array with N entries of guesses 
# which VarAnneal interprets as the parameters being time-dependent.  Here we're sticking with a 
# static parameter:
P0 = np.array([4.0 * np.random.rand() + 6.0])

# Options for L-BFGS-B in scipy.  These set the tolerance levels for termination in f and its 
# gradient, as well as the maximum number of iterations before moving on.  See the manpage for 
# L-BFGS-B in scipy.optimize.minimize at 
# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb 
# for more information.
BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 'maxiter':1000000}

myannealer.anneal(X0, P0, alpha, beta, RM, RF0, Lidx, Pidx, dt_model=dt_model, 
                  init_to_data=True, disc='SimpsonHermite', method='L-BFGS-B', 
                  opt_args=BFGS_options, adolcID=0)
```
That's it!  Let the annealer run, and at the end save the results.  VarAnneal saves to NumPy .npy archives, which 
are far more efficient for storing multi-dimensional arrays (which is the case here), and use compression so the 
resulting files are far smaller than plain-text with much greater precision.  They are all saved over the whole annealing 
run, meaning that each array is structured like (N_beta, ...) where N_beta is the number of beta values visited 
during the annealing.  The ... represents the appropriate dimensions for whatever data is being saved.
```python
myannealer.save_paths("paths.npy")  # Path estimates
myannealer.save_params("params.npy")  # Parameter estimates
myannealer.save_action_errors("action_errors.npy")  # Action and individual error terms
```

### References
[1] J.C. Quinn, *A path integral approach to data assimilation in stochastic nonlinear systems.* Ph.D. 
thesis in physics, UC San Diego, https://escholarship.org/uc/item/obm253qk (2010).

[2] J. Ye, N. Kadakia, P. Rozdeba, H.D.I. Abarbanel, and J.C. Quinn.  *Improved  variational methods in 
statistical data assimilation.*  Nonlin. Proc. Geophys. **22**, 205-213 (2015).

### Author
Paul Rozdeba  
May 23, 2017

### License
This project is licensed under the MIT license.  See LICENSE for more details.
