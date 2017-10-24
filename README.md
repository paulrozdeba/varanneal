# VarAnneal
VarAnneal is a Python module that uses the variational annealing (VA) algorithm first proposed by J.C. Quinn<sup>[1]</sup>
and described in detail in Ye et. al.<sup>[2]</sup> to 
perform state and parameter estimation in partially observed dynamical systems.  This method requires optimization 
of a cost function called the "action" that balances measurement error (deviations of state estimates from observations) 
and model error (deviations of trajectories from the assumed dynamical behavior given a state and parameter estimate). 
Derivatives of the cost function are computed using automatic differentiation (AD) as implemented in 
[PYADOLC](https://github.com/b45ch1/pyadolc), a Python wrapper around [ADOL-C](https://projects.coin-or.org/ADOL-C).  

### Get VarAnneal
VarAnneal is under constant development.  When you clone this repository, the master branch is the most up-to-date with the most bug fixes, new features, and (hopefully) the fewest bugs; while tags represent previous versions of the code.  You can clone VarAnneal to your computer with the command
```bash
$ git clone https://github.com/paulrozdeba/varanneal
```
which, when you open the directory, will automatically have the master branch checked out.  You can check out other tags by entering this directory and executing
```bash
$ git pull
$ git fetch --tags
$ git checkout tags/<tagname>
```
where the various tags can be found with `git tags --list`, or in "Branch" dropdown menu near the top of this page.

### Install
VarAnneal requires you have the following software installed on your computer:
1. Python 2 (tested on 2.7.9, probably will work on anything ≥ 2.7).
2. NumPy (tested on ≥ 1.12.0).
3. SciPy (tested on ≥ 0.18.1).
4. [PYADOLC](https://github.com/b45ch1/pyadolc)  

You should follow the installation instructions for PYADOLC in the readme page linked to above.
If you're running any Linux 
distribution you should be able to follow the Ubuntu installation instructions with some minor 
modifications (mostly finding the right packages in your distribution's repository, which may have 
slightly different names from the Ubuntu repo).

**Caveat:** Building PYADOLC currently fails with boost 1.65.1.  Currently (as of 10/23/17), the newest version 
of boost available for Ubuntu 17.10 (Artful Aardvark) is 1.62.0.1, so Ubuntu users should not have this 
problem.  If you are running some other distribution with more up-to-date libraries, like Fedora or 
Arch Linux, make sure to hold back boost to 1.64.0 at the latest while this issue is resolved.  
If you are installing in Mac OS with homebrew, you should similarly hold back boost to 
an older version.  Use these commands to install boost 1.59, and to get your system to properly link to them:
```bash
$ brew install boost@1.59
$ brew install boost-python@1.59
$ brew link boost@1.59 --force
$ brew link boost-python@1.59 --force
```

Once you have this all installed, clone this git repo somewhere on your computer; I usually like putting 
cloned repos in `~/.local/src`.  There is a setup.py file in the repo so, if you use Anaconda for example, 
then follow the usual procedure for installation.  If you don't, you can easily install the package with 
setuptools (comes automatically with [pip](https://pip.pypa.io/en/stable/installing/)). Install it locally 
with the following steps:
```bash
python2 setup.py build
python2 setup.py install --user
```
and now you should be able to import varanneal from anywhere (repeat this procedure for additional users 
on the same machine).  Note: I plan to get VarAnneal up on PyPI eventually...

### Usage
(This example loosely follows the code found in the examples folder in this repository, for the case of 
state and parameter estimation in an ODE dynamical system. Check out the neural network examples too to 
see how that works; eventually I'll update this README with instructions for neural networks too).
Start by importing VarAnneal, as well as NumPy which we'll need too, then instantiate an Annealer object:
```python
import numpy as np
from varanneal import va_ode

myannealer = va_ode.Annealer()
```
Alternatively the following syntax for importing/using varanneal (ODE version) works too:
```python
import varanneal
myannealer = varanneal.va_ode.Annealer()
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

### Contributors
Thanks to Nirag Kadakia and Uriel I. Morone for their contributions to this project.

### License
This project is licensed under the MIT license.  See LICENSE for more details.
