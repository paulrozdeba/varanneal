# VarAnneal

### Overview
VarAnneal is a Python module that uses the variational annealing (VA) algorithm introduced by Ye et. al.<sup>[1]</sup> to 
perform state and parameter estimation in partially observed dynamical systems.  This method requires optimization 
of a cost function that balances measurement error (deviations of state estimates from observations) and model error 
(deviations of trajectories from the assumed dynamical behavior given a state and parameter estimate).  Derivatives 
of the cost function are computed using automatic differentiation (AD) using pyadolc<sup>[2]</sup>, a Python wrapper around 
ADOL-C.

### References
[1] J. Ye, N. Kadakia, P. Rozdeba, H.D.I. Abarbanel, and J.C. Quinn.  *Improved  variational methods in 
statistical data assimilation.*  Nonlin. Proc. Geophys. **22**, 205-213 (2015).

[2] S.F. Walter.  *PYADOLC, a wrapper for ADOL-C.*  [https://github.com/b45ch1/pyadolc](https://github.com/b45ch1/pyadolc)
