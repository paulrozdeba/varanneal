"""
Generate random parameters for the data-generating network.

The weights are drawn from a normal distribution ~N(0,1).
The biases are all set to zero.
"""

import numpy as np

# Set up network structure
D_in = 10
D_out = 10
D_hidden = 10
N = 100  # number of layers, including input and output

Nparamsets = 10  # number of different parameter sets to generate

np.random.seed(17439860)  # change the seed to get different parameters

for i in xrange(Nparamsets):
    # Generate parameter set i
    W = []  # weights
    b = []  # biases

    # First layer
    Wn = (2.0*np.random.rand(D_hidden, D_in) - 1.0) / float(D_in)
    bn = np.zeros(D_hidden)
    W.append(Wn)
    b.append(bn)

    for n in xrange(1, N - 1):
        # Loop through layers and generate weights from n to n+1
        Wn = (2.0*np.random.rand(D_hidden, D_hidden) - 1.0) / float(D_hidden)
        bn = np.zeros(D_hidden)
        W.append(Wn)
        b.append(bn)

    # Last layer
    Wn = (2.0*np.random.rand(D_out, D_hidden) - 1.0) / float(D_hidden)
    bn = np.zeros(D_out)
    W.append(Wn)
    b.append(bn)

    W = np.array(W)
    b = np.array(b)

    # Save this parameter set to file
    np.save("params/W_%d.npy"%(i+1,), W)
    np.save("params/b_%d.npy"%(i+1,), b)
