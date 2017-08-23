"""
Generate lots of input/output pairs from the data-generating network.
"""

import numpy as np

# Define the network activations
def activation(x, W, b):
    linpart = np.dot(W, x) + b
    return 1.0 / (1.0 + np.exp(-linpart))

# Set up network structure
D_in = 10
D_out = 10
D_hidden = 10
N = 100  # number of layers, including input and output

# Noise level in input/output data
sigma = 0.005  # standard deviation of Gaussian noise
suffix = "sm0p005"  # optional suffix to append to filenames

# Generate lots of random input/output pairs
Nparam = 2  # number of different network parametrizations to generate data with
Nexamples = 10000  # number of examples to generate
# Uncomment corresponding two lines to generate training, validation, or
# test data.
folder = "training"
np.random.seed(43650832)  # training
#folder = "validation"
#np.random.seed(69856438)  # validation
#folder = "test"
#np.random.seed(78943689)  # test

for i in xrange(Nparam):  # parameter index
    # Load in parameters
    W = np.load("params/W_%d.npy"%(i+1,))
    b = np.load("params/b_%d.npy"%(i+1,))
    for j in xrange(Nexamples):
        y = []
        # Draw random input from N(0, sigma^2)
        yin = np.random.randn(D_in)
        # Normalize to have mean 0 and sigma=1
        yin = (yin - np.average(yin)) / np.std(yin)
        y.append(yin)
        for n in xrange(N - 1):
            y.append(activation(y[n], W[n], b[n]))
        np.save("%s/param%d/truestates_%d.npy"%(folder, i+1, j+1), y)

        if sigma > 0:
            # Add noise to input and output to simulate observation noise
            noisy_input = y[0] + sigma*np.random.randn(D_in)
            # Output is clipped to valid sigmoid range
            noisy_output = np.clip(y[-1] + sigma*np.random.randn(D_out), 0.0001, 0.9999)
            noisy_io = np.array([noisy_input, noisy_output])
            np.save("%s/param%d/noisyio_%s_%d.npy"%(folder, i+1, suffix, j+1), noisy_io)
