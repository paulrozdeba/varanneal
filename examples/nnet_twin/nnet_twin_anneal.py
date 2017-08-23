"""
"Twin" neural network training example.
"""

import numpy as np
import varanneal_neuralnet
import sys, time

L = 10  # number of observed components
M = 5  # number of training examples
suffix = "sm0p005"  # suffix to specify 
# Next is the ADOLC tape ID. This only needs to be changed if you're
# running multiple instances simultaneously (to avoid using the tape from 
# another instance with the same ID, although I'm not sure even this will 
# happen if you re-use an ID).
adolcID = 0

# Define the transfer function
def sigmoid(x, W, b):
    linpart = np.dot(W, x) + b
    return 1.0 / (1.0 + np.exp(-linpart))

# Network structure
N = 20  # Total number of layers
D_in = 10  # Number of neurons in the input layer
D_out = 10  # Number of neurons in the output layer
D_hidden = 10  # Number of neurons in the hidden layers

# Don't need to touch the next few lines, unless you want to manually alter 
# the network structure.
structure = np.zeros(N, dtype='int')
structure[0] = D_in  # 3 neurons in the input layer
structure[N-1] = D_out  # 2 neurons in the output layer
for i in range(1, N-1):
    structure[i] = D_hidden  # 5 neurons in the hidden layers

# Indices of measured components. Next line also doesn't need to be touched 
# unless you want to manually alter them.
Lidx = [np.linspace(0, L-1, L, dtype='int'), np.linspace(0, L-1, L, dtype='int')]

################################################################################
# Action/annealing parameters
################################################################################
# RM, RF0
RM = 1.0 / (.005**2)
RF0 = 1.0e-8 * RM * float(np.sum(structure) - structure[0]) / float(structure[0] + structure[-1])  # Initial RF value
# alpha, and beta "ladder"
alpha = 1.1
beta_array = np.linspace(0, 435, 436)

################################################################################
# Input and output data
################################################################################
data_dir = "../../data/training/param1/"
data_in = []
data_out = []

# Load all examples into an array
for n in range(M):
    data = np.load(data_dir + "noisyio_sm0p005_%d.npy"%(n+1,))
    data_in.append(data[0][Lidx[0]])
    data_out.append(data[-1][Lidx[1]])

data_in = np.array(data_in)
data_out = np.array(data_out)

################################################################################
# Initial path/parameter guesses
################################################################################
np.random.seed(89072545)
# Neuron states
# First example
Xin = np.random.randn(D_in)
Xin = (Xin - np.average(Xin)) / np.std(Xin)
X0 = [Xin]
for n in xrange(N-1):
    X0.append(0.2*np.random.rand(D_hidden) + 0.4)

# Now the rest of the M-1 examples
for m in xrange(M - 1):
    Xin = np.random.randn(D_in)
    Xin = (Xin - np.average(Xin)) / np.std(Xin)
    X0.append(Xin)
    for n in xrange(N-1):
        X0.append(0.2*np.random.rand(D_hidden) + 0.4)

X0 = np.array(X0).flatten()

# Parameters
NP = np.sum(structure[1:]*structure[:-1] + structure[1:])  # total number of parameters
Pidx = []  # array for storing estimated parameter indices
P0 = np.array([], dtype=np.float64)  # array for initial parameter

# Next four lines are index initializations, don't need to touch
W_i0 = 0
W_if = structure[0]*structure[1]
b_i0 = W_if
b_if = b_i0 + structure[1]

for n in xrange(N - 1):
    # Only weights are estimated. Thus, Pidx will contain only weight indices,
    # and biases will be set to zero.
    Pidx.append(range(W_i0, W_if)[:])
    if n == 0:
        P0 = np.append(P0, (2.0*np.random.rand(structure[n]*structure[n+1]) - 1.0) / D_in)
    else:
        P0 = np.append(P0, (2.0*np.random.rand(structure[n]*structure[n+1]) - 1.0) / D_hidden)
    P0 = np.append(P0, np.zeros(structure[n+1]))

    # Update indices, also don't need to touch.
    if n < N - 2:
        W_i0 = b_if
        W_if = W_i0 + structure[n+1]*structure[n+2]
        b_i0 = W_if
        b_if = b_i0 + structure[n+2]

P0 = np.array(P0).flatten()
Pidx = np.array(Pidx).flatten().tolist()

################################################################################
# Annealing
################################################################################
# Initialize Annealer
anneal1 = varanneal_neuralnet.Annealer()
# Set the network structure
anneal1.set_structure(structure)
# Set the activation function
anneal1.set_activation(sigmoid)
# Set the input and output data
anneal1.set_input_data(data_in)
anneal1.set_output_data(data_out)

# Run the annealing using L-BFGS-B
BFGS_options = {'gtol':1.0e-12, 'ftol':1.0e-12, 'maxfun':1000000, 'maxiter':1000000}
tstart = time.time()
anneal1.anneal(X0, P0, alpha, beta_array, RM, RF0, Pidx, Lidx=Lidx,
               method='L-BFGS-B', opt_args=BFGS_options, adolcID=adolcID)
print("\nADOL-C annealing completed in %f s."%(time.time() - tstart))

# Save the *entire* estimated state, warning: big array!!
#anneal1.save_states("states.npy")
# Save estimates of input/output pairs only.
anneal1.save_io("io.npy")
# Save estimated weights and biases.
anneal1.save_Wb("W.npy", "b.npy")
# Save action and constituent error terms.
anneal1.save_action_errors("aerr.npy")
