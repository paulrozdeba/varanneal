In this example, variational annealing (VA) is used to train a "rectangular" 
neural network on input/output examples from *another* rectangular network. 
Here, rectangular is taken to mean that the input, output, and hidden layers 
all have the same number of neurons.

The data-generating network has 100 layers, and 10 neurons per layer (by default). 
To run this example, you should first generate this data. Go in to the "data" 
folder here, and run
    python2 gen_params.py
    python2 gen_io_pairs.py
To change the number of parametrizations, examples, etc., as well as how the 
parameters and examples are generated, you can edit these two files to your 
liking.

Once you've generated the training data, go back up one directory and run 
   python2 nnet_twin_anneal.py
This will carry out VA for network state and parameter estimation. To change 
various parameters for the *estimated* network, edit nnet_twin_anneal.py to 
alter the estimated network structure, number of training examples, etc.
