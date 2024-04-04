# @ Train.py
# Trains a simple MNIST classifier

#######################################################################################################################
### Librerie

import sys                             # for the CLI execution
import os

import jax.numpy as jnp                # Parallel computing / Autograd
from jax import grad, jit, vmap
from jax import random

from MLP.mlp import *                  # Mini-Package for Multi Layer Perceptron


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

#######################################################################################################################
### Data Loading

def load_data():
    x_train       = np.genfromtxt(__location__ + '/X_train_small.csv', delimiter = ',', max_rows = 10000)
    y_train       = np.genfromtxt(__location__ + '/Y_train_small.csv', delimiter = ',', max_rows = 10000)
    
    x_test        = np.genfromtxt(__location__ + '/X_test_small.csv', delimiter = ',', max_rows = 1000)
    y_test        = np.genfromtxt(__location__ + '/Y_test_small.csv', delimiter = ',', max_rows = 1000)
    
    
    return (x_train,y_train),(x_test,y_test)


def main():
    # define the output folder through the command line
    args          = sys.argv
    output_folder = __location__ + "/" + args[1]	
    topology_csv  = __location__ + "/" + args[2]
    
    print("Selected folder %s" % output_folder)   
    
    # data loading
    onehot = lambda y : np.concatenate([
        (y == cifra)[:,None] * 1.
    for cifra in np.arange(10)
    ], axis = 1)

    # train-test split
    (x_train, y_train), (x_test, y_test) = load_data()
    
    print("Data loaded.")
    print("\tX_train.shape = %s" % str(x_train.shape))
    print("\tX_test.shape  = %s" % str(x_test.shape))
    print("\tY_train.shape = %s" % str(y_train.shape))
    print("\tY_test.shape  = %s" % str(y_test.shape))
    

    # Define the network topology // an extremely simple topology, just for testing
    
    #    O\
    #    O>O\/O\/O
    #    O>O><O><O
    #    O>O/\O/\O
    #    O/
    
    #layer_sizes = [14*14, 10, 10, 10]
    # leggo la topologia dal file
    layer_sizes  = np.genfromtxt(topology_csv, delimiter = ','); 
    layer_sizes[0] = layer_sizes[0]**2
    layer_sizes    = [ int(l) for l in layer_sizes]
    print("Network Topology loaded")
    print("\t %s " % layer_sizes)
    
    
    mask, params = init_network_params(layer_sizes, random.PRNGKey(0))
    print("\n")    
    print("Training loop started")
    batch_epochs = 100
    num_epochs = batch_epochs * 10
    batch_size = 128
    n_targets = 10
    
    print("---------------------------------------------------------")

    # Training loop
    import time
    mask, params = init_network_params(layer_sizes, random.PRNGKey(0))
    for epoch in range(num_epochs):
        start_time = time.time()
        for i in range(10):
            batch = np.random.choice(len(x_train), size = 500)
            x,y = x_train[batch],y_train[batch]
            params = update(params, mask, x, y)
            
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, mask, x_train, y_train)
        test_acc = accuracy(params, mask,  x_test, y_test)
        if epoch % batch_epochs == 0:
            print("\t \t Epoch {} of {} in {:0.2f} sec".format(epoch,num_epochs,epoch_time))
            print("\t \t Training set accuracy {:0.5f}".format(train_acc))
            print("\t \t Test set accuracy {:0.5f}".format(test_acc))
            print("---------------------------------------------------------")
    # Controllo se la cartella esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Saving the data
    for i,p in enumerate(params):
        np.save(output_folder + "/W_%i.npy" % i, p[0])
        np.save(output_folder + "/b_%i.npy" % i, p[1])
    
    print("\n")
    print("Model saved")

if __name__ == "__main__":
    main()