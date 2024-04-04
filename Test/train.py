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
    data_train       = np.genfromtxt(__location__ + '/mnist_train.csv', delimiter = ',', max_rows = 10000)
    x_train, y_train = data_train[:,1:], data_train[:,0]
    
    data_test        = np.genfromtxt(__location__ + '/mnist_test.csv', delimiter = ',', max_rows = 1000)
    x_test, y_test   = data_test[:,1:], data_test[:,0]
    
    x_train          = x_train.reshape(-1,28,28)
    x_test           = x_test.reshape(-1,28,28)
    
    return (x_train,y_train),(x_test,y_test)


def main():
    # define the output folder through the command line
    args          = sys.argv
    output_folder = __location__ + "/" + args[1]	
    print("Selected folder %s" % output_folder)   
    
    # data loading
    onehot = lambda y : np.concatenate([
        (y == cifra)[:,None] * 1.
    for cifra in np.arange(10)
    ], axis = 1)

    # train-test split
    (x_train, y_train), (x_test, y_test) = load_data()

    # image compression (just for the sake of speed)
    x_train = np.array([ np.array(Image.fromarray(x).resize((14,14))) for x in x_train])
    x_test  = np.array([ np.array(Image.fromarray(x).resize((14,14))) for x in x_test])

    # flattening (MLP)
    x_train = x_train.reshape(-1,14*14)
    x_test  = x_test.reshape(-1,14*14)
    y_test  = onehot(y_test)
    y_train = onehot(y_train)
    
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
    
    layer_sizes = [14*14, 10, 10, 10]
    
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