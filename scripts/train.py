# train.py — trains a simple MLP classifier on MNIST

import sys
import os
import csv
import time
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import jax.numpy as jnp                # Parallel computing / Autograd
from jax import grad, jit, vmap
from jax import random

from mlp.mlp import *

import numpy as np

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
_data_dir    = os.path.realpath(os.path.join(__location__, '..', 'data'))

def load_data():
    x_train       = np.genfromtxt(_data_dir + '/X_train_small.csv', delimiter = ',', max_rows = 10000)
    y_train       = np.genfromtxt(_data_dir + '/Y_train_small.csv', delimiter = ',', max_rows = 10000)

    x_test        = np.genfromtxt(_data_dir + '/X_test_small.csv', delimiter = ',', max_rows = 1000)
    y_test        = np.genfromtxt(_data_dir + '/Y_test_small.csv', delimiter = ',', max_rows = 1000)

    return (x_train,y_train),(x_test,y_test)

def main():
    # define the output folder through the command line
    args          = sys.argv
    output_folder = os.path.abspath(args[1])
    topology_csv  = os.path.abspath(args[2])

    print("Selected folder %s" % output_folder)

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


    layer_sizes  = np.genfromtxt(topology_csv, delimiter = ',');
    layer_sizes[0] = layer_sizes[0]**2
    layer_sizes    = [ int(l) for l in layer_sizes]
    print("Network Topology loaded")
    print("\t %s " % layer_sizes)


    print("\n")
    print("Training loop started")
    batch_epochs = 100
    num_epochs = batch_epochs * 10
    batch_size = 128
    n_targets = 10

    print("---------------------------------------------------------")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    log_path = os.path.join(output_folder, "training_log.csv")
    log_file = open(log_path, 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(['epoch', 'train_acc', 'test_acc', 'train_loss', 'epoch_time_s'])

    # Training loop
    network = init_network_params(layer_sizes, random.PRNGKey(0))
    for epoch in range(num_epochs):
        start_time = time.time()
        for i in range(10):
            batch = np.random.choice(len(x_train), size = 500)
            x,y = x_train[batch],y_train[batch]
            network = update(network, x, y)
        epoch_time = time.time() - start_time

        train_acc  = float(accuracy(network, x_train, y_train))
        test_acc   = float(accuracy(network, x_test, y_test))
        train_loss = float(loss(network, x_train, y_train))
        writer.writerow([epoch, round(train_acc, 6), round(test_acc, 6),
                         round(train_loss, 6), round(epoch_time, 3)])
        log_file.flush()
        if epoch % batch_epochs == 0:
            print("\t \t Epoch {} of {} in {:0.2f} sec".format(epoch,num_epochs,epoch_time))
            print("\t \t Training set accuracy {:0.5f}".format(train_acc))
            print("\t \t Test set accuracy {:0.5f}".format(test_acc))
            print("---------------------------------------------------------")

    log_file.close()
    print("Training log saved to:", log_path)

    # Saving the data
    for i, layer in enumerate(network):
        np.save(output_folder + "/W_%i.npy" % i, layer.W)
        np.save(output_folder + "/b_%i.npy" % i, layer.b)

    print("\n")
    print("Model saved")

if __name__ == "__main__":
    main()
