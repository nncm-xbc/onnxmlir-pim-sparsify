# train.py — trains a simple MLP classifier on MNIST

import sys
import os
import csv
import json
import time
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from mlp.mlp import *

import numpy as np


def main():
    cfg_path = os.path.abspath(sys.argv[1])
    with open(cfg_path) as f:
        cfg = json.load(f)

    # paths in config are relative to the repo root (parent of experiments/)
    repo_root     = os.path.dirname(os.path.dirname(cfg_path))
    output_folder = os.path.join(repo_root, 'artifacts', cfg['name'])

    def resolve(p):
        return os.path.join(repo_root, p)

    print("Output folder: %s" % output_folder)

    x_train = np.genfromtxt(resolve(cfg['data']['x_train']), delimiter=',', max_rows=10000)
    y_train = np.genfromtxt(resolve(cfg['data']['y_train']), delimiter=',', max_rows=10000)
    x_test  = np.genfromtxt(resolve(cfg['data']['x_test']),  delimiter=',', max_rows=1000)
    y_test  = np.genfromtxt(resolve(cfg['data']['y_test']),  delimiter=',', max_rows=1000)

    print("Data loaded.")
    print("\tX_train.shape = %s" % str(x_train.shape))
    print("\tX_test.shape  = %s" % str(x_test.shape))

    import mlp.mlp as _mlp
    _mlp.hidden_activation = _mlp._ACTS[cfg.get('hidden_activation', 'relu')]

    layer_sizes = cfg['topology']
    epochs      = cfg['train']['epochs']
    batch_size  = cfg['train']['batch_size']

    print("Topology: %s" % layer_sizes)
    print("Training loop started")
    print("---------------------------------------------------------")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    log_path = os.path.join(output_folder, "training_log.csv")
    log_file = open(log_path, 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(['epoch', 'train_acc', 'test_acc', 'train_loss', 'epoch_time_s'])

    network = init_network_params(layer_sizes, random.PRNGKey(0))
    for epoch in range(epochs):
        start_time = time.time()
        for _ in range(10):
            batch   = np.random.choice(len(x_train), size=batch_size)
            x, y    = x_train[batch], y_train[batch]
            network = update(network, x, y)
        epoch_time = time.time() - start_time

        train_acc  = float(accuracy(network, x_train, y_train))
        test_acc   = float(accuracy(network, x_test, y_test))
        train_loss = float(loss(network, x_train, y_train))
        writer.writerow([epoch, round(train_acc, 6), round(test_acc, 6),
                         round(train_loss, 6), round(epoch_time, 3)])
        log_file.flush()
        if epoch % 100 == 0:
            print("\tEpoch {} of {} in {:0.2f} sec".format(epoch, epochs, epoch_time))
            print("\tTrain acc {:0.5f} | Test acc {:0.5f}".format(train_acc, test_acc))
            print("---------------------------------------------------------")

    log_file.close()
    print("Training log saved to:", log_path)

    for i, layer in enumerate(network):
        np.save(os.path.join(output_folder, "W_%i.npy" % i), layer.W)
        np.save(os.path.join(output_folder, "b_%i.npy" % i), layer.b)

    print("Model saved to:", output_folder)


if __name__ == "__main__":
    main()
