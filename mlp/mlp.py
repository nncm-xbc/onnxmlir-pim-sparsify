# mlp.py — JAX MLP primitives: parameter init, forward pass, training utilities

from typing import NamedTuple
import jax.numpy as jnp
from jax import grad, jit, lax, vmap
from jax import random
from jax.scipy.special import logsumexp
import numpy as np
import os

class Layer(NamedTuple):
    W:    np.ndarray  # weight matrix, shape (n_out, n_in)
    b:    np.ndarray  # bias vector,   shape (n_out,)
    mask: np.ndarray  # sparsity mask, shape (n_out, n_in), 1=active 0=pruned

# Randomly initialize weights and biases for a single dense layer.
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected network with the given layer sizes.
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    layers = []
    for (m, n), k in zip(zip(sizes[:-1], sizes[1:]), keys):
        w, b = random_layer_params(m, n, k)
        layers.append(Layer(W=w, b=b, mask=np.ones((n, m))))
    return layers

def load_network_params(folder_name):
    count = sum(1 for f in os.listdir(folder_name) if f.startswith('W_') and f.endswith('.npy'))
    print("Found %d layers in %s" % (count, folder_name))
    layers = []
    for i in range(count):
        W = np.load(folder_name + "/W_%d.npy" % i)
        b = np.load(folder_name + "/b_%d.npy" % i)
        layers.append(Layer(W=W, b=b, mask=np.ones_like(W)))
    return layers

def relu(x):
    return jnp.maximum(0, x)

# Activation registry — extend here to support new activations.
_ACTS = {
    'relu':    relu,
    'tanh':    jnp.tanh,
    'sigmoid': jax.nn.sigmoid,
    'linear':  lambda x: x,
}

# Hidden-layer activation used by predict().
# Set this (e.g. mlp.hidden_activation = mlp._ACTS['tanh']) before
# the first JAX trace — i.e., before any training or sparsification call.
hidden_activation = relu

def predict(network, image):
    activations = image
    for layer in network[:-1]:
        outputs = jnp.dot(lax.stop_gradient(layer.mask) * layer.W, activations) + layer.b
        activations = hidden_activation(outputs)
    final = network[-1]
    logits = jnp.dot(lax.stop_gradient(final.mask) * final.W, activations) + final.b
    return logits - logsumexp(logits)

batched_predict = vmap(predict, in_axes=(None, 0))

step_size = 0.01

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(network, images, targets):
    target_class    = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(network, images), axis=1)
    return jnp.mean(predicted_class == target_class)

def loss(network, images, targets):
    preds = batched_predict(network, images)
    return -jnp.mean(preds * targets)

@jit
def update(network, x, y):
    grads = grad(loss)(network, x, y)
    return [Layer(W=l.W - step_size * g.W, b=l.b - step_size * g.b, mask=l.mask)
            for l, g in zip(network, grads)]
