# @ MLP

##########################################################################################################################################################

# libraries

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import numpy as np
import os

##########################################################################################################################################################

# A helper function to randomly initialize weights and biases
# for a dense neural network layer

def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_mask(m,n):
    return np.ones((n, m))

def get_full_mask(params):
    return [ np.array(p[0])*0. + 1 for p in params]

##########################################################################################################################################################

# Initialize all layers for a fully-connected neural network with sizes "sizes"

def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return ( [init_mask(m,n) for m,n in zip(sizes[:-1],sizes[1:])],  [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)])
  
def load_network_params(folder_name):
  # count how many files in the folder
  count = 0
  for root, _, files in os.walk(folder_name):
    for _ in files:
      count += 1
  ret = []  
  count = int(count/2)
  print("Found %d layers in %s" % (count,folder_name))
  
  for i in range(count):
    tupla = (np.load(folder_name + "/W_%d.npy" % i), np.load(folder_name + "/b_%d.npy" % i)) 
    ret.append( tupla )
  return ret
##########################################################################################################################################################

# Activation function

def relu(x):
    return jnp.maximum(0, x)

# Inference function

def predict(params, mask, image):
    # per-example predictions
    activations = image
    for m,(w,b) in zip(mask[:-1],params[:-1]):
        outputs = jnp.dot(m * w, activations) + b
        activations = relu(outputs)

    final_mask, (final_w, final_b) = mask[-1], params[-1]
    logits = jnp.dot(final_mask * final_w, activations) + final_b
    return logits - logsumexp(logits)


# Make a batched version of the `predict` function

batched_predict = vmap(predict, in_axes=(None,None, 0))

##########################################################################################################################################################

# Training utilities
step_size = 0.01



def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)
  
def accuracy(params, mask, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params,mask, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params, mask, images, targets):
  preds = batched_predict(params, mask, images)
  return -jnp.mean(preds * targets)

@jit
def update(params, mask, x, y):
  grads = grad(loss)(params,mask, x, y)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]
          