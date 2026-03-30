# sparsifier.py

from mlp.mlp import *
import jax
import sys

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Build a random sample from the input domain Ω (uniform pixel noise).
# input_dim is inferred from the first layer weight matrix shape (out, in).
def make_omega(network, n_samples=10000):
    input_dim = network[0].W.shape[1]
    return np.random.randint(0, 256, size=(n_samples, input_dim)).astype(np.float32)

# Estimate the manifold distance between two networks by comparing
# their outputs over a shared sample from Ω.
def d(net_1, net_2, omega):
    return jnp.sum(
        (batched_predict(net_1, omega)
       - batched_predict(net_2, omega)) ** 2
    )
d = jax.jit(d)
d_grad = jax.grad(d)
d_grad = jax.jit(d_grad)

# construct a copy of the network
def clone_network(network):
    return [Layer(W=np.array(l.W).copy(), b=np.array(l.b).copy(), mask=np.array(l.mask).copy())
            for l in network]

def _zero_weight(layer, i, j):
    new_W    = np.array(layer.W).copy();    new_W[i, j]    = 0.
    new_mask = np.array(layer.mask).copy(); new_mask[i, j] = 0.
    return Layer(W=new_W, b=layer.b, mask=new_mask)


########################################################################################################################################
# Adjust Function : minimize the distance on the plane locally isomorphic to the manifold of parameters.
# given
#       - a network not adjusted
#       - a network that we would like to mimic semantically
#
# Returns
#       - a network, with the same sparsity pattern of the
#         input network but optimized to have minimal local distance
#         in the manifold of parameters with respect to the cmp_net

def adjust(net, cmp_net, omega):
    net  = clone_network(net)
    alfa = 1e-11
    while alfa > 1e-14:
        gradiente  = d_grad(net, cmp_net, omega)
        new_net    = [Layer(W=l.W - alfa * g.W, b=l.b - alfa * g.b, mask=l.mask)
                      for l, g in zip(net, gradiente)]
        if d(new_net, cmp_net, omega) < d(net, cmp_net, omega):
            net   = new_net
            alfa *= 1.001
        else:
            alfa *= 0.5
    return net
########################################################################################################################################


########################################################################################################################################
# Prune Function : function for increasing the sparsity pattern of a network
# given
#       - a network
#
# The function computes every possible variation of the initial network obtainable
# by, for each variation, putting a different weight to 0.
# The optimal candidate is the variation that minimizes the local distance with the
# original network.
#
# returns
#       - a network, with increased sparsity and optionally adjusted to minimize
#         distance to the original network in parameter space
#
def prune(net, og_net, omega, doAdjust=True):
    minimo     = 1e16
    minimo_idx = 0
    minimo_i   = 0
    minimo_j   = 0

    for idx, layer in enumerate(net):
        for i, row in enumerate(layer.W):
            for j, el in enumerate(row):
                if net[idx].W[i, j] != 0.:
                    new_net = clone_network(net)
                    new_net[idx] = _zero_weight(new_net[idx], i, j)
                    distanza = d(og_net, new_net, omega)
                    if distanza < minimo:
                        minimo     = distanza
                        minimo_idx = idx
                        minimo_i   = i
                        minimo_j   = j
                    if minimo == 0:  # weight does not affect distance in parameter space
                        new_net[minimo_idx] = _zero_weight(new_net[minimo_idx], minimo_i, minimo_j)
                        return new_net
    new_net = clone_network(net)
    new_net[minimo_idx] = _zero_weight(new_net[minimo_idx], minimo_i, minimo_j)
    if doAdjust:
        new_net = adjust(new_net, og_net, omega)
    return new_net
########################################################################################################################################


def main():
    args          = sys.argv
    input_folder  = os.path.abspath(args[1])
    output_folder = input_folder + "/sparsified"
    x_test_file   = os.path.abspath(args[2])
    y_test_file   = os.path.abspath(args[3])

    print("Load the validation set for the user to evaluate the goodness")
    x_test = np.genfromtxt(x_test_file, delimiter = ',', max_rows = 1000)
    y_test = np.genfromtxt(y_test_file, delimiter = ',', max_rows = 1000)

    print("Load the parameters from the folder")
    og_net = load_network_params(input_folder)
    print("")
    print("Accuracy in validation:", accuracy(og_net, x_test, y_test))
    print("Construct the omega sample for the sparsification")
    print("")
    perturbed_net = [Layer(W=l.W + np.random.normal(size=l.W.shape) * .00001, b=l.b, mask=l.mask) for l in og_net]
    print("Compute the local distance between a random perturbation of the input network and the input network itself")
    omega = make_omega(og_net)
    print(">>>", d(og_net, perturbed_net, omega))

    print("Starting sparsification loop")
    print("At every iteration the network gets:")
    print("\t 1. Pruned   — remove the least influential parameter")
    print("\t 2. Adjusted — compensate via remaining parameters")
    net = clone_network(og_net)
    for i in range(500):
        NZ = np.sum([(l.W != 0).sum() for l in net])  # non-zero entries
        print("validation accuracy = %.3f" % accuracy(net, x_test, y_test),
              " | non zero elements = %d" % NZ)
        omega = make_omega(net)  # fresh sample each iteration
        net = prune(net, og_net, omega, True)

    # Saving the data
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, l in enumerate(net):
        np.save(output_folder + "/W_%i.npy" % i, l.W)
        np.save(output_folder + "/b_%i.npy" % i, l.b)


if __name__ == "__main__":
    main()
