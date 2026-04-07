# sparsifier.py

from mlp.mlp import *
import csv
import jax
import sys
import time


class PruneMeta(NamedTuple):
    layer_idx:     int
    i:             int
    j:             int
    distanza:      float
    prune_time_s:  float
    adjust_time_s: float

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

    probe_net   = clone_network(net)  # single copy shared across all candidates
    search_done = False
    prune_t0 = time.perf_counter()
    for idx, layer in enumerate(net):
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                if layer.W[i, j] == 0.:
                    continue
                # zero candidate in-place, evaluate, restore
                saved_W    = probe_net[idx].W[i, j]
                saved_mask = probe_net[idx].mask[i, j]
                probe_net[idx].W[i, j]    = 0.
                probe_net[idx].mask[i, j] = 0.
                distanza = d(og_net, probe_net, omega)
                probe_net[idx].W[i, j]    = saved_W
                probe_net[idx].mask[i, j] = saved_mask

                if distanza < minimo:
                    minimo     = distanza
                    minimo_idx = idx
                    minimo_i   = i
                    minimo_j   = j
                    if minimo == 0:  # weight does not affect distance — exit early
                        search_done = True
                        break
            if search_done:
                break
        if search_done:
            break
    prune_time_s = time.perf_counter() - prune_t0

    # apply the winning zero permanently
    probe_net[minimo_idx].W[minimo_i, minimo_j]    = 0.
    probe_net[minimo_idx].mask[minimo_i, minimo_j] = 0.
    adjust_t0 = time.perf_counter()
    if doAdjust and minimo > 0:
        probe_net = adjust(probe_net, og_net, omega)
    adjust_time_s = time.perf_counter() - adjust_t0

    meta = PruneMeta(
        layer_idx=minimo_idx,
        i=minimo_i,
        j=minimo_j,
        distanza=float(minimo),
        prune_time_s=prune_time_s,
        adjust_time_s=adjust_time_s,
    )
    return probe_net, meta
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
    print("Accuracy in validation: %.4f" % float(accuracy(og_net, x_test, y_test)))
    total_W = int(sum(l.W.size for l in og_net))
    print("Total parameters: %d" % total_W)
    perturbed_net = [Layer(W=l.W + np.random.normal(size=l.W.shape) * .00001, b=l.b, mask=l.mask) for l in og_net]
    omega = make_omega(og_net)
    print("Perturbation distance (sanity check): %.4e" % float(d(og_net, perturbed_net, omega)))

    print("Starting sparsification loop")
    print("At every iteration the network gets:")
    print("\t 1. Pruned   — remove the least influential parameter")
    print("\t 2. Adjusted — compensate via remaining parameters")
    net = clone_network(og_net)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    log_path = os.path.join(output_folder, "sparsification_log.csv")
    with open(log_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(['step', 'NZ', 'total_W', 'sparsity', 'val_acc', 'd_manifold', 'd_W'])

        for i in range(500):
            NZ         = int(np.sum([(l.W != 0).sum() for l in net]))
            sparsity   = 1.0 - NZ / total_W
            val_acc    = float(accuracy(net, x_test, y_test))
            omega      = make_omega(net)
            d_manifold = float(d(net, og_net, omega))

            print("step {:4d} | acc={:.4f} | NZ={:6d} | sparsity={:.4f} | d_m={:.4e}".format(
                i, val_acc, NZ, sparsity, d_manifold))

            W_snapshot = [np.array(l.W).copy() for l in net]
            net, meta = prune(net, og_net, omega, True)
            d_W = float(np.sqrt(sum(
                np.sum((np.array(l.W) - w) ** 2) for l, w in zip(net, W_snapshot)
            )))

            writer.writerow([i, NZ, total_W, round(sparsity, 6), round(val_acc, 6),
                             "{:.6e}".format(d_manifold), "{:.6e}".format(d_W)])
            log_file.flush()

    print("Sparsification log saved to:", log_path)

    # Saving the data
    for i, l in enumerate(net):
        np.save(output_folder + "/W_%i.npy" % i, l.W)
        np.save(output_folder + "/b_%i.npy" % i, l.b)


if __name__ == "__main__":
    main()
