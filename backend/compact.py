"""Compaction: turn structured (neuron-level) sparsity into a smaller dense net.

The neuron sparsifiers (``sparsifier/neuron_*.py``) *zero* a pruned neuron's
fan-in row, bias and fan-out column but never change tensor shapes — so the
saved ``W_i.npy`` keep their original dimensions. Fed to a shape-driven
crossbar compiler (onnx-mlir's PIM backend: cores/crossbars/mvmuls come from
``ceil(dim / crossbar_size)``, weight *values* are never inspected), that
sparsity buys nothing: a pruned net compiles to the same hardware as the dense
one.

This module physically removes the dead hidden neurons, producing a strictly
smaller net that computes the identical function (mathematically; numerically
equal up to float summation order, ~1 ulp, since XLA reduces differently shaped
matmuls in a different order). A hidden neuron is
removable iff its fan-out (its column in the next layer's ``W``) is entirely
zero — then its output is read by nothing and dropping its row of ``W[l]``,
``b[l]`` and its column of ``W[l+1]`` cannot change the output, for any
activation. The neuron pruners guarantee both sides are zeroed, so every pruned
neuron qualifies.

Input dims (columns of ``W[0]``) and output dims (rows of ``W[-1]``) are never
touched — they are the network's I/O contract. Input-feature compaction (P3,
dropping all-zero input columns) is a separate transform because it changes the
input shape.

Public API: :func:`compact_network`, :func:`compact_params`, :func:`crossbar_cost`.
"""

import math
import os
import sys

import numpy as np

from mlp.mlp import Layer, load_network_params


def compact_network(net):
    """Return a copy of ``net`` with fan-out-dead hidden neurons removed.

    ``net`` is a list of :class:`mlp.mlp.Layer`. The returned net computes the
    identical function on every input. Masks are rebuilt as ``W != 0``.
    """
    L = len(net)
    # keep_out[l]: indices of layer l's output neurons to retain.
    keep_out = [None] * L
    keep_out[L - 1] = np.arange(net[L - 1].W.shape[0])  # output classes: keep all
    for l in range(L - 1):
        # hidden neuron i (row i of W[l]) survives iff consumed downstream,
        # i.e. column i of W[l+1] has any non-zero.
        fanout_nonzero = np.any(net[l + 1].W != 0, axis=0)
        keep_out[l] = np.nonzero(fanout_nonzero)[0]

    out = []
    for l in range(L):
        rows = keep_out[l]
        cols = np.arange(net[l].W.shape[1]) if l == 0 else keep_out[l - 1]
        W = net[l].W[np.ix_(rows, cols)]
        b = net[l].b[rows]
        out.append(Layer(W=W, b=b, mask=(W != 0).astype(W.dtype)))
    return out


def topology_of(net):
    """[in, h1, ..., out] layer sizes of a net."""
    return [net[0].W.shape[1]] + [l.W.shape[0] for l in net]


def crossbar_cost(topology, crossbar_size=128, crossbars_per_core=64):
    """Crossbars / cores / mvmuls a Gemm-only MLP needs on the PIM backend.

    Mirrors onnx-mlir-priv exactly: ``AnnotateReplication.cpp`` (cores) and
    ``Gemm.cpp`` (tiling). One Gemm per consecutive layer pair; for (n_in,n_out):
    input_tiles=ceil(n_in/S), output_tiles=ceil(n_out/S),
    cores=ceil(input_tiles/C)*output_tiles, crossbars=mvmuls=input_tiles*output_tiles.
    """
    def ceil(a, b):
        return -(-a // b)

    crossbars = cores = mvmuls = 0
    for n_in, n_out in zip(topology[:-1], topology[1:]):
        it = ceil(n_in, crossbar_size)
        ot = ceil(n_out, crossbar_size)
        crossbars += it * ot
        mvmuls += it * ot
        cores += ceil(it, crossbars_per_core) * ot
    return {"crossbars": crossbars, "cores": cores, "mvmuls": mvmuls}


def compact_params(in_folder, out_folder):
    """Load a params folder, compact it, save W_i/b_i to ``out_folder``.

    Returns ``(old_topology, new_topology)``.
    """
    net = load_network_params(in_folder)
    old_topo = topology_of(net)
    comp = compact_network(net)
    new_topo = topology_of(comp)
    os.makedirs(out_folder, exist_ok=True)
    for i, layer in enumerate(comp):
        np.save(os.path.join(out_folder, "W_%d.npy" % i), layer.W)
        np.save(os.path.join(out_folder, "b_%d.npy" % i), layer.b)
    return old_topo, new_topo


def save_compacted(net, output_folder, crossbar_size=128):
    """Save the compacted ``net`` to ``<output_folder>/compact/`` + crossbar cost.

    Called by the neuron sparsifiers after their final save, so every structured
    run leaves a hardware-ready (shape-shrunk) net next to the zeroed one.
    """
    import json
    comp = compact_network(net)
    old, new = topology_of(net), topology_of(comp)
    folder = os.path.join(output_folder, "compact")
    os.makedirs(folder, exist_ok=True)
    for i, layer in enumerate(comp):
        np.save(os.path.join(folder, "W_%d.npy" % i), layer.W)
        np.save(os.path.join(folder, "b_%d.npy" % i), layer.b)
    report = {
        "crossbar_size": crossbar_size,
        "topology_before": [int(x) for x in old],
        "topology_after": [int(x) for x in new],
        "cost_before": crossbar_cost(old, crossbar_size=crossbar_size),
        "cost_after": crossbar_cost(new, crossbar_size=crossbar_size),
    }
    with open(os.path.join(folder, "crossbar_cost.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Compacted %s -> %s | crossbars %d -> %d (S=%d)" % (
        old, new, report["cost_before"]["crossbars"],
        report["cost_after"]["crossbars"], crossbar_size))
    return comp


def _demo():
    """Self-check: compaction preserves the function and actually shrinks the net."""
    import mlp.mlp as mlp

    rng = np.random.default_rng(0)
    topo = [8, 6, 5, 3]
    net = []
    for n_in, n_out in zip(topo[:-1], topo[1:]):
        W = rng.standard_normal((n_out, n_in))
        b = rng.standard_normal(n_out)
        net.append(Layer(W=W, b=b, mask=np.ones((n_out, n_in))))

    # Prune hidden neuron 2 of layer-0 output and neuron 1 of layer-1 output:
    # zero fan-in row + bias + fan-out column (what the neuron pruners do).
    net[0] = net[0]._replace(W=net[0].W.copy(), b=net[0].b.copy())
    net[1] = net[1]._replace(W=net[1].W.copy(), b=net[1].b.copy())
    net[2] = net[2]._replace(W=net[2].W.copy())
    net[0].W[2, :] = 0.0; net[0].b[2] = 0.0; net[1].W[:, 2] = 0.0
    net[1].W[1, :] = 0.0; net[1].b[1] = 0.0; net[2].W[:, 1] = 0.0
    net = [Layer(W=l.W, b=l.b, mask=(l.W != 0).astype(l.W.dtype)) for l in net]

    comp = compact_network(net)

    assert topology_of(net) == [8, 6, 5, 3]
    assert topology_of(comp) == [8, 5, 4, 3], topology_of(comp)

    X = rng.standard_normal((32, 8))
    y0 = np.asarray(mlp.batched_predict(net, X))
    y1 = np.asarray(mlp.batched_predict(comp, X))
    assert np.allclose(y0, y1, rtol=1e-12, atol=1e-12), "compaction changed the function (max |d|=%g)" % np.abs(y0 - y1).max()

    c0 = crossbar_cost([8, 6, 5, 3], crossbar_size=4, crossbars_per_core=64)
    c1 = crossbar_cost([8, 5, 4, 3], crossbar_size=4, crossbars_per_core=64)
    assert c1["crossbars"] <= c0["crossbars"]
    print("compact.py self-check OK:", topology_of(net), "->", topology_of(comp),
          "| crossbars", c0["crossbars"], "->", c1["crossbars"])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _demo()
    elif len(sys.argv) == 3:
        old, new = compact_params(sys.argv[1], sys.argv[2])
        S = int(os.environ.get("CROSSBAR_SIZE", "128"))
        c_old = crossbar_cost(old, crossbar_size=S)
        c_new = crossbar_cost(new, crossbar_size=S)
        print("topology:  %s -> %s" % (old, new))
        print("crossbars: %d -> %d  (%.0f%%)" % (
            c_old["crossbars"], c_new["crossbars"],
            100 * (1 - c_new["crossbars"] / c_old["crossbars"]) if c_old["crossbars"] else 0))
        print("cores:     %d -> %d" % (c_old["cores"], c_new["cores"]))
        print("mvmuls:    %d -> %d  (@ crossbar_size=%d)" % (c_old["mvmuls"], c_new["mvmuls"], S))
    else:
        print("usage: python -m backend.compact [<in_params_folder> <out_params_folder>]")
        sys.exit(2)
