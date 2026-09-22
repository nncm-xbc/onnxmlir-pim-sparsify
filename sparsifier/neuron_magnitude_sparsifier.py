"""Magnitude-scored *neuron* pruning — structured counterpart to
:mod:`sparsifier.magnitude_sparsifier`.

Lifts the weight-level magnitude criterion (score = |w|) to neuron
granularity. A "neuron" is defined exactly as in
:mod:`sparsifier.neuron_sparsifier`: hidden neuron ``i`` of weight matrix
``l`` owns its incoming row ``W[l][i, :]`` **and** its outgoing column
``W[l+1][:, i]`` (plus bias ``b[l][i]``).

Aggregation: **mean** of ``|w|`` over the neuron's active weights.
Mean rather than sum because neurons in different layers have very
different fan-in (e.g. 196 incoming for the first hidden layer vs 10 for
the second), so a raw sum would systematically spare wide-fan-in neurons.
Mean makes the per-neuron score comparable across layers. No data / omega
is needed for scoring; omega/og_net are only used by adjust().

Selection: argmin mean(|w|) — the neuron whose weights are smallest on
average is removed. Removal + adjust are identical to
:mod:`sparsifier.neuron_sparsifier`, so the CSV log is directly comparable.

Public API: :func:`prune_neuron_magnitude`, :func:`run_neuron_sparsifier`,
:class:`NeuronPruneMeta`.
"""

import csv
import json
import os
import sys
import time
from typing import NamedTuple

import numpy as np

from mlp.mlp import Layer, accuracy, load_network_params
from sparsifier.sparsifier import adjust, clone_network, d, make_omega


class NeuronPruneMeta(NamedTuple):
    layer_idx: int        # weight-matrix index l; neuron is row W[l][neuron_idx, :]
    neuron_idx: int       # neuron index i within that layer
    score: float          # aggregated criterion value at the winning candidate
    prune_time_s: float
    adjust_time_s: float


def _neuron_active_weights(net, l, i):
    """Return (values, present) — |w| is applied by the caller. `values` are the
    raw weight values of neuron i (incoming row of W[l] + outgoing col of W[l+1])
    restricted to still-active (mask != 0) positions; `present` is False if the
    neuron has no active weights left."""
    row_m = np.asarray(net[l].mask[i, :]) != 0.0
    col_m = np.asarray(net[l + 1].mask[:, i]) != 0.0
    vals = np.concatenate([
        np.asarray(net[l].W[i, :], dtype=np.float64)[row_m],
        np.asarray(net[l + 1].W[:, i], dtype=np.float64)[col_m],
    ])
    return vals, vals.size > 0


def _remove_neuron(net, l, i):
    """Zero neuron i of layer l in-place: incoming row + bias + outgoing column."""
    net[l].W[i, :] = 0.0
    net[l].mask[i, :] = 0.0
    net[l].b[i] = 0.0
    net[l + 1].W[:, i] = 0.0
    net[l + 1].mask[:, i] = 0.0


def prune_neuron_magnitude(net, og_net, omega, doAdjust=True):
    """Remove the hidden neuron with the smallest mean(|w|); then optionally adjust.

    og_net and omega are unused by selection but required by adjust().
    Returns (pruned_net, NeuronPruneMeta).
    """
    prune_t0 = time.perf_counter()

    best_score = float('inf')
    best_l, best_i = 0, 0
    for l in range(len(net) - 1):
        for i in range(net[l].W.shape[0]):
            if np.all(net[l].mask[i, :] == 0.0):
                continue  # neuron already pruned
            vals, present = _neuron_active_weights(net, l, i)
            if not present:
                continue
            score = float(np.abs(vals).mean())
            if score < best_score:
                best_score, best_l, best_i = score, l, i

    prune_time_s = time.perf_counter() - prune_t0

    result_net = clone_network(net)
    _remove_neuron(result_net, best_l, best_i)

    adjust_t0 = time.perf_counter()
    if doAdjust and best_score > 0:
        result_net = adjust(result_net, og_net, omega)
    adjust_time_s = time.perf_counter() - adjust_t0

    return result_net, NeuronPruneMeta(
        layer_idx=best_l, neuron_idx=best_i, score=best_score,
        prune_time_s=prune_time_s, adjust_time_s=adjust_time_s,
    )


########################################################################
# Shared neuron-sparsification loop.
#
# Byte-for-byte the same loop / CSV schema / checkpoint layout as
# sparsifier.neuron_sparsifier.main() — the ONLY thing that varies between
# criteria is `prune_fn`. neuron_obd/neuron_obs import this so all four
# neuron logs (manifold, magnitude, obd, obs) are directly comparable.
########################################################################


def run_neuron_sparsifier(cfg_path, prune_fn, output_subdir, log_name, loop_label):
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path) as f:
        cfg = json.load(f)

    repo_root     = os.path.dirname(os.path.dirname(cfg_path))
    input_folder  = os.path.join(repo_root, 'artifacts', cfg['name'])
    output_folder = os.path.join(input_folder, output_subdir)

    def resolve(p):
        return os.path.join(repo_root, p)

    x_test = np.genfromtxt(resolve(cfg['data']['x_test']), delimiter=',', max_rows=1000)
    y_test = np.genfromtxt(resolve(cfg['data']['y_test']), delimiter=',', max_rows=1000)

    sp = cfg['sparsify']

    import mlp.mlp as _mlp
    _mlp.hidden_activation = _mlp._ACTS[cfg.get('hidden_activation', 'relu')]

    print("Load the parameters from the folder")
    og_net = load_network_params(input_folder)
    print("Accuracy in validation: %.4f" % float(accuracy(og_net, x_test, y_test)))

    hidden_layer_indices = list(range(len(og_net) - 1))
    total_neurons = int(sum(og_net[l].W.shape[0] for l in hidden_layer_indices))
    print("Total hidden neurons: %d" % total_neurons)

    omega = make_omega(og_net, n_samples=sp['omega_samples'])
    print(
        "Perturbation distance (sanity check): %.4e"
        % float(d(og_net, [
            Layer(W=l.W + np.random.normal(size=l.W.shape) * 0.00001, b=l.b, mask=l.mask)
            for l in og_net
        ], omega))
    )

    print("Starting %s loop" % loop_label)
    net = clone_network(og_net)

    os.makedirs(output_folder, exist_ok=True)

    log_path = os.path.join(output_folder, 'neuron_sparsification_log.csv')
    with open(log_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)

        layer_neuron_cols = ['layer_%d_neurons' % l for l in hidden_layer_indices]
        header = [
            'step', 'neurons_pruned', 'total_neurons', 'neuron_sparsity',
            'val_acc', 'd_manifold', 'd_W',
            'prune_time_s', 'adjust_time_s',
            'candidate_layer', 'candidate_neuron',
        ] + layer_neuron_cols
        writer.writerow(header)

        for step in range(sp['steps']):
            layer_neuron_counts = [
                int((net[l].W != 0).any(axis=1).sum())
                for l in hidden_layer_indices
            ]
            neurons_pruned  = total_neurons - sum(layer_neuron_counts)
            neuron_sparsity = neurons_pruned / total_neurons
            val_acc         = float(accuracy(net, x_test, y_test))
            d_manifold      = float(d(net, og_net, omega))

            print(
                "step {:4d} | acc={:.4f} | neurons_pruned={:4d}/{:4d} | "
                "sparsity={:.4f} | d_m={:.4e}".format(
                    step, val_acc, neurons_pruned, total_neurons,
                    neuron_sparsity, d_manifold,
                )
            )

            W_snapshot = [np.array(l.W).copy() for l in net]
            net, meta  = prune_fn(net, og_net, omega, doAdjust=sp['do_adjust'])
            d_W = float(
                np.sqrt(sum(
                    np.sum((np.array(l.W) - w) ** 2)
                    for l, w in zip(net, W_snapshot)
                ))
            )

            layer_neuron_counts_after = [
                int((net[l].W != 0).any(axis=1).sum())
                for l in hidden_layer_indices
            ]
            writer.writerow([
                step,
                total_neurons - sum(layer_neuron_counts_after),
                total_neurons,
                round(neuron_sparsity, 6),
                round(val_acc, 6),
                '{:.6e}'.format(d_manifold),
                '{:.6e}'.format(d_W),
                round(meta.prune_time_s, 4),
                round(meta.adjust_time_s, 4),
                meta.layer_idx,
                meta.neuron_idx,
            ] + layer_neuron_counts_after)
            log_file.flush()

            if step % sp['checkpoint_every'] == 0:
                ckpt_dir = os.path.join(output_folder, 'checkpoints', 'step_%04d' % step)
                os.makedirs(ckpt_dir, exist_ok=True)
                for li, layer in enumerate(net):
                    np.save(os.path.join(ckpt_dir, 'W_%d.npy' % li), layer.W)

    print("Neuron sparsification log saved to:", log_path)
    for i, layer in enumerate(net):
        np.save(os.path.join(output_folder, 'W_%i.npy' % i), layer.W)
        np.save(os.path.join(output_folder, 'b_%i.npy' % i), layer.b)
    # Shape-shrunk copy for crossbar backends (pruned neurons physically removed).
    from backend.compact import save_compacted
    save_compacted(net, output_folder)


def _selfcheck():
    """Tiny synthetic check: the neuron with the smallest weights is chosen and
    fully removed (row + matching column zeroed)."""
    rng = np.random.default_rng(0)
    def layer(nout, nin, scale):
        W = (rng.standard_normal((nout, nin)) * scale)
        return Layer(W=W, b=rng.standard_normal(nout), mask=np.ones((nout, nin)))
    net = [layer(3, 4, 1.0), layer(3, 3, 1.0), layer(2, 3, 1.0)]
    # Make neuron (l=0, i=1) unambiguously the smallest-magnitude neuron.
    net[0].W[1, :] = 1e-6
    net[1].W[:, 1] = 1e-6
    omega = make_omega(net, n_samples=50)
    pruned, meta = prune_neuron_magnitude(net, net, omega, doAdjust=False)
    assert (meta.layer_idx, meta.neuron_idx) == (0, 1), (meta.layer_idx, meta.neuron_idx)
    assert np.all(pruned[0].W[1, :] == 0) and np.all(pruned[1].W[:, 1] == 0)
    assert np.all(pruned[0].mask[1, :] == 0) and np.all(pruned[1].mask[:, 1] == 0)
    print("neuron_magnitude selfcheck OK:", meta)


def main():
    run_neuron_sparsifier(
        sys.argv[1], prune_neuron_magnitude,
        output_subdir='neuron_magnitude_sparsified',
        log_name='neuron_sparsification_log.csv',
        loop_label='Neuron magnitude sparsification',
    )


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selfcheck':
        _selfcheck()
    else:
        main()
