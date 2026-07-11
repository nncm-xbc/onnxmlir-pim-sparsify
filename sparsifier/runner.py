"""Shared config→loop→CSV→checkpoint→save driver for weight-pruning sparsifiers.

Every weight-level method (manifold, magnitude, kwon, obd, obs, lazarevich)
runs the identical pipeline: load a JSON config, resolve data paths, load the
network, then iterate {measure NZ/sparsity/acc/d_manifold, snapshot W, call the
method's prune, record d_W, write a CSV row, periodically checkpoint} and
finally save the sparsified weights. Only the prune step, the output location,
the Omega source, and whether an ``adjust_time_s`` column exists differ, so
those are the parameters below.

``prune_fn`` contract:  ``prune_fn(net, og_net, omega, doAdjust, activations)``
returning ``(pruned_net, meta)`` where ``meta`` exposes ``layer_idx``, ``i``,
``j``, ``prune_time_s`` and (unless ``has_adjust_time=False``) ``adjust_time_s``.
"""

import csv
import json
import os

import numpy as np

from mlp.mlp import Layer, accuracy, load_network_params
from sparsifier.sparsifier import clone_network, d, make_omega


def _latest_checkpoint(output_folder):
    """(step, dir) of the highest-numbered step_XXXX checkpoint, or None."""
    ckpt_root = os.path.join(output_folder, 'checkpoints')
    if not os.path.isdir(ckpt_root):
        return None
    steps = []
    for name in os.listdir(ckpt_root):
        d = os.path.join(ckpt_root, name)
        # a resumable checkpoint must have b_*.npy (old W-only ones can't restore
        # the adjusted biases, so we ignore them and start fresh).
        if name.startswith('step_') and os.path.isdir(d) and \
                any(f.startswith('b_') for f in os.listdir(d)):
            try:
                steps.append(int(name[len('step_'):]))
            except ValueError:
                pass
    if not steps:
        return None
    s = max(steps)
    return s, os.path.join(ckpt_root, 'step_%04d' % s)


def _truncate_log(log_path, keep_through_step):
    """Keep the header + data rows whose step (col 0) <= keep_through_step."""
    with open(log_path) as f:
        rows = f.readlines()
    if not rows:
        return
    kept = [rows[0]]
    for line in rows[1:]:
        c0 = line.split(',', 1)[0].strip()
        if c0.isdigit() and int(c0) <= keep_through_step:
            kept.append(line)
    with open(log_path, 'w') as f:
        f.writelines(kept)


def _build_omega(og_net, x_test, sp, omega_source):
    if omega_source == "data":
        # Lazarevich: real calibration images as Omega instead of noise.
        n_omega = min(sp['omega_samples'], len(x_test))
        if n_omega < sp['omega_samples']:
            print("WARNING: omega_samples=%d requested but only %d calibration images available; using all %d." % (sp['omega_samples'], len(x_test), n_omega))
        omega = x_test[:n_omega].astype(np.float64)
        print("Omega: %d real calibration images (data-driven)" % n_omega)
        return omega
    return make_omega(og_net, n_samples=sp['omega_samples'])


def run_sparsifier(cfg_path, prune_fn, *, output_subdir, log_name,
                   omega_source="noise", has_adjust_time=True,
                   loop_label="sparsification"):
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path) as f:
        cfg = json.load(f)

    # paths in config are relative to the repo root (parent of experiments/)
    repo_root     = os.path.dirname(os.path.dirname(cfg_path))
    input_folder  = os.path.join(repo_root, 'artifacts', cfg['name'])
    output_folder = os.path.join(input_folder, output_subdir)

    def resolve(p):
        return os.path.join(repo_root, p)

    x_test = np.genfromtxt(resolve(cfg['data']['x_test']), delimiter=',', max_rows=1000)
    y_test = np.genfromtxt(resolve(cfg['data']['y_test']), delimiter=',', max_rows=1000)

    sp = cfg['sparsify']

    # Set hidden activation before the first JAX trace.
    import mlp.mlp as _mlp
    _mlp.hidden_activation = _mlp._ACTS[cfg.get('hidden_activation', 'relu')]
    # Activation list for the C++ extension (one entry per layer).
    # A topology of T sizes has T-1 weight layers: T-2 hidden + 1 output.
    act_name    = cfg.get('hidden_activation', 'relu')
    activations = [act_name] * (len(cfg['topology']) - 2) + ['linear']

    print("Load the parameters from the folder")
    og_net = load_network_params(input_folder)
    print("Accuracy in validation: %.4f" % float(accuracy(og_net, x_test, y_test)))
    total_W = int(sum(l.W.size for l in og_net))
    print("Total parameters: %d" % total_W)

    # Deterministic Omega so a resumed run reproduces the same sample as the
    # interrupted one (and for reproducibility generally). Was previously unseeded.
    np.random.seed(int(sp.get('omega_seed', cfg.get('train', {}).get('seed', 0))))
    omega = _build_omega(og_net, x_test, sp, omega_source)
    print(
        "Perturbation distance (sanity check): %.4e"
        % float(d(og_net, [
            Layer(W=l.W + np.random.normal(size=l.W.shape) * 0.00001, b=l.b, mask=l.mask)
            for l in og_net
        ], omega))
    )

    print("Starting %s loop" % loop_label)
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, log_name)

    # Resume from the latest checkpoint if one exists and the log is unfinished:
    # a hung/killed run left checkpoints behind, the watchdog relaunches us with
    # the same command, and we pick up where progress was last durably saved.
    # RESUME=0 forces a fresh run.
    start_step, net = 0, clone_network(og_net)
    resume = _latest_checkpoint(output_folder) if os.environ.get('RESUME', '1') != '0' else None
    if resume is not None and os.path.exists(log_path):
        ckpt_step, ckpt_dir = resume
        net = load_network_params(ckpt_dir)      # W,b restored; mask = (W != 0)
        start_step = ckpt_step + 1
        _truncate_log(log_path, ckpt_step)        # drop rows past the checkpoint
        print("RESUME: %s from checkpoint step %d -> starting at step %d"
              % (loop_label, ckpt_step, start_step))

    log_mode = 'a' if start_step > 0 else 'w'
    with open(log_path, log_mode, newline='') as log_file:
        writer = csv.writer(log_file)
        if start_step == 0:
            layer_NZ_cols = ['layer_%d_NZ' % li for li in range(len(og_net))]
            time_cols = ['prune_time_s', 'adjust_time_s'] if has_adjust_time else ['prune_time_s']
            header = [
                'step', 'NZ', 'total_W', 'sparsity', 'val_acc',
                'd_manifold', 'd_W',
            ] + time_cols + [
                'candidate_layer', 'candidate_i', 'candidate_j',
            ] + layer_NZ_cols
            writer.writerow(header)

        for i in range(start_step, sp['steps']):
            NZ = int(np.sum([(l.W != 0).sum() for l in net]))
            sparsity   = 1.0 - NZ / total_W
            val_acc    = float(accuracy(net, x_test, y_test))
            d_manifold = float(d(net, og_net, omega))

            print(
                'step {:4d} | acc={:.4f} | NZ={:6d} | sparsity={:.4f} | d_m={:.4e}'.format(
                    i, val_acc, NZ, sparsity, d_manifold
                )
            )

            W_snapshot = [np.array(l.W).copy() for l in net]
            net, meta  = prune_fn(net, og_net, omega, sp['do_adjust'], activations)
            d_W = float(
                np.sqrt(sum(
                    np.sum((np.array(l.W) - w) ** 2)
                    for l, w in zip(net, W_snapshot)
                ))
            )

            time_vals = ([round(meta.prune_time_s, 4), round(meta.adjust_time_s, 4)]
                         if has_adjust_time else [round(meta.prune_time_s, 4)])
            layer_nz_vals = [int((l.W != 0).sum()) for l in net]
            writer.writerow([
                i, NZ, total_W, round(sparsity, 6), round(val_acc, 6),
                '{:.6e}'.format(d_manifold), '{:.6e}'.format(d_W),
            ] + time_vals + [
                meta.layer_idx, meta.i, meta.j,
            ] + layer_nz_vals)
            log_file.flush()

            if i % sp['checkpoint_every'] == 0:
                ckpt_dir = os.path.join(output_folder, 'checkpoints', 'step_%04d' % i)
                os.makedirs(ckpt_dir, exist_ok=True)
                for li, layer in enumerate(net):
                    np.save(os.path.join(ckpt_dir, 'W_%d.npy' % li), layer.W)
                    np.save(os.path.join(ckpt_dir, 'b_%d.npy' % li), layer.b)

    print("%s log saved to: %s" % (loop_label, log_path))
    for i, l in enumerate(net):
        np.save(os.path.join(output_folder, 'W_%i.npy' % i), l.W)
        np.save(os.path.join(output_folder, 'b_%i.npy' % i), l.b)
