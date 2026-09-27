"""Semantic pullback metric  ĝ = JᵀJ / B  of the trained MLP (thesis figure).

J is the Jacobian of the batched log-softmax output F̂: ℝ^N → ℝ^{B·10}
on the Ω sample (B = 10⁴, np.random.seed(0)); parameters are flattened in
the order (W_0, b_0, W_1, b_1, W_2, b_2), row-major.

Panels:
  (a) eigenvalue spectrum of ĝ at the dense seed-0 network, numerical kernel dim
  (b) exact removal cost d̂ = d/B vs second-order estimate ĝ_ii w_i² per weight
  (c) |A|, rank and effective rank of ĝ_A on the active set along e05

Outputs:
  /home/simon/repos/ThesisTex/main_doc/Images/fig_pullback_metric.pdf
  /tmp/qloop/numerics.md   (every number, with the command that produced it)

Usage (from the repo root; needs the GPU, never alongside another JAX process):
  JAX_PLATFORMS=cuda PYTHONPATH=. /home/simon/venv/general/bin/python3 visualize/fig_pullback_metric.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.stats import pearsonr, spearmanr

from mlp.mlp import Layer, batched_predict, load_network_params
from sparsifier.sparsifier import make_omega, d, clone_network, _zero_weight
from sparsifier.obd_sparsifier import _diag_hessian
from visualize.thesis_figures import _save, REPO, OUT_DEFAULT, plt

BASELINE = os.path.join(REPO, 'artifacts', 'baseline')
E05 = os.path.join(REPO, 'artifacts', 'e05_stupidity_point', 'sparsified')
NUMERICS = '/tmp/qloop/numerics.md'
COMMAND = ('cd /home/simon/repos/onnxmlir-pim-sparsify && JAX_PLATFORMS=cuda PYTHONPATH=. '
           '/home/simon/venv/general/bin/python3 visualize/fig_pullback_metric.py')
B = 10000
CHUNK = 500
THRESHOLDS = (1e-8, 1e-10, 1e-12, 1e-15)
COLLAPSE_SPARSITY = 90.56


# ------------------------------------------------------------------ flattening
def load_f64(folder):
    return [Layer(W=np.asarray(l.W, np.float64), b=np.asarray(l.b, np.float64),
                  mask=np.asarray(l.mask, np.float64)) for l in load_network_params(folder)]


def load_ckpt_W(folder, biases):
    """e05 checkpoints are W-only (written before the runner saved biases):
    take W (mask = W != 0) from the checkpoint and the biases from `biases`."""
    n = len(biases)
    return [Layer(W=np.load(os.path.join(folder, 'W_%d.npy' % k)).astype(np.float64),
                  b=np.asarray(biases[k], np.float64),
                  mask=(np.load(os.path.join(folder, 'W_%d.npy' % k)) != 0).astype(np.float64))
            for k in range(n)]


def unit_stats(net, omega):
    """Per hidden layer: (#dead units, #always-on units) on Ω (ReLU pre-activations)."""
    a, out = omega.T, []
    for l in net[:-1]:
        z = (l.mask * l.W) @ a + l.b[:, None]
        out.append((int((z < 0).all(1).sum()), int((z > 0).all(1).sum())))
        a = np.maximum(z, 0)
    return out


def index_map(net):
    """flat index -> (layer, 'W'|'b', i, j); j = -1 for biases."""
    rows = []
    for l, layer in enumerate(net):
        n_out, n_in = layer.W.shape
        rows += [(l, 'W', i, j) for i in range(n_out) for j in range(n_in)]
        rows += [(l, 'b', i, -1) for i in range(n_out)]
    return rows


def flatten(net):
    return np.concatenate([np.concatenate([l.W.ravel(), l.b.ravel()]) for l in net])


def active_mask(net):
    """Active coordinates: mask == 1 for weights, every bias."""
    return np.concatenate([np.concatenate([l.mask.ravel() != 0, np.ones(l.b.size, bool)])
                           for l in net])


def make_unflatten(net):
    shapes = [(l.W.shape, l.b.shape) for l in net]
    masks = [jnp.asarray(l.mask) for l in net]

    def unflatten(theta):
        out, k = [], 0
        for (ws, bs), m in zip(shapes, masks):
            nw, nb = ws[0] * ws[1], bs[0]
            out.append(Layer(W=theta[k:k + nw].reshape(ws), b=theta[k + nw:k + nw + nb], mask=m))
            k += nw + nb
        return out
    return unflatten


# ------------------------------------------------------------- pullback metric
def pullback_metric(net, omega, active):
    """ĝ_A = JᵀJ / B restricted to the coordinates where `active` is True.

    Jacobian per chunk of CHUNK samples via jacfwd (N tangents; same matrix as
    jacrev, far less memory than B·10 cotangents); JᵀJ accumulated on device.
    """
    theta0 = jnp.asarray(flatten(net))
    act_idx = jnp.asarray(np.flatnonzero(active))
    unflatten = make_unflatten(net)

    def f(theta_a, x):
        return batched_predict(unflatten(theta0.at[act_idx].set(theta_a)), x).reshape(-1)

    jac = jax.jit(jax.jacfwd(f))
    gram = jax.jit(lambda J: J.T @ J)
    G = jnp.zeros((len(act_idx), len(act_idx)))
    ta = theta0[act_idx]
    for s in range(0, len(omega), CHUNK):
        G = G + gram(jac(ta, jnp.asarray(omega[s:s + CHUNK])))
    return np.asarray(G) / len(omega)


def spectrum(G):
    lam = np.linalg.eigvalsh(G)[::-1]
    return lam


def eff_rank(lam):
    p = np.clip(lam, 0, None)
    p = p / p.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# ----------------------------------------------------------------------- main
def main():
    t_all = time.time()
    print('jax backend:', jax.default_backend(), jax.devices())
    os.makedirs(os.path.dirname(NUMERICS), exist_ok=True)
    lines = [f'# Pullback-metric numerics\n\nProduced by:\n\n```\n{COMMAND}\n```\n',
             f'Conventions: ĝ = JᵀJ/B with B = {B}, J = Jacobian of batched log-softmax '
             'output on Ω (np.random.seed(0); make_omega(net, 10000)); d̂ := d/B where d '
             'is sparsifier.d (sum over samples and outputs). Flat order (W_0,b_0,W_1,b_1,W_2,b_2).\n']

    net = load_f64(BASELINE)
    e05_dense = load_f64(os.path.dirname(E05))
    print('baseline vs e05 top-level params, max |diff|:',
          max(float(np.abs(a.W - b.W).max()) for a, b in zip(net, e05_dense)))
    imap = index_map(net)
    N = len(imap)
    n_w = sum(l.W.size for l in net)
    n_zero_w = sum(int((l.W == 0).sum()) for l in net)
    print(f'N = {N} ({n_w} weights + {N - n_w} biases); exactly-zero weights in dense net: {n_zero_w}')
    np.random.seed(0)
    omega = make_omega(net, B)
    omega_dev = jnp.asarray(omega)

    # ---------------------------------------------------------------- (a)
    t0 = time.time()
    G = pullback_metric(net, omega, np.ones(N, bool))
    lam = spectrum(G)
    lam_max = lam[0]
    kdim = {thr: int((lam < thr * lam_max).sum()) for thr in THRESHOLDS}
    k10 = kdim[1e-10]
    lam_min_nonker = lam[N - k10 - 1]
    print(f'(a) {time.time() - t0:.1f}s  lam_max={lam_max:.6e}  kernel dim @1e-8/1e-10/1e-12/1e-15 = '
          f'{kdim[1e-8]}/{k10}/{kdim[1e-12]}/{kdim[1e-15]}  lam_min(non-kernel)={lam_min_nonker:.6e}  '
          f'lam[N-1]={lam[-1]:.3e}  sym err={np.abs(G - G.T).max():.2e}')
    # which coordinates carry an exactly-zero diagonal (dead-unit coordinates)
    diag = np.diag(G)
    zero_diag = [imap[i] for i in np.flatnonzero(diag == 0)]
    zd_summary = {}
    for l, kind, i, j in zero_diag:
        zd_summary[(l, kind)] = zd_summary.get((l, kind), 0) + 1
    print('    exactly-zero ĝ_ii by (layer, kind):', zd_summary)
    ustats = unit_stats(net, omega)
    print('    hidden units (dead, always-on) per layer on Ω:', ustats)
    tail = [(N - k, lam[N - k - 1] / lam_max) for k in range(290, 345)]

    # ---------------------------------------------------------------- (b)
    t0 = time.time()
    w_idx = [k for k, r in enumerate(imap) if r[1] == 'W']
    c_exact = np.zeros(n_w)
    for n, k in enumerate(w_idx):
        l, _, i, j = imap[k]
        probe = clone_network(net)
        probe[l] = _zero_weight(net[l], i, j)
        c_exact[n] = float(d(net, probe, omega_dev)) / B
    theta = flatten(net)
    c_est = diag[w_idx] * theta[w_idx] ** 2
    z_exact = c_exact == 0
    z_est = c_est == 0
    both = ~z_exact & ~z_est
    lc, le = np.log10(c_exact[both]), np.log10(c_est[both])
    rho = spearmanr(lc, le).correlation
    r = pearsonr(lc, le)[0]
    ratio = c_est[both] / c_exact[both]
    frac2 = float(((ratio >= 0.5) & (ratio <= 2)).mean())
    med_ratio = float(np.median(ratio))
    # OBD diagonal: H_ii of d_W (sum normalisation) ⇒ H_ii/(2B) should equal ĝ_ii
    H = np.concatenate([np.asarray(h).ravel() for h in _diag_hessian(net, net, omega_dev)])
    obd_disc = float(np.abs(H / (2 * B) - diag[w_idx]).max() / diag[w_idx].max())
    print(f'(b) {time.time() - t0:.1f}s  #c=0: {z_exact.sum()}  #ĉ=0: {z_est.sum()}  '
          f'sets equal: {np.array_equal(z_exact, z_est)}  |c=0 & ĉ>0|={int((z_exact & ~z_est).sum())} '
          f'|c>0 & ĉ=0|={int((~z_exact & z_est).sum())}  spearman={rho:.4f} pearson(log)={r:.4f} '
          f'frac within 2x={frac2:.4f} median ratio={med_ratio:.4f}  OBD max disc={obd_disc:.3e}')
    print(f'    c_exact (non-zero) range [{c_exact[~z_exact].min():.3e}, {c_exact.max():.3e}]  '
          f'c_est (non-zero) range [{c_est[~z_est].min():.3e}, {c_est.max():.3e}]  '
          f'ratio range [{ratio.min():.3e}, {ratio.max():.3e}]')

    # ---------------------------------------------------------------- (c)
    t0 = time.time()
    log = pd.read_csv(os.path.join(E05, 'sparsification_log.csv'))
    # e05 checkpoints hold W only; biases are taken from the dense net (b0) and,
    # as a second variant, from the final sparsified net (bf). Neither is the
    # true mid-run bias (adjust() moves biases by O(1..30) over the run).
    b_variants = {'b0': [l.b for l in net], 'bf': [l.b for l in load_f64(E05)]}
    rows = []
    for name in sorted(os.listdir(os.path.join(E05, 'checkpoints'))):
        step = int(name.split('_')[1])
        # checkpoint step_k is written after the k-th removal (step_0000 has 2159
        # weights), so take sparsity from the checkpoint itself, not the log row.
        nnz = sum(int((np.load(os.path.join(E05, 'checkpoints', name, 'W_%d.npy' % k)) != 0).sum())
                  for k in range(len(net)))
        sp = 100.0 * (1 - nnz / n_w)
        rec = {'step': step, 'sparsity_pct': sp, 'nnz_W': nnz,
               'log_sparsity_pct': float(log.loc[log.step == step, 'sparsity'].iloc[0]) * 100}
        for tag, bias in b_variants.items():
            try:
                cnet = load_ckpt_W(os.path.join(E05, 'checkpoints', name), bias)
                act = active_mask(cnet)
                lam_c = spectrum(pullback_metric(cnet, omega, act))
            except Exception as e:  # report and continue
                print(f'    checkpoint {name} ({tag}) FAILED: {e!r}')
                continue
            rec['n_active'] = int(act.sum())
            rec[f'rank_{tag}'] = int((lam_c >= 1e-10 * lam_c[0]).sum())
            rec[f'eff_{tag}'] = eff_rank(lam_c)
            rec[f'lmax_{tag}'] = float(lam_c[0])
            rec[f'units_{tag}'] = unit_stats(cnet, omega)
        rows.append(rec)
        print('    ' + '  '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
                                for k, v in rec.items()), flush=True)
    ck = pd.DataFrame(rows)
    print(f'(c) {time.time() - t0:.1f}s  {len(ck)} checkpoints')

    # ------------------------------------------------------------- figure
    plt.rcParams['axes.titlesize'] = 8
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.0, 2.6))

    floor = 1e-18 * lam_max  # eigenvalues ≤ 0 cannot be drawn on a log axis
    ax_a.semilogy(np.arange(1, N + 1), np.maximum(lam, floor), color='C0', lw=0.9)
    ax_a.axvline(N - k10 + 0.5, color='C3', lw=0.8, ls='--')
    ax_a.annotate(rf'$\dim\ker\hat g = {k10}$' + f'\n(of N = {N})', xy=(N - k10, lam_max * 1e-14),
                  xytext=(N * 0.08, lam_max * 1e-15), fontsize=7, color='C3', va='top',
                  arrowprops=dict(arrowstyle='->', lw=0.6, color='C3'))
    ax_a.axhline(1e-10 * lam_max, color='0.6', lw=0.6, ls=':')
    ax_a.text(N * 0.03, 1e-10 * lam_max * 2.5, r'$10^{-10}\lambda_{\max}$', fontsize=7, color='0.4')
    ax_a.set_xlabel('eigenvalue index (descending)')
    ax_a.set_ylabel(r'eigenvalue of $\hat g(w^0)$')
    ax_a.set_title(r'(a) spectrum of $\hat g$ at the dense network')

    ax_b.loglog(c_exact[both], c_est[both], '.', ms=2.5, alpha=0.5, color='C0', mew=0)
    lo, hi = c_exact[both].min() * 0.5, c_exact[both].max() * 2
    ax_b.plot([lo, hi], [lo, hi], color='0.3', lw=0.8, ls='--', label='identity')
    ax_b.set_xlabel(r'exact removal cost $c_i$')
    ax_b.set_ylabel(r'$\hat g_{ii}(w^0)\, w_i^2$')
    ax_b.set_title(r'(b) exact removal cost vs $g_{ii} w_i^2$')
    ax_b.text(0.03, 0.97, rf'$\rho_s = {rho:.3f}$' + f'\nwithin 2x: {100 * frac2:.1f}%'
              + f'\n{int(z_exact.sum())} zero-cost weights omitted',
              transform=ax_b.transAxes, fontsize=7, va='top')
    ax_b.legend(loc='lower right', frameon=False)

    ax_c.plot(ck.sparsity_pct, ck.n_active, color='0.4', lw=0.9, label=r'$|A|$ (active coords)')
    ax_c.plot(ck.sparsity_pct, ck.rank_b0, color='C0', lw=1.1, label=r'rank $\hat g_A$ ($\geq 10^{-10}\lambda_{\max}$)')
    ax_c.plot(ck.sparsity_pct, ck.eff_b0, color='C1', lw=1.1, label='effective rank')
    ax_c.plot(ck.sparsity_pct, ck.rank_bf, color='C0', lw=0.8, ls=':', label='same, final-run biases')
    ax_c.plot(ck.sparsity_pct, ck.eff_bf, color='C1', lw=0.8, ls=':')
    ax_c.set_yscale('log')
    ax_c.axvline(COLLAPSE_SPARSITY, color='C3', lw=0.8, ls='--')
    ax_c.text(COLLAPSE_SPARSITY - 1.5, 2.0, f'collapse\n{COLLAPSE_SPARSITY}%',
              ha='right', va='bottom', fontsize=7, color='C3')
    ax_c.set_ylim(1.5, 4000)
    ax_c.set_xlabel('weight sparsity (%)')
    ax_c.set_ylabel('count')
    ax_c.set_title(r'(c) rank of $\hat g$ on the active set along the run')
    ax_c.legend(loc='upper right', frameon=False, fontsize=7)

    _save(fig, OUT_DEFAULT, 'fig_pullback_metric')

    # ------------------------------------------------------------ numerics
    wall = time.time() - t_all
    lines.append('## Findings\n\n| quantity | value | how computed |\n|---|---|---|')
    lines += [
        f'| N (flat parameters) | {N} = {n_w} W + {N - n_w} b | index_map |',
        f'| exactly-zero weights in dense net | {n_zero_w} | (W == 0).sum() |',
        f'| λ_max(ĝ(w⁰)) | {lam_max:.6e} | eigvalsh(JᵀJ/B) |',
        f'| kernel dim @ 1e-8 λ_max | {kdim[1e-8]} | #eigenvalues < thr·λ_max |',
        f'| kernel dim @ 1e-10 λ_max | {k10} | idem |',
        f'| kernel dim @ 1e-12 λ_max | {kdim[1e-12]} | idem |',
        f'| kernel dim @ 1e-15 λ_max (float64 zero) | {kdim[1e-15]} | idem |',
        f'| λ_min(non-kernel, 1e-10) | {lam_min_nonker:.6e} | λ[N−k−1] |',
        f'| smallest eigenvalue (raw) | {lam[-1]:.3e} | eigvalsh |',
        f'| coordinates with ĝ_ii == 0 exactly | {len(zero_diag)} by (layer,kind): {zd_summary} | diag(ĝ) == 0 |',
        f'| #weights with exact cost c_i = 0 | {int(z_exact.sum())} | d(net, net with W[i,j]=0, mask=0, Ω)/B |',
        f'| #weights with estimate ĉ_i = 0 | {int(z_est.sum())} | ĝ_ii w_i² |',
        f'| zero sets coincide? | {np.array_equal(z_exact, z_est)} (c=0&ĉ>0: {int((z_exact & ~z_est).sum())}, c>0&ĉ=0: {int((~z_exact & z_est).sum())}) | set comparison |',
        f'| #non-zero pairs used for stats | {int(both.sum())} | |',
        f'| Spearman ρ (log10 pairs) | {rho:.4f} | scipy.stats.spearmanr |',
        f'| Pearson r (log10 pairs) | {r:.4f} | scipy.stats.pearsonr |',
        f'| fraction with ĉ_i/c_i ∈ [1/2, 2] | {frac2:.4f} | |',
        f'| median ĉ_i/c_i | {med_ratio:.4f} | np.median |',
        f'| ratio range | [{ratio.min():.3e}, {ratio.max():.3e}] | |',
        f'| c_i non-zero range (d̂ units) | [{c_exact[~z_exact].min():.3e}, {c_exact.max():.3e}] | |',
        f'| OBD: max_i abs(H_ii/(2B) − ĝ_ii) / max ĝ_ii | {obd_disc:.3e} | _diag_hessian(net, net, Ω) |',
        f'| hidden units (dead, always-on) per layer at w⁰ on Ω | {ustats} | unit_stats (pre-activation sign over all 10⁴ samples) |',
        f'| wall-clock total | {wall:.0f} s | time.time() |',
        f'| JAX backend | {jax.default_backend()} | |',
    ]
    lines.append('\n## Eigenvalue tail of ĝ(w⁰) around the kernel boundary (index k from the top, λ_k/λ_max)\n')
    lines.append(' '.join(f'{k}:{v:.2e}' for k, v in tail) + '\n')
    lines.append('\n## (c) e05 checkpoints: ĝ_A on the active set\n\n'
                 'WARNING: e05 checkpoints contain W only (no b_k.npy). Columns *_b0 use the dense biases, '
                 '*_bf the final sparsified biases; the true mid-run biases are unknown. '
                 '|A| = active weights + 30 biases. units = [(dead, always-on) for hidden layer 1, 2].\n\n'
                 'sparsity % is computed from the checkpoint W (nnz/2160); the log row for the same step '
                 'reports one weight more (checkpoint k is written after removal k).\n\n'
                 '| step | sparsity % | abs(A) | rank_b0 | eff_b0 | λ_max_b0 | units_b0 | rank_bf | eff_bf | λ_max_bf | units_bf |\n'
                 '|---|---|---|---|---|---|---|---|---|---|---|')
    lines += [f"| {r['step']} | {r['sparsity_pct']:.2f} | {r['n_active']} | {r['rank_b0']} | {r['eff_b0']:.2f} | "
              f"{r['lmax_b0']:.4e} | {r['units_b0']} | {r['rank_bf']} | {r['eff_bf']:.2f} | {r['lmax_bf']:.4e} | {r['units_bf']} |"
              for r in rows]
    lines.append(f'\nFigure: {os.path.join(OUT_DEFAULT, "fig_pullback_metric.pdf")}\n')
    with open(NUMERICS, 'w') as fh:
        fh.write('\n'.join(lines))
    print('wrote', NUMERICS)
    print(f'total wall-clock {wall:.0f}s')


if __name__ == '__main__':
    main()
