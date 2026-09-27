"""Which directions span the numerical null space of ĝ(w⁰)?  (kernel-basis check)

Candidates: (i) per-live-hidden-unit rescalings, (ii) dead-unit coordinates,
(iii) logit shift, (iv) GL(|S|)+translation reparametrisations of every
always-on block S (units with pre-activation > 0 on all of Ω).  Every candidate
is tested against ĝ = JᵀJ/B built exactly as in fig_pullback_metric.py.

Output: /tmp/qloop/kernel_basis.md (and the same text on stdout).
Usage (from the repo root; needs the GPU, never alongside another JAX process):
  JAX_PLATFORMS=cuda PYTHONPATH=. /home/simon/venv/general/bin/python3 visualize/check_kernel_basis.py
"""
import numpy as np
import jax

from sparsifier.sparsifier import make_omega
from visualize.fig_pullback_metric import BASELINE, B, load_f64, index_map, flatten, pullback_metric

OUT = '/tmp/qloop/kernel_basis.md'
NULL_THR, BAND_THR, RES_THR = 1e-15, 1e-10, 1e-10


def main():
    net = load_f64(BASELINE)
    imap = index_map(net)
    N = len(imap)
    theta = flatten(net)
    np.random.seed(0)
    omega = make_omega(net, B)

    # flat-index helpers (order W_0,b_0,W_1,b_1,W_2,b_2; W row-major)
    off, k = {}, 0
    for l, layer in enumerate(net):
        off[(l, 'W')], k = k, k + layer.W.size
        off[(l, 'b')], k = k, k + layer.b.size
    n_in = [l.W.shape[1] for l in net]
    n_out = [l.W.shape[0] for l in net]
    rowW = lambda l, i: off[(l, 'W')] + i * n_in[l] + np.arange(n_in[l])
    colW = lambda l, j: off[(l, 'W')] + np.arange(n_out[l]) * n_in[l] + j
    ib = lambda l, i: off[(l, 'b')] + i
    unit_coords = lambda l, j: np.concatenate([rowW(l, j), [ib(l, j)], colW(l + 1, j)])

    # unit activity on Ω
    a, nact = omega.T, []
    for l in net[:-1]:
        z = (l.mask * l.W) @ a + l.b[:, None]
        nact.append((z > 0).sum(1))
        a = np.maximum(z, 0)
    dead = [np.flatnonzero(n == 0) for n in nact]
    always = [np.flatnonzero(n == B) for n in nact]

    G = pullback_metric(net, omega, np.ones(N, bool))
    lam, V = np.linalg.eigh(G)
    lam, V = lam[::-1], V[:, ::-1]
    lam_max = lam[0]
    n_null = int((lam < NULL_THR * lam_max).sum())
    band = np.flatnonzero((lam >= NULL_THR * lam_max) & (lam < BAND_THR * lam_max))

    # ---------------------------------------------------------- candidates
    fams = {}

    def add(fam, v):
        fams.setdefault(fam, []).append(v)

    for l in range(2):
        for j in range(n_out[l]):
            if j in dead[l]:
                continue
            v = np.zeros(N)
            v[rowW(l, j)] = theta[rowW(l, j)]
            v[ib(l, j)] = theta[ib(l, j)]
            v[colW(l + 1, j)] = -theta[colW(l + 1, j)]
            add('(i) rescaling (live hidden unit)', v)
    dead_idx = sorted(set(np.concatenate([unit_coords(l, j) for l in range(2) for j in dead[l]]).tolist()))
    for i in dead_idx:
        v = np.zeros(N)
        v[i] = 1
        add('(ii) dead-unit coordinate', v)
    v = np.zeros(N)
    v[off[(2, 'b')] + np.arange(n_out[2])] = 1
    add('(iii) logit shift', v)
    for l in range(2):
        for a_ in always[l]:
            for b_ in always[l]:
                v = np.zeros(N)
                v[rowW(l, a_)] += theta[rowW(l, b_)]
                v[ib(l, a_)] += theta[ib(l, b_)]
                v[colW(l + 1, b_)] -= theta[colW(l + 1, a_)]
                add(f'(iv) hidden{l + 1} always-on GL ' + ('diag E_aa (= rescaling)' if a_ == b_ else 'off-diag E_ab'), v)
            v = np.zeros(N)
            v[ib(l, a_)] = 1
            v[off[(l + 1, 'b')] + np.arange(n_out[l + 1])] = -theta[colW(l + 1, a_)]
            add(f'(iv) hidden{l + 1} always-on bias shift', v)
    for j in range(n_out[1]):  # (v) logit shift by h_j(x): W_2[:,j] += 1 (log-softmax invariance)
        if j not in dead[1]:
            v = np.zeros(N)
            v[colW(2, j)] = 1
            add('(v) per-unit logit shift W_2[:,j] += 1 (NEW, not in prop:kernel)', v)

    # ---------------------------------------------------------- residuals
    lines = ['# Kernel-basis check for ĝ(w⁰)\n',
             f'λ_max = {lam_max:.6e}; N = {N}; numerical null dim (λ/λ_max < 1e-15) = {n_null}; '
             f'dim below 1e-10 = {n_null + len(band)}; dead coords built = {len(dead_idx)}, '
             f'coords with ĝ_ii == 0: {int((np.diag(G) == 0).sum())}, sets equal: '
             f'{sorted(np.flatnonzero(np.diag(G) == 0).tolist()) == dead_idx}\n',
             '| family | #vectors | max residual | #with residual<1e-12 |', '|---|---|---|---|']
    res = {}
    for fam, vs in fams.items():
        C = np.array(vs).T
        r = np.linalg.norm(G @ C, axis=0) / (lam_max * np.linalg.norm(C, axis=0))
        res[fam] = (C, r)
        lines.append(f'| {fam} | {C.shape[1]} | {r.max():.2e} | {int((r < 1e-12).sum())} |'
                     + ('' if r.max() < RES_THR else '  **NOT a null direction**'))
    Q = V[:, lam < NULL_THR * lam_max]

    def leftover(prefixes):
        C = np.concatenate([C for f, (C, r) in res.items() if f.startswith(prefixes) and r.max() < RES_THR], axis=1)
        C = C / np.linalg.norm(C, axis=0)
        U, s, _ = np.linalg.svd(C, full_matrices=False)
        rank = int((s > 1e-10 * s[0]).sum())
        U = U[:, :rank]
        L, sp, _ = np.linalg.svd(Q - U @ (U.T @ Q), full_matrices=False)
        n_left = int((sp > 1e-8).sum())
        return C.shape[1], rank, s / s[0], L[:, :n_left], sp

    lines.append('\n## Rank of candidate families and unexplained remainder of the 305-dim numerical null space\n')
    lines.append('| families | #vectors | rank (s > 1e-10 s_max) | smallest kept / largest dropped s/s_max | unexplained (sp > 1e-8) | sp of leftovers (desc) |')
    lines.append('|---|---|---|---|---|---|')
    runs = {}
    for name, pre in [('(i)+(ii)+(iii)  [prop:kernel]', ('(i)', '(ii)', '(iii)')),
                      ('(i)-(iv)  [+ always-on hypothesis]', ('(i)', '(ii)', '(iii)', '(iv)')),
                      ('(i)-(v)  [+ per-unit logit shift]', ('(i)', '(ii)', '(iii)', '(iv)', '(v)'))]:
        nv, rank, s, L, sp = leftover(pre)
        runs[name] = L
        lines.append(f'| {name} | {nv} | {rank} | {s[rank - 1]:.1e} / {s[rank] if rank < len(s) else 0:.1e} | {L.shape[1]} | '
                     + ' '.join(f'{x:.1e}' for x in sp[:L.shape[1] + 2]) + ' |')
    L = runs['(i)-(iv)  [+ always-on hypothesis]']
    n_left = L.shape[1]
    sp_iv = leftover(('(i)', '(ii)', '(iii)', '(iv)'))[4]
    lines.append(f'\nresidual ‖ĝu‖/λ_max of the (i)-(iv) leftover vectors: max {np.linalg.norm(G @ L, axis=0).max() / lam_max if n_left else 0:.1e}')

    def label(i):
        l, kind, a_, b_ = imap[i]
        return f'L{l}.{kind}[{a_}' + (f',{b_}]' if kind == 'W' else ']')

    def unit_mass(v):
        m = {(l, j): float((v[unit_coords(l, j)] ** 2).sum()) for l in range(2) for j in range(n_out[l])}
        m[('out', 'b')] = float((v[off[(2, 'b')]:off[(2, 'b')] + n_out[2]] ** 2).sum())
        return sorted(m.items(), key=lambda kv: -kv[1])

    lines.append('\n## Unexplained null vectors after (i)-(iv): top-5 coordinates by |entry| (basis inside the leftover space is arbitrary; sp < 1e-2 = numerical residue)\n')
    for c in range(n_left):
        v = L[:, c]
        top = np.argsort(-np.abs(v))[:5]
        um = unit_mass(v)[:2]
        lines.append(f'- u{c} (sp={sp_iv[c]:.1e}): ' + ', '.join(f'{label(i)}={v[i]:+.3f}' for i in top)
                     + '  | unit mass: ' + ', '.join(f'{k}:{m:.2f}' for k, m in um))
    if n_left:
        P = np.diag(L @ L.T)
        bykind = {}
        for i, (l, kind, _, _) in enumerate(imap):
            bykind[(l, kind)] = bykind.get((l, kind), 0) + P[i]
        lines.append('\nbasis-invariant: trace of leftover projector by (layer, kind): '
                     + ', '.join(f'{k}: {m:.2f}' for k, m in bykind.items()))
        um = {(l, j): float(P[unit_coords(l, j)].sum()) for l in range(2) for j in range(n_out[l])}
        lines.append('trace of leftover projector on the coords incident to each hidden unit (layer, unit): '
                     + ', '.join(f'{k}: {m:.2f}' for k, m in sorted(um.items(), key=lambda kv: -kv[1])[:8]))

    lines.append(f'\n## Eigenvectors in the band 1e-15 ≤ λ/λ_max < 1e-10 (0-based indices {band.min()}–{band.max()} from the top)\n')
    lines.append('| index | λ/λ_max | residual ‖ĝv‖/λ_max | top units by mass (layer, unit): mass, #active samples |')
    lines.append('|---|---|---|---|')
    for idx in band:
        v = V[:, idx]
        um = unit_mass(v)[:3]
        desc = ', '.join(f'{k}: {m:.2f}' + (f' (active {nact[k[0]][k[1]]}/{B})' if k[0] != 'out' else '') for k, m in um)
        lines.append(f'| {idx} | {lam[idx] / lam_max:.2e} | {np.linalg.norm(G @ v) / lam_max:.2e} | {desc} |')

    lines.append('\n## Hidden-unit activity on Ω (#samples with pre-activation > 0, of 10000)\n')
    for l in range(2):
        lines.append(f'- hidden layer {l + 1}: ' + ' '.join(f'u{j}:{n}' for j, n in enumerate(nact[l]))
                     + f'  → dead {dead[l].tolist()}, always-on {always[l].tolist()}')
    txt = '\n'.join(lines) + '\n'
    print(txt)
    with open(OUT, 'w') as fh:
        fh.write(txt)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
