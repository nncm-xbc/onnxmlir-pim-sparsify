"""Lazarevich et al. 2021 baseline — data-driven Omega.

Identical to the manifold sparsifier (sparsifier.sparsifier) except that the
Monte Carlo sample Omega is drawn from the *real* test-image distribution
instead of uniform pixel noise.  Any difference in the sparsity/accuracy
trajectory vs the manifold baseline is attributable solely to the choice of
Omega distribution (real MNIST images vs uniform noise).

Public API: :func:`main` (entry point only — reuses prune/adjust from
sparsifier.sparsifier directly, via the shared run_sparsifier driver with
omega_source="data").
"""

from sparsifier.sparsifier import prune


def main():
    import sys

    from sparsifier.runner import run_sparsifier
    run_sparsifier(
        sys.argv[1],
        lambda net, og, om, doAdjust, activations: prune(net, og, om, activations=activations, doAdjust=doAdjust),
        output_subdir='lazarevich_sparsified',
        log_name='lazarevich_sparsification_log.csv',
        omega_source='data',
        loop_label='Lazarevich sparsification',
    )


if __name__ == '__main__':
    main()
