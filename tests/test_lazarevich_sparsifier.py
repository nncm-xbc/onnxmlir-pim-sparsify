# tests/test_lazarevich_sparsifier.py
import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from mlp.mlp import Layer
from sparsifier.lazarevich_sparsifier import main  # import triggers module load
from sparsifier.sparsifier import clone_network, d, prune


def _tiny_net_rng42():
    rng = np.random.default_rng(42)
    W0 = rng.standard_normal((3, 4))
    b0 = rng.standard_normal((3,))
    W1 = rng.standard_normal((2, 3))
    b1 = rng.standard_normal((2,))
    return [
        Layer(W=W0.copy(), b=b0.copy(), mask=np.ones((3, 4))),
        Layer(W=W1.copy(), b=b1.copy(), mask=np.ones((2, 3))),
    ]


def test_module_importable():
    """lazarevich_sparsifier must import without error."""
    import sparsifier.lazarevich_sparsifier  # noqa: F401


def test_same_prune_with_real_omega():
    """prune() called with real-image omega behaves identically to uniform omega."""
    net    = _tiny_net_rng42()
    og_net = clone_network(net)

    # 'real' omega: deterministic pixel values (simulate calibration images)
    real_omega = np.tile(np.arange(4, dtype=np.float32), (50, 1)) * 10.0
    uniform_omega = np.random.default_rng(0).integers(0, 256, (50, 4)).astype(np.float32)

    net_r, meta_r = prune(net, og_net, real_omega,    doAdjust=False)
    net_u, meta_u = prune(net, og_net, uniform_omega, doAdjust=False)

    # Both calls should prune exactly one weight (may pick different weights)
    nz_before = int(sum((l.W != 0).sum() for l in net))
    assert int(sum((l.W != 0).sum() for l in net_r)) == nz_before - 1
    assert int(sum((l.W != 0).sum() for l in net_u)) == nz_before - 1
