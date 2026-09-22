# prep_dataset.py — build data/<name>/{X,Y}_{train,test}_small.csv for a new dataset
# using the exact pipeline that produced the MNIST CSVs in data/:
#   raw pixels as float in [0,255] -> PIL mode 'F' bicubic resize to 14x14
#   (hence values slightly outside [0,255], same as MNIST) -> 196 features;
#   labels one-hot over 10 classes; 20000 train / 10000 test rows (the MNIST
#   CSVs are Colab's 20k/10k sample, train.py reads the first 10k/1k).
#
# usage: python scripts/prep_dataset.py {fashion,kmnist,usps,s1_teacher,s2_planted}
# See docs/dataset-research.md for why these datasets.

import os
import sys

import numpy as np
from PIL import Image

_repo = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
N_TRAIN, N_TEST, SIDE = 20000, 10000, 14


def _resize(imgs):
    return np.stack([np.array(Image.fromarray(im.astype(np.float64)).resize((SIDE, SIDE)))
                     for im in imgs]).reshape(len(imgs), -1)


def _torchvision(cls):
    root = os.path.join(_repo, 'data', '_downloads')
    out = []
    for train in (True, False):
        ds = cls(root, train=train, download=True)
        x = np.asarray(ds.data, dtype=np.float64)
        y = np.asarray(ds.targets)
        if x.max() <= 1.0:  # USPS ships [0,1]? keep the MNIST [0,255] scale
            x = x * 255.0
        out += [x, y]
    return out


def _px(x):
    """Standard-normal features -> pixel range: +-3 sigma maps onto [0,255], same as noise-Omega."""
    return np.clip(127.5 + 42.5 * x, 0, 255)


def _s1_teacher():
    """S1, structureless null: x ~ N(0, I_196), labels from a random [196,10,10,10]
    ReLU teacher (He-normal, zero bias, seed 1) with logits standardised per column
    over the train set (balances classes, still realisable). Data seed 0.
    Teacher weights saved as teacher.npz for student-teacher alignment."""
    d = SIDE * SIDE
    x = np.random.default_rng(0).standard_normal((N_TRAIN + N_TEST, d))
    t = np.random.default_rng(1)
    Ws = [t.standard_normal((o, i)) * np.sqrt(2.0 / i) for i, o in zip([d, 10, 10], [10, 10, 10])]
    z = x
    for W in Ws[:-1]:
        z = np.maximum(z @ W.T, 0)
    z = z @ Ws[-1].T
    # Per-column standardisation alone leaves classes 5-35% (checked), so also fit
    # per-class logit offsets on the train split until argmax is ~uniform. Both are
    # an affine map on the last layer, so the task stays realisable by [196,10,10,10].
    mu, sd = z[:N_TRAIN].mean(0), z[:N_TRAIN].std(0)
    z = (z - mu) / sd
    off = np.zeros(10)
    for _ in range(2000):
        freq = np.bincount(np.argmax(z[:N_TRAIN] + off, 1), minlength=10) / N_TRAIN
        off -= 0.05 * (freq - 0.1) / 0.1
    y = np.argmax(z + off, axis=1)
    extra = {'teacher.npz': dict(W_0=Ws[0], W_1=Ws[1], W_2=Ws[2], logit_mu=mu, logit_sd=sd, logit_offset=off)}
    x = _px(x)
    return x[:N_TRAIN], y[:N_TRAIN], x[N_TRAIN:], y[N_TRAIN:], extra


def _s2_planted():
    """S2, planted non-spatial structure: make_classification with 20 informative of 196
    features, scattered to random pixels by a saved permutation."""
    from sklearn.datasets import make_classification
    x, y = make_classification(n_samples=N_TRAIN + N_TEST, n_features=SIDE * SIDE, n_informative=20,
                               n_redundant=0, n_repeated=0, n_classes=10, n_clusters_per_class=1,
                               class_sep=1.5, flip_y=0.0, shuffle=False, random_state=0)
    rows = np.random.default_rng(1).permutation(len(x))  # shuffle=False also sorts rows by class
    x, y = x[rows], y[rows]
    perm = np.random.default_rng(0).permutation(SIDE * SIDE)
    x = x[:, perm]  # column j of the output = original feature perm[j]
    x = _px((x - x.mean(0)) / x.std(0))
    informative_px = np.nonzero(perm < 20)[0]
    extra = {'informative_pixels.npy': informative_px}
    return x[:N_TRAIN], y[:N_TRAIN], x[N_TRAIN:], y[N_TRAIN:], extra


def build(name):
    extra = {}
    if name in ('s1_teacher', 's2_planted'):
        xtr, ytr, xte, yte, extra = (_s1_teacher if name == 's1_teacher' else _s2_planted)()
    else:
        import torchvision.datasets as tvd
        cls = {'fashion': tvd.FashionMNIST, 'kmnist': tvd.KMNIST, 'usps': tvd.USPS}[name]
        xtr, ytr, xte, yte = _torchvision(cls)
        rng = np.random.default_rng(0)  # fixed subsample, class mix as in the source
        itr = rng.permutation(len(xtr))[:N_TRAIN]
        ite = rng.permutation(len(xte))[:N_TEST]
        xtr, ytr, xte, yte = _resize(xtr[itr]), ytr[itr], _resize(xte[ite]), yte[ite]

    onehot = lambda y: np.eye(10)[y.astype(int)]
    out = os.path.join(_repo, 'data', name)
    os.makedirs(out, exist_ok=True)
    for fn, arr in [('X_train_small', xtr), ('Y_train_small', onehot(ytr)),
                    ('X_test_small', xte), ('Y_test_small', onehot(yte))]:
        # %.9g round-trips the float32 PIL output exactly and keeps files < 100 MB (GitHub limit)
        np.savetxt(os.path.join(out, fn + '.csv'), arr, delimiter=',', fmt='%.9g')
    for fn, obj in extra.items():
        (np.savez if fn.endswith('.npz') else np.save)(os.path.join(out, fn), **obj) if isinstance(obj, dict) \
            else np.save(os.path.join(out, fn), obj)
    print('%s: X_train %s X_test %s -> %s' % (name, xtr.shape, xte.shape, out))
    print('train class counts:', np.bincount(ytr.astype(int), minlength=10))


if __name__ == '__main__':
    build(sys.argv[1])
