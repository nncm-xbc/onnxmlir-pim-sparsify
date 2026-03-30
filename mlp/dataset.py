# mlp/dataset.py

from typing import NamedTuple
import numpy as np


class DatasetStats(NamedTuple):
    input_min:  float
    input_max:  float
    n_classes:  int
    n_features: int


def load_dataset(path: str, source: str = "csv"):
    """Load a dataset. Returns ((x_train, y_train), (x_test, y_test), DatasetStats).

    source="csv"     — path is a folder with X_train.csv, Y_train.csv,
                       X_test.csv, Y_test.csv
    source="pytorch" — path is a torchvision dataset name, e.g. "MNIST"
    """
    if source == "csv":
        return _load_csv(path)
    elif source == "pytorch":
        return _load_pytorch(path)
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'csv' or 'pytorch'.")


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    return (labels[:, None] == np.arange(n_classes)).astype(np.float32)


def _load_csv(folder: str):
    import os

    x_train = np.genfromtxt(os.path.join(folder, "X_train.csv"), delimiter=",")
    y_train = np.genfromtxt(os.path.join(folder, "Y_train.csv"), delimiter=",")
    x_test  = np.genfromtxt(os.path.join(folder, "X_test.csv"),  delimiter=",")
    y_test  = np.genfromtxt(os.path.join(folder, "Y_test.csv"),  delimiter=",")

    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)

    # Infer n_classes; convert label-encoded Y to one-hot if needed
    if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
        y_train = y_train.ravel()
        y_test  = y_test.ravel()
        n_classes = len(np.unique(y_train))
        y_train = _one_hot(y_train.astype(int), n_classes)
        y_test  = _one_hot(y_test.astype(int),  n_classes)
    else:
        n_classes = y_train.shape[1]
        y_train = y_train.astype(np.float32)
        y_test  = y_test.astype(np.float32)

    stats = DatasetStats(
        input_min  = float(x_train.min()),
        input_max  = float(x_train.max()),
        n_classes  = n_classes,
        n_features = x_train.shape[1],
    )
    return (x_train, y_train), (x_test, y_test), stats


def _load_pytorch(dataset_name: str):
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])

    dataset_cls = getattr(torchvision.datasets, dataset_name)
    train_ds = dataset_cls(root="/tmp/torchvision_data", train=True,  download=True, transform=transform)
    test_ds  = dataset_cls(root="/tmp/torchvision_data", train=False, download=True, transform=transform)

    def to_arrays(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
        X, y = next(iter(loader))
        X = X.numpy().reshape(len(ds), -1).astype(np.float32)
        return X, y.numpy()

    x_train, y_train_labels = to_arrays(train_ds)
    x_test,  y_test_labels  = to_arrays(test_ds)

    n_classes = len(np.unique(y_train_labels))
    y_train = _one_hot(y_train_labels, n_classes)
    y_test  = _one_hot(y_test_labels,  n_classes)

    stats = DatasetStats(
        input_min  = 0.0,
        input_max  = 1.0,
        n_classes  = n_classes,
        n_features = x_train.shape[1],
    )
    return (x_train, y_train), (x_test, y_test), stats
