# mlp/dataset.py

from typing import NamedTuple
import numpy as np

from mlp.mlp import one_hot


class DatasetStats(NamedTuple):
    input_min:  float
    input_max:  float
    n_classes:  int
    n_features: int


def load_dataset(path: str, source: str = "csv"):
    """Load a dataset. Returns ((x_train, y_train), (x_test, y_test), DatasetStats).

    source="csv" — path is a folder with X_train.csv, Y_train.csv,
                   X_test.csv, Y_test.csv
    """
    if source == "csv":
        return _load_csv(path)
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'csv'.")


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
        y_train = one_hot(y_train.astype(int), n_classes)
        y_test  = one_hot(y_test.astype(int),  n_classes)
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
