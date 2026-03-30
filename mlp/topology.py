# mlp/topology.py

_VALID_ACTIVATIONS = {"relu", "tanh", "sigmoid", "linear"}


def load_topology(csv_path: str) -> tuple:
    """Load a topology CSV. Returns (sizes, activations).

    CSV format:
        Row 1: comma-separated actual layer sizes, e.g. 196,64,10
        Row 2: optional activations, one per weight matrix (len(sizes)-1 entries)
                e.g. relu,linear   for a 3-entry sizes row

    Defaults when row 2 is absent: relu for all hidden layers, linear for last.
    """
    rows = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([tok.strip() for tok in line.split(",")])

    if len(rows) == 0:
        raise ValueError(f"Topology CSV is empty or has no data rows: {csv_path}")

    try:
        sizes = [int(x) for x in rows[0]]
    except ValueError as e:
        for x in rows[0]:
            try:
                int(x)
            except ValueError:
                raise ValueError(f"Non-integer token '{x}' in sizes row of {csv_path}") from e
        raise
    n_layers = len(sizes) - 1

    if len(rows) >= 2:
        activations = rows[1]
        if len(activations) != n_layers:
            raise ValueError(
                f"Activations row length {len(activations)} does not match topology "
                f"weight matrices {n_layers} (len(sizes)-1). They must match."
            )
        for act in activations:
            if act not in _VALID_ACTIVATIONS:
                raise ValueError(
                    f"Unknown activation '{act}'. "
                    f"Supported: {sorted(_VALID_ACTIVATIONS)}"
                )
    else:
        activations = ["relu"] * (n_layers - 1) + ["linear"]

    return sizes, activations


def save_topology(csv_path: str, sizes: list, activations: list) -> None:
    """Write a two-row topology CSV.

    Validates that activations has the correct length (len(sizes) - 1) and
    that all activation names are in _VALID_ACTIVATIONS.
    """
    n_layers = len(sizes) - 1
    if len(activations) != n_layers:
        raise ValueError(
            f"Activations row length {len(activations)} does not match topology "
            f"weight matrices {n_layers} (len(sizes)-1). They must match."
        )
    for act in activations:
        if act not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{act}'. "
                f"Supported: {sorted(_VALID_ACTIVATIONS)}"
            )

    with open(csv_path, "w") as f:
        f.write(",".join(str(s) for s in sizes) + "\n")
        f.write(",".join(activations) + "\n")
