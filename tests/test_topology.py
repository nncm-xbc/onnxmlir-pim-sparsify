# tests/test_topology.py
import os, tempfile, pytest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mlp.topology import load_topology, save_topology

def write_csv(tmp_path, content):
    p = tmp_path / "topology.csv"
    p.write_text(content)
    return str(p)

def test_load_single_row_defaults(tmp_path):
    """Single-row CSV: activations default to relu...relu, linear."""
    path = write_csv(tmp_path, "196,64,10\n")
    sizes, acts = load_topology(path)
    assert sizes == [196, 64, 10]
    assert acts == ["relu", "linear"]

def test_load_two_row(tmp_path):
    """Two-row CSV: activations are read from second row."""
    path = write_csv(tmp_path, "196,64,10\ntanh,linear\n")
    sizes, acts = load_topology(path)
    assert sizes == [196, 64, 10]
    assert acts == ["tanh", "linear"]

def test_load_mismatched_lengths_raises(tmp_path):
    """Activations row length must equal len(sizes)-1."""
    path = write_csv(tmp_path, "196,64,10\nrelu\n")
    with pytest.raises(ValueError, match="length"):
        load_topology(path)

def test_load_unknown_activation_raises(tmp_path):
    """Unknown activation names raise ValueError with helpful message."""
    path = write_csv(tmp_path, "196,64,10\nrelu,swish\n")
    with pytest.raises(ValueError, match="swish"):
        load_topology(path)

def test_save_load_roundtrip(tmp_path):
    """save_topology → load_topology is a round-trip."""
    path = str(tmp_path / "topo.csv")
    save_topology(path, [196, 64, 10], ["tanh", "linear"])
    sizes, acts = load_topology(path)
    assert sizes == [196, 64, 10]
    assert acts == ["tanh", "linear"]

def test_load_four_layer(tmp_path):
    """Four-layer topology with three activations."""
    path = write_csv(tmp_path, "196,10,10,10\nrelu,relu,linear\n")
    sizes, acts = load_topology(path)
    assert sizes == [196, 10, 10, 10]
    assert acts == ["relu", "relu", "linear"]
