import numpy as np
import pytest

from mlx_atomistic.topology import Topology


def test_topology_validates_index_shapes():
    with pytest.raises(ValueError, match="bonds"):
        Topology.from_sequences(n_atoms=2, bonds=[(0, 1, 2)])
    with pytest.raises(ValueError, match="angles"):
        Topology.from_sequences(n_atoms=3, angles=[(0, 1)])
    with pytest.raises(ValueError, match="dihedrals"):
        Topology.from_sequences(n_atoms=4, dihedrals=[(0, 1, 2)])


def test_topology_rejects_out_of_range_indices():
    with pytest.raises(ValueError, match="outside"):
        Topology.from_sequences(n_atoms=2, bonds=[(0, 2)])


def test_topology_normalizes_duplicate_exclusions():
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1)],
        exclusions=[(2, 3), (3, 2), (0, 1)],
    )

    assert np.array(topology.exclusions).tolist() == [[0, 1], [2, 3]]
    assert np.array(topology.nonbonded_pairs()).tolist() == [[0, 2], [0, 3], [1, 2], [1, 3]]


def test_topology_one_four_pairs_default_to_dihedral_endpoints():
    topology = Topology.from_sequences(
        n_atoms=4,
        bonds=[(0, 1), (1, 2), (2, 3)],
        dihedrals=[(0, 1, 2, 3)],
    )
    pairs = topology.nonbonded_pairs()
    scales = topology.pair_scales(pairs, one_four_scale=0.5)

    assert [0, 3] in np.array(pairs).tolist()
    assert np.array(scales)[np.array(pairs).tolist().index([0, 3])] == 0.5

