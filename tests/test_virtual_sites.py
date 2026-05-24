import mlx.core as mx
import numpy as np
import pytest

from mlx_atomistic.artifacts import (
    MLXCompatibilityError,
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
)
from mlx_atomistic.md import LangevinThermostat, SimulationConfig, simulate_npt, simulate_nvt
from mlx_atomistic.prep.io import load_prepared_system, save_prepared_system
from mlx_atomistic.prep.schema import (
    ARTIFACT_VERSION,
    PreparedSystem,
    PreparedSystemMetadata,
    empty_indices,
)
from mlx_atomistic.topology import Topology
from mlx_atomistic.virtual_sites import (
    TIP4P_EW_HOH_ANGLE_DEGREES,
    TIP4P_EW_OH_DISTANCE_ANGSTROM,
    TIP4P_EW_OM_DISTANCE_ANGSTROM,
    LocalCoordinates,
    OutOfPlane,
    ThreeParticleAverage,
    TwoParticleAverage,
    VirtualSiteManager,
    compute_virtual_site_positions,
    tip4p_ew_reference_positions,
    tip4p_ew_virtual_site,
)


class TestTwoParticleAverage:
    def test_midpoint(self):
        site = TwoParticleAverage(particle1=0, particle2=1, weight1=0.5, weight2=0.5)
        positions = mx.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=mx.float32
        )
        result = site.compute_position(positions)
        expected = mx.array([1.0, 0.0, 0.0], dtype=mx.float32)
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-6)

    def test_weighted_average(self):
        site = TwoParticleAverage(particle1=0, particle2=1, weight1=0.3, weight2=0.7)
        positions = mx.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=mx.float32
        )
        result = site.compute_position(positions)
        expected = mx.array([7.0, 0.0, 0.0], dtype=mx.float32)
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-6)

    def test_parent_atoms(self):
        site = TwoParticleAverage(particle1=2, particle2=5, weight1=1.0, weight2=0.0)
        assert site.parent_atoms == (2, 5)

    def test_rejects_negative_indices(self):
        with pytest.raises(ValueError, match="non-negative"):
            TwoParticleAverage(particle1=-1, particle2=1, weight1=0.5, weight2=0.5)

    def test_rejects_same_particles(self):
        with pytest.raises(ValueError, match="distinct"):
            TwoParticleAverage(particle1=0, particle2=0, weight1=0.5, weight2=0.5)


class TestThreeParticleAverage:
    def test_equal_weights(self):
        site = ThreeParticleAverage(
            particle1=0, particle2=1, particle3=2,
            weight1=1.0 / 3, weight2=1.0 / 3, weight3=1.0 / 3,
        )
        positions = mx.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            dtype=mx.float32,
        )
        result = site.compute_position(positions)
        expected = mx.array([1.0, 1.0, 0.0], dtype=mx.float32)
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-6)

    def test_parent_atoms(self):
        site = ThreeParticleAverage(
            particle1=0, particle2=1, particle3=2,
            weight1=1.0, weight2=0.0, weight3=0.0,
        )
        assert site.parent_atoms == (0, 1, 2)

    def test_rejects_non_distinct(self):
        with pytest.raises(ValueError, match="distinct"):
            ThreeParticleAverage(
                particle1=0, particle2=0, particle3=1,
                weight1=0.5, weight2=0.3, weight3=0.2,
            )


class TestOutOfPlane:
    def test_planar_molecule(self):
        positions = mx.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=mx.float32,
        )
        site = OutOfPlane(
            particle1=0, particle2=1, particle3=2,
            weight1=0.0, weight2=0.0, weight3=0.0,
            distance=1.0,
        )
        result = site.compute_position(positions)
        assert abs(float(result[2])) > 0.5

    def test_parent_atoms(self):
        site = OutOfPlane(
            particle1=0, particle2=1, particle3=2,
            weight1=0.0, weight2=0.0, weight3=0.0,
            distance=0.5,
        )
        assert site.parent_atoms == (0, 1, 2)


class TestLocalCoordinates:
    def test_zero_offsets_at_origin(self):
        positions = mx.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=mx.float32,
        )
        site = LocalCoordinates(
            particle1=0, particle2=1, particle3=2,
            weight1=1.0, weight2=0.0, weight3=0.0,
            local_x=0.0, local_y=0.0, local_z=0.0,
        )
        result = site.compute_position(positions)
        np.testing.assert_allclose(np.asarray(result), [0.0, 0.0, 0.0], atol=1e-6)

    def test_x_offset_along_bond(self):
        positions = mx.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=mx.float32,
        )
        site = LocalCoordinates(
            particle1=0, particle2=1, particle3=2,
            weight1=1.0, weight2=0.0, weight3=0.0,
            local_x=0.5, local_y=0.0, local_z=0.0,
        )
        result = site.compute_position(positions)
        np.testing.assert_allclose(np.asarray(result)[0], 0.5, atol=1e-6)

    def test_z_offset_perpendicular(self):
        positions = mx.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=mx.float32,
        )
        site = LocalCoordinates(
            particle1=0, particle2=1, particle3=2,
            weight1=1.0, weight2=0.0, weight3=0.0,
            local_x=0.0, local_y=0.0, local_z=1.0,
        )
        result = site.compute_position(positions)
        assert abs(float(result[2])) > 0.9


class TestComputeVirtualSitePositions:
    def test_empty(self):
        result = compute_virtual_site_positions((), mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32))
        assert np.asarray(result).shape == (0, 3)

    def test_multiple_sites(self):
        positions = mx.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=mx.float32,
        )
        sites = (
            TwoParticleAverage(particle1=0, particle2=1, weight1=0.5, weight2=0.5),
            ThreeParticleAverage(
                particle1=0, particle2=1, particle3=2,
                weight1=1.0 / 3, weight2=1.0 / 3, weight3=1.0 / 3,
            ),
        )
        result = compute_virtual_site_positions(sites, positions)
        assert np.asarray(result).shape == (2, 3)
        np.testing.assert_allclose(np.asarray(result[0]), [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(result[1]), [2.0 / 3, 2.0 / 3, 0.0], atol=1e-5
        )


class TestVirtualSiteForceRedistribution:
    def test_linear_site_force_redistributes_to_parent_atoms(self):
        site = TwoParticleAverage(particle1=0, particle2=1, weight1=0.25, weight2=0.75)
        manager = VirtualSiteManager((site,), n_real_atoms=2)
        positions = mx.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=mx.float32,
        )
        extended_positions = manager.extend_positions(positions)
        forces = mx.array(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-4.0, 8.0, 2.0]],
            dtype=mx.float32,
        )

        redistributed = manager.redistribute_forces(forces, extended_positions)

        expected = np.asarray(
            [[0.0, 2.0, 0.5], [-3.0, 8.0, 1.5]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(np.asarray(redistributed), expected, atol=1e-6)


class _VirtualQuadraticForce:
    name = "virtual_quadratic"
    supports_virial = True

    def __init__(self, target):
        self.target = mx.array(target, dtype=mx.float32)

    def energy_forces(self, positions, cell=None, pairs=None):
        del cell, pairs
        displacement = positions[2] - self.target
        energy = 0.5 * mx.sum(displacement * displacement)
        forces = mx.zeros_like(positions).at[2].add(-displacement)
        return energy, forces


class _VirtualXEnergy:
    name = "virtual_x"
    supports_virial = True

    def energy_forces(self, positions, cell=None, pairs=None):
        del cell, pairs
        return positions[2, 0], mx.zeros_like(positions)


def _tip4p_prepared_system() -> PreparedSystem:
    positions = tip4p_ew_reference_positions()
    metadata = PreparedSystemMetadata(
        artifact_version=ARTIFACT_VERSION,
        source={"kind": "test"},
        selections={"atom_count": 4, "hydrogen_count": 2, "water_atom_count": 4},
        units={
            "coordinates": "angstrom",
            "mass": "dalton",
            "charge": "elementary_charge",
            "energy": "kilojoule_per_mole",
            "time": "picosecond",
            "temperature": "kelvin",
        },
        parameter_source="tip4p_ew_test",
        compatibility_report={
            "engine": "mlx_atomistic",
            "production_force_field": True,
            "hydrogens_present": True,
            "hydrogen_count": 2,
            "water_present": True,
            "supported_terms": ["virtual_site", "nonbonded_lj_coulomb"],
            "required_terms": ["virtual_site", "nonbonded_lj_coulomb"],
            "unsupported_terms": [],
            "rejected_terms": [],
            "virtual_sites_present": True,
            "water_model": "tip4p_ew",
            "term_counts": {"virtual_site": 1},
        },
    )
    return PreparedSystem(
        metadata=metadata,
        symbols=np.asarray(["O", "H", "H", "M"], dtype=str),
        atom_names=np.asarray(["O", "H1", "H2", "M"], dtype=str),
        atom_types=np.asarray(["OW", "HW", "HW", "MW"], dtype=str),
        residue_names=np.asarray(["WAT", "WAT", "WAT", "WAT"], dtype=str),
        residue_ids=np.asarray([1, 1, 1, 1], dtype=np.int32),
        chain_ids=np.asarray(["A", "A", "A", "A"], dtype=str),
        positions=positions,
        velocities=np.zeros_like(positions, dtype=np.float32),
        masses=np.asarray([15.999, 1.008, 1.008, 0.0], dtype=np.float32),
        charges=np.asarray([0.0, 0.52422, 0.52422, -1.04844], dtype=np.float32),
        sigma=np.asarray([3.16435, 1.0, 1.0, 1.0], dtype=np.float32),
        epsilon=np.asarray([0.680946, 0.0, 0.0, 0.0], dtype=np.float32),
        bonds=empty_indices(2),
        bond_k=np.asarray([], dtype=np.float32),
        bond_length=np.asarray([], dtype=np.float32),
        angles=empty_indices(3),
        angle_k=np.asarray([], dtype=np.float32),
        angle_theta=np.asarray([], dtype=np.float32),
        dihedrals=empty_indices(4),
        dihedral_k=np.asarray([], dtype=np.float32),
        dihedral_periodicity=np.asarray([], dtype=np.float32),
        dihedral_phase=np.asarray([], dtype=np.float32),
        nonbonded_pairs=empty_indices(2),
        ligand_mask=np.zeros((4,), dtype=bool),
        receptor_mask=np.zeros((4,), dtype=bool),
        restraint_mask=np.zeros((4,), dtype=bool),
        reference_positions=positions.copy(),
        virtual_site_parent_atoms=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        virtual_site_weights=np.asarray(
            [[0.78664654, 0.10667673, 0.10667673, 0.0]],
            dtype=np.float32,
        ),
        virtual_site_types=np.asarray(["tip4p_ew"], dtype=str),
    )


class TestVirtualSiteMDIntegration:
    def test_simulate_nvt_redistributes_virtual_site_forces_to_real_atoms(self):
        site = TwoParticleAverage(particle1=0, particle2=1, weight1=0.25, weight2=0.75)
        manager = VirtualSiteManager((site,), n_real_atoms=2)
        config = SimulationConfig(steps=0, pressure_diagnostics=False, virtual_sites=manager)
        positions = mx.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=mx.float32)
        velocities = mx.zeros_like(positions)

        result = simulate_nvt(
            positions,
            velocities,
            force_terms=_VirtualQuadraticForce([1.0, -2.0, 0.5]),
            config=config,
            thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=1),
        )

        np.testing.assert_allclose(
            np.asarray(result.final_state.forces),
            [[-0.125, -0.5, 0.125], [-0.375, -1.5, 0.375]],
            atol=1e-6,
        )
        assert np.asarray(result.sampled_positions).shape == (1, 2, 3)

    def test_simulate_nvt_reconstructs_virtual_site_positions_each_timestep(self):
        site = TwoParticleAverage(particle1=0, particle2=1, weight1=0.5, weight2=0.5)
        manager = VirtualSiteManager((site,), n_real_atoms=2)
        config = SimulationConfig(
            dt=0.1,
            steps=1,
            diagnostic_interval=1,
            pressure_diagnostics=False,
            virtual_sites=manager,
        )
        positions = mx.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=mx.float32)
        velocities = mx.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=mx.float32)

        result = simulate_nvt(
            positions,
            velocities,
            force_terms=_VirtualXEnergy(),
            config=config,
            thermostat=LangevinThermostat(temperature=0.0, friction=0.0, seed=1),
        )

        np.testing.assert_allclose(np.asarray(result.potential_energy[-1]), 1.2, atol=1e-6)
        assert np.asarray(result.final_state.positions).shape == (2, 3)

    def test_simulate_nvt_and_npt_signatures_do_not_expose_virtual_sites(self):
        import inspect

        assert "virtual_sites" not in inspect.signature(simulate_nvt).parameters
        assert "virtual_sites" not in inspect.signature(simulate_npt).parameters


class TestTip4pLikeConfiguration:
    """Test virtual-site position reconstruction for a TIP4P-like water model.

    TIP4P places the negative charge site M along the bisector of the H-O-H
    angle, displaced from the oxygen toward the hydrogens. The M site is
    modeled as a ThreeParticleAverage with weights chosen so that the site
    falls at a known position along the bisector.
    """

    def test_tip4p_m_site_on_bisector(self):
        oh_distance = 0.9572
        hoh_angle_deg = 104.52

        hoh_angle_rad = np.radians(hoh_angle_deg)
        half_angle = hoh_angle_rad / 2.0

        oxygen_pos = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        h1_pos = np.asarray(
            [oh_distance * np.cos(half_angle), oh_distance * np.sin(half_angle), 0.0],
            dtype=np.float32,
        )
        h2_pos = np.asarray(
            [oh_distance * np.cos(half_angle), -oh_distance * np.sin(half_angle), 0.0],
            dtype=np.float32,
        )

        positions = mx.array(np.stack([oxygen_pos, h1_pos, h2_pos]), dtype=mx.float32)

        m_site = ThreeParticleAverage(
            particle1=0, particle2=1, particle3=2,
            weight1=0.7869, weight2=0.1066, weight3=0.1066,
        )
        m_pos = np.asarray(m_site.compute_position(positions), dtype=np.float64)

        bisector = (h1_pos + h2_pos) / 2.0 - oxygen_pos
        bisector_unit = bisector / np.linalg.norm(bisector)

        m_vec = m_pos - np.asarray(oxygen_pos, dtype=np.float64)
        m_along = np.dot(m_vec, bisector_unit)
        m_perp = m_vec - m_along * bisector_unit
        assert np.linalg.norm(m_perp) < 1e-5

        assert m_along > 0

    def test_tip4p_m_site_exact_weights(self):
        oh_distance = 0.9572
        hoh_angle_deg = 104.52
        d_om = 0.15

        hoh_angle_rad = np.radians(hoh_angle_deg)
        half_angle = hoh_angle_rad / 2.0

        oxygen_pos = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        h1_pos = np.asarray(
            [oh_distance * np.cos(half_angle), oh_distance * np.sin(half_angle), 0.0],
            dtype=np.float32,
        )
        h2_pos = np.asarray(
            [oh_distance * np.cos(half_angle), -oh_distance * np.sin(half_angle), 0.0],
            dtype=np.float32,
        )

        bisector_dir = (h1_pos + h2_pos) / 2.0 - oxygen_pos
        bisector_unit = bisector_dir / np.linalg.norm(bisector_dir)
        target_m = oxygen_pos + d_om * bisector_unit

        w2 = w3 = d_om / (2.0 * oh_distance * np.cos(half_angle))
        w1 = 1.0 - 2.0 * w2

        m_site = ThreeParticleAverage(
            particle1=0, particle2=1, particle3=2,
            weight1=float(w1), weight2=float(w2), weight3=float(w3),
        )
        positions = mx.array(np.stack([oxygen_pos, h1_pos, h2_pos]), dtype=mx.float32)
        m_pos = np.asarray(m_site.compute_position(positions), dtype=np.float32)
        np.testing.assert_allclose(m_pos, target_m, atol=1e-5)

    def test_tip4p_ew_factory_matches_model_geometry(self):
        positions = tip4p_ew_reference_positions()
        site = tip4p_ew_virtual_site(0, 1, 2)

        oh1 = np.linalg.norm(positions[1] - positions[0])
        oh2 = np.linalg.norm(positions[2] - positions[0])
        angle = np.degrees(
            np.arccos(
                np.dot(positions[1] - positions[0], positions[2] - positions[0])
                / (oh1 * oh2)
            )
        )
        m_pos = np.asarray(site.compute_position(mx.array(positions[:3])), dtype=np.float32)

        np.testing.assert_allclose(oh1, TIP4P_EW_OH_DISTANCE_ANGSTROM, atol=1e-6)
        np.testing.assert_allclose(oh2, TIP4P_EW_OH_DISTANCE_ANGSTROM, atol=1e-6)
        np.testing.assert_allclose(angle, TIP4P_EW_HOH_ANGLE_DEGREES, atol=1e-5)
        np.testing.assert_allclose(
            np.linalg.norm(m_pos - positions[0]),
            TIP4P_EW_OM_DISTANCE_ANGSTROM,
            atol=1e-6,
        )
        np.testing.assert_allclose(m_pos, positions[3], atol=1e-6)

    def test_tip4p_virtual_site_arrays_survive_prepared_system_round_trip(self, tmp_path):
        positions = tip4p_ew_reference_positions()
        parent_atoms = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
        weights = np.asarray([[0.78664654, 0.10667673, 0.10667673, 0.0]], dtype=np.float32)
        site_types = np.asarray(["tip4p_ew"], dtype=str)
        metadata = PreparedSystemMetadata(
            artifact_version=ARTIFACT_VERSION,
            source={"kind": "test"},
            selections={"atom_count": 4, "hydrogen_count": 2},
            units={
                "coordinates": "angstrom",
                "mass": "dalton",
                "charge": "elementary_charge",
                "energy": "kilojoule_per_mole",
                "time": "picosecond",
                "temperature": "kelvin",
            },
            parameter_source="tip4p_ew_test",
            compatibility_report={
                "engine": "mlx_atomistic",
                "supported_terms": ["virtual_site"],
                "required_terms": ["virtual_site"],
                "unsupported_terms": [],
                "rejected_terms": [],
                "virtual_sites_present": True,
                "water_model": "tip4p_ew",
            },
        )
        prepared = PreparedSystem(
            metadata=metadata,
            symbols=np.asarray(["O", "H", "H", "M"], dtype=str),
            atom_names=np.asarray(["O", "H1", "H2", "M"], dtype=str),
            atom_types=np.asarray(["OW", "HW", "HW", "MW"], dtype=str),
            residue_names=np.asarray(["WAT", "WAT", "WAT", "WAT"], dtype=str),
            residue_ids=np.asarray([1, 1, 1, 1], dtype=np.int32),
            chain_ids=np.asarray(["A", "A", "A", "A"], dtype=str),
            positions=positions,
            velocities=np.zeros_like(positions, dtype=np.float32),
            masses=np.asarray([15.999, 1.008, 1.008, 0.0], dtype=np.float32),
            charges=np.asarray([0.0, 0.52422, 0.52422, -1.04844], dtype=np.float32),
            sigma=np.asarray([3.16435, 1.0, 1.0, 1.0], dtype=np.float32),
            epsilon=np.asarray([0.680946, 0.0, 0.0, 0.0], dtype=np.float32),
            bonds=empty_indices(2),
            bond_k=np.asarray([], dtype=np.float32),
            bond_length=np.asarray([], dtype=np.float32),
            angles=empty_indices(3),
            angle_k=np.asarray([], dtype=np.float32),
            angle_theta=np.asarray([], dtype=np.float32),
            dihedrals=empty_indices(4),
            dihedral_k=np.asarray([], dtype=np.float32),
            dihedral_periodicity=np.asarray([], dtype=np.float32),
            dihedral_phase=np.asarray([], dtype=np.float32),
            nonbonded_pairs=empty_indices(2),
            ligand_mask=np.zeros((4,), dtype=bool),
            receptor_mask=np.zeros((4,), dtype=bool),
            restraint_mask=np.zeros((4,), dtype=bool),
            reference_positions=positions.copy(),
            virtual_site_parent_atoms=parent_atoms,
            virtual_site_weights=weights,
            virtual_site_types=site_types,
        )

        save_prepared_system(prepared, tmp_path)
        reloaded = load_prepared_system(tmp_path)

        np.testing.assert_array_equal(reloaded.virtual_site_parent_atoms, parent_atoms)
        np.testing.assert_allclose(reloaded.virtual_site_weights, weights, atol=1e-7)
        np.testing.assert_array_equal(reloaded.virtual_site_types, site_types)

    def test_tip4p_artifact_load_validates_virtual_site_arrays(self, tmp_path):
        prepared = _tip4p_prepared_system()
        broken = PreparedSystem(
            **{
                **prepared.__dict__,
                "virtual_site_parent_atoms": empty_indices(4),
                "virtual_site_weights": np.empty((0, 4), dtype=np.float32),
                "virtual_site_types": np.asarray([], dtype=str),
            }
        )

        save_prepared_system(broken, tmp_path)

        with pytest.raises(MLXCompatibilityError, match="at least one virtual site"):
            load_prepared_mlx_artifact(tmp_path, require_production=True)

    def test_tip4p_artifact_build_exposes_real_atoms_and_virtual_site_manager(self, tmp_path):
        prepared = _tip4p_prepared_system()
        save_prepared_system(prepared, tmp_path)

        artifact = load_prepared_mlx_artifact(tmp_path, require_production=True)
        system, _, _ = build_mlx_system_from_artifact(artifact)

        assert system.atom_count == 3
        assert np.asarray(system.positions).shape == (3, 3)
        assert np.asarray(system.masses).shape == (3,)
        assert system.virtual_sites is not None
        assert system.virtual_sites.n_real_atoms == 3
        assert system.virtual_sites.n_virtual_sites == 1

    def test_runner_propagates_artifact_virtual_sites_to_simulation_config(self, monkeypatch):
        from mlx_atomistic.prep import runner

        captured = {}

        def fake_simulate_nvt(*args, **kwargs):
            captured["config"] = kwargs["config"]
            return object()

        monkeypatch.setattr(runner, "simulate_nvt", fake_simulate_nvt)

        runner.run_mlx(
            _tip4p_prepared_system(),
            steps=0,
            minimize_steps=0,
            equilibration_steps=0,
            require_production=True,
        )

        assert captured["config"].virtual_sites is not None
        assert captured["config"].virtual_sites.n_virtual_sites == 1


class TestTopologyVirtualSites:
    def test_default_virtual_sites(self):
        topology = Topology.from_sequences(n_atoms=3)
        assert topology.virtual_sites == ()
        assert topology.virtual_site_types == ()

    def test_virtual_sites_stored(self):
        site = TwoParticleAverage(particle1=0, particle2=1, weight1=0.5, weight2=0.5)
        topology = Topology.from_sequences(
            n_atoms=3,
            virtual_sites=[site],
            virtual_site_types=["two_particle_average"],
        )
        assert len(topology.virtual_sites) == 1
        assert topology.virtual_site_types == ("two_particle_average",)

    def test_virtual_site_types_length_mismatch(self):
        site = TwoParticleAverage(particle1=0, particle2=1, weight1=0.5, weight2=0.5)
        with pytest.raises(ValueError, match="same length"):
            Topology.from_sequences(
                n_atoms=3,
                virtual_sites=[site],
                virtual_site_types=["a", "b"],
            )

    def test_backward_compatible_no_virtual_sites(self):
        topology = Topology(n_atoms=3, bonds=[(0, 1)])
        assert topology.virtual_sites == ()
        assert topology.virtual_site_types == ()
