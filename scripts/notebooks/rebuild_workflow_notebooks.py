"""Build the curated workflow notebooks.

The old numbered notebooks are useful as milestone history, but the active
notebook surface should read like a compact lab manual.  This script writes the
current curated workflow notebooks with narrative Markdown, KaTeX equations,
and executable visualization cells.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "notebooks" / "workflows"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def notebook(cells: list[dict]) -> dict:
    normalized_cells = []
    for index, cell in enumerate(cells):
        normalized = dict(cell)
        normalized.setdefault("id", f"cell-{index:02d}")
        normalized_cells.append(normalized)
    return {
        "cells": normalized_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(name: str, cells: list[dict]) -> None:
    path = NOTEBOOK_DIR / name
    path.write_text(json.dumps(notebook(cells), indent=2) + "\n")


SETUP_CODE = r"""
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np


def find_repo_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("could not locate repository root")


ROOT = find_repo_root()
"""


def md_core() -> list[dict]:
    return [
        markdown(
            r"""
# Molecular Mechanics Core: topology → force terms → trajectory

This notebook is the compact MD entry point.  It uses a water-like toy molecule
to connect the theory to the package API:

$$
E_\mathrm{MM}
= E_\mathrm{bond}
+ E_\mathrm{angle}
+ E_\mathrm{dihedral}
+ E_\mathrm{LJ}
+ E_\mathrm{Coulomb}.
$$

The point is not chemical realism.  The point is to verify the data model:
typed atoms, topology, bonded terms, nonbonded exclusions, constraints, and
per-term energy diagnostics.
"""
        ),
        code(SETUP_CODE),
        markdown(
            r"""
## Reduced-unit Hamiltonian used here

The MD layer uses reduced internal units for these examples.  That means the
time, mass, length, charge, and energy scales are chosen by the model instead
of SI constants.  In this convention the simulated Hamiltonian is simply

$$
H(r,p) = K(p) + U(r),
\qquad
K(p) = \sum_i \frac{|p_i|^2}{2m_i}
      = \frac{1}{2}\sum_i m_i |v_i|^2.
$$

The force on atom \(i\) is the negative gradient of the potential energy:

$$
F_i = -\nabla_i U(r).
$$

For a correct NVE integrator, \(H\) should remain nearly constant.  Small
oscillatory drift is expected from finite timestep integration; systematic
drift usually means the timestep is too large, constraints are loose, or the
force implementation is wrong.
"""
        ),
        markdown(
            r"""
## Build a typed system

The topology says which atoms are bonded and which angle is meaningful.  The
force field maps atom types to parameters.  Bonded pairs are excluded from the
nonbonded pair list, while the H–H pair remains active.
"""
        ),
        markdown(
            r"""
### Force-field terms in this toy system

The water-like system below uses two O–H harmonic bonds and one H–O–H harmonic
angle:

$$
E_\mathrm{bond}
= \frac{1}{2}\sum_{(ij)}
k_{ij}\left(r_{ij}-r_{ij}^{0}\right)^2,
\qquad
r_{ij}=|r_i-r_j|.
$$

$$
E_\mathrm{angle}
= \frac{1}{2}\sum_{(ijk)}
k_{ijk}\left(\theta_{ijk}-\theta_{ijk}^{0}\right)^2.
$$

The nonbonded term is evaluated only for non-excluded pairs.  For a pair
\((ij)\),

$$
E_{ij}^\mathrm{LJ}
=4\epsilon_{ij}\left[
\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12}
-\left(\frac{\sigma_{ij}}{r_{ij}}\right)^6
\right],
\qquad
E_{ij}^\mathrm{Coulomb}
= k_e\frac{q_iq_j}{r_{ij}}.
$$

In a real force field these parameters would come from AMBER, CHARMM, OPLS,
GROMOS, or another parameter source.  Here they are programmatic so the data
model remains inspectable.
"""
        ),
        code(
            r"""
from mlx_atomistic import (
    AngleParameter,
    AtomType,
    BondParameter,
    Cell,
    DistanceConstraints,
    ForceField,
    MMSystem,
    NonbondedParameter,
    Topology,
)
from mlx_atomistic.md import SimulationConfig, simulate_nve

cell = Cell.orthorhombic((8.0, 8.0, 8.0))
topology = Topology.from_sequences(
    n_atoms=3,
    bonds=[(0, 1), (0, 2)],
    angles=[(1, 0, 2)],
    partial_charges=[-0.8, 0.4, 0.4],
)

positions = np.array(
    [
        [4.000, 4.000, 4.000],
        [4.957, 4.000, 4.000],
        [3.760, 4.927, 4.000],
    ],
    dtype=np.float32,
)
velocities = np.array(
    [
        [0.000, 0.010, 0.000],
        [0.000, -0.020, 0.000],
        [0.000, 0.010, 0.000],
    ],
    dtype=np.float32,
)

force_field = ForceField(
    atom_types=[AtomType("O", 16.0), AtomType("H", 1.0)],
    nonbonded=[
        NonbondedParameter("O", sigma=1.0, epsilon=0.20),
        NonbondedParameter("H", sigma=0.6, epsilon=0.04),
    ],
    bonds=[BondParameter(("O", "H"), k=250.0, length=0.957)],
    angles=[AngleParameter(("H", "O", "H"), k=40.0, angle=np.deg2rad(104.5))],
    cutoff=3.0,
    lj_shift=True,
)
system = MMSystem.from_sequences(
    symbols=["O", "H", "H"],
    atom_types=["O", "H", "H"],
    positions=positions,
    velocities=velocities,
    topology=topology,
    masses=force_field.masses_for(["O", "H", "H"]),
    charges=[-0.8, 0.4, 0.4],
    cell=cell,
)
terms = force_field.build_force_terms(system)

print("force terms:", [getattr(term, "name", type(term).__name__) for term in terms])
print("topology exclusions:", np.asarray(topology.exclusions).tolist())
print("nonbonded pairs:", np.asarray(topology.nonbonded_pairs()).tolist())
"""
        ),
        markdown(
            r"""
## Run constrained NVE

Velocity Verlet updates the microcanonical trajectory.  SHAKE/RATTLE-style
pair-distance constraints keep the two O–H distances close to the target:

$$
H = K(p) + U(r), \qquad \Delta E(t)=E(t)-E(0).
$$
"""
        ),
        markdown(
            r"""
### What SHAKE/RATTLE is enforcing

Each constrained pair defines a holonomic constraint

$$
g_a(r) = |r_i-r_j|^2 - d_{ij}^2 = 0.
$$

SHAKE corrects positions after the unconstrained position drift so that
\(g_a(r)\approx 0\).  RATTLE corrects velocities so the velocity is tangent to
the constraint surface:

$$
\dot g_a(r,v)
= 2(r_i-r_j)\cdot(v_i-v_j)
\approx 0.
$$

The plotted `constraint_max_error` is the largest residual distance error after
the correction step.  It should be close to the configured tolerance and should
not grow over time.
"""
        ),
        code(
            r"""
constraints = DistanceConstraints(
    pairs=[(0, 1), (0, 2)],
    distances=[0.957, 0.957],
    tolerance=1e-5,
    max_iterations=40,
)
result = simulate_nve(
    system.positions,
    system.velocities,
    masses=system.masses,
    cell=cell,
    force_terms=terms,
    config=SimulationConfig(dt=0.0005, steps=300, sample_interval=10),
    constraints=constraints,
)

mx.eval(result.total_energy, result.constraint_max_error)
times = np.arange(result.total_energy.shape[0]) * 0.0005
energy = np.asarray(result.total_energy)
drift = energy - energy[0]
constraint_error = np.asarray(result.constraint_max_error)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(times, drift)
axes[0].set_title("NVE total-energy drift")
axes[0].set_xlabel("time / reduced units")
axes[0].set_ylabel("E(t) - E(0)")
axes[1].plot(times, constraint_error)
axes[1].set_title("maximum constraint error")
axes[1].set_xlabel("time / reduced units")
axes[1].set_ylabel("|r_ij - d_ij|")
fig.tight_layout()
"""
        ),
        markdown(
            r"""
## Inspect energy decomposition and trajectory geometry

The dense scalar diagnostics preserve every integration step, while positions
are sampled sparsely.  That is the intended production pattern: keep scalar
diagnostics cheap and dense, but avoid storing every coordinate frame unless the
workflow needs it.
"""
        ),
        markdown(
            r"""
### Reading the plots

The left plot should answer: which physical term is dominating the potential
energy?  The right plot should answer: are the imposed distances actually held
fixed in the sampled trajectory?

For this toy system the H–H pair is not bonded, so it remains in the nonbonded
path.  The O–H bonded pairs are removed from ordinary LJ/Coulomb evaluation to
avoid double-counting short-range interactions.
"""
        ),
        code(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, series in result.potential_energy_by_term.items():
    axes[0].plot(times, np.asarray(series), label=name)
axes[0].set_title("potential-energy components")
axes[0].set_xlabel("time / reduced units")
axes[0].set_ylabel("energy")
axes[0].legend()

sampled = np.asarray(result.sampled_positions)
bond_01 = np.linalg.norm(sampled[:, 0] - sampled[:, 1], axis=1)
bond_02 = np.linalg.norm(sampled[:, 0] - sampled[:, 2], axis=1)
axes[1].plot(np.asarray(result.sampled_time), bond_01, label="O-H 1")
axes[1].plot(np.asarray(result.sampled_time), bond_02, label="O-H 2")
axes[1].axhline(0.957, color="black", linewidth=1, linestyle="--")
axes[1].set_title("sampled constrained bond lengths")
axes[1].set_xlabel("time / reduced units")
axes[1].set_ylabel("distance")
axes[1].legend()
fig.tight_layout()
"""
        ),
        code(
            r"""
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
frame0 = np.asarray(result.sampled_positions[0])
framef = np.asarray(result.sampled_positions[-1])
for frame, alpha, label in [(frame0, 0.35, "initial"), (framef, 1.0, "final")]:
    ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], s=[90, 45, 45], alpha=alpha, label=label)
    for i, j in [(0, 1), (0, 2)]:
        ax.plot(
            [frame[i, 0], frame[j, 0]],
            [frame[i, 1], frame[j, 1]],
            [frame[i, 2], frame[j, 2]],
            color="tab:gray",
            alpha=alpha,
        )
ax.set_title("water-like constrained molecule")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
"""
        ),
    ]


def md_validation() -> list[dict]:
    return [
        markdown(
            r"""
# MD Validation and Performance Diagnostics

For MD, speed is meaningless until forces and stability are trustworthy.  The
basic checks are:

$$
F_i^\alpha \approx -\frac{E(r_i^\alpha+\epsilon)-E(r_i^\alpha-\epsilon)}{2\epsilon},
\qquad
\Delta E_\mathrm{NVE}(t)=E(t)-E(0).
$$

This workbook turns those checks into plots instead of buried test output.
"""
        ),
        code(SETUP_CODE),
        markdown(
            r"""
## What a force check is proving

For any differentiable potential energy \(E(r)\), the Cartesian force component
is

$$
F_i^\alpha = -\frac{\partial E}{\partial r_i^\alpha}.
$$

The central-difference estimate used here is

$$
F_{i,\mathrm{fd}}^\alpha
=-\frac{
E(r_i^\alpha+\epsilon)-E(r_i^\alpha-\epsilon)
}{2\epsilon}
+\mathcal O(\epsilon^2).
$$

The validation report records two complementary errors:

$$
e_\mathrm{max}=\max_{i,\alpha}
\left|F_i^\alpha-F_{i,\mathrm{fd}}^\alpha\right|,
\qquad
e_\mathrm{rms}=
\sqrt{\frac{1}{3N}
\sum_{i,\alpha}
\left(F_i^\alpha-F_{i,\mathrm{fd}}^\alpha\right)^2 }.
$$

`max` is good for finding a single bad coordinate.  `rms` is better for seeing
whether the whole force field is noisy.
"""
        ),
        markdown(
            r"""
## Finite-difference force checks

Each force term is compared against central finite differences on a seeded toy
geometry.  The bar plot makes the outlier coordinate visible.
"""
        ),
        code(
            r"""
from mlx_atomistic.validation import run_force_validation_suite, summarize_validation_results

validation = run_force_validation_suite(cases_per_term=1, epsilon=1e-3, tolerance=5e-3)
summary = summarize_validation_results(validation)
rows = [item.to_dict() for item in validation]
print(summary)

labels = [row["case_name"] for row in rows]
errors = [row["max_abs_error"] for row in rows]
tolerances = [row["tolerance"] for row in rows]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(labels, errors, label="max |F - F_fd|")
ax.plot(labels, tolerances, color="tab:red", marker="o", label="tolerance")
ax.set_yscale("log")
ax.set_ylabel("force error")
ax.set_title("finite-difference force validation")
ax.tick_params(axis="x", rotation=30)
ax.legend()
fig.tight_layout()
"""
        ),
        markdown(
            r"""
## LJ liquid smoke trajectory

The neighbor list should preserve the same qualitative physics while reducing
the number of evaluated pairs.  The following small run is intentionally quick:
it checks finite energies, pair counts, rebuild counts, temperature, and drift.
"""
        ),
        markdown(
            r"""
### Neighbor-list model

The all-pairs nonbonded loop has \(\mathcal O(N^2)\) candidate pairs:

$$
N_\mathrm{pairs} = \frac{N(N-1)}{2}.
$$

With a cutoff \(r_c\), only nearby pairs are physically evaluated.  A Verlet
neighbor list uses a larger search radius \(r_c+r_\mathrm{skin}\), then reuses
that list until atoms have moved far enough:

$$
\max_i |\Delta r_i| > \frac{1}{2}r_\mathrm{skin}.
$$

The pair-count plot is a direct proxy for nonbonded work.  The rebuild-count
plot tells us how often we pay the more expensive list-construction cost.
"""
        ),
        code(
            r"""
from mlx_atomistic import Cell
from mlx_atomistic.md import LennardJonesPotential, SimulationConfig, simulate_nve
from mlx_atomistic.neighbors import NeighborListManager

def cubic_lattice(n_side: int, spacing: float) -> np.ndarray:
    grid = np.array(np.meshgrid(*(np.arange(n_side) for _ in range(3)), indexing="ij"))
    return (grid.reshape(3, -1).T + 0.5).astype(np.float32) * spacing

rng = np.random.default_rng(11)
positions = cubic_lattice(3, 1.55)
velocities = rng.normal(scale=0.04, size=positions.shape).astype(np.float32)
velocities -= velocities.mean(axis=0, keepdims=True)
masses = np.ones(positions.shape[0], dtype=np.float32)
cell = Cell.orthorhombic((5.5, 5.5, 5.5))
potential = LennardJonesPotential(cutoff=2.5, shift=True)
neighbors = NeighborListManager(cell=cell, cutoff=2.5, skin=0.4)

lj = simulate_nve(
    positions,
    velocities,
    masses=masses,
    cell=cell,
    force_terms=potential,
    neighbor_manager=neighbors,
    config=SimulationConfig(dt=0.002, steps=250, sample_interval=10),
)
mx.eval(lj.total_energy, lj.temperature, lj.pair_count, lj.rebuild_count)

time = np.arange(lj.total_energy.shape[0]) * 0.002
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(time, np.asarray(lj.energy_drift))
axes[0].set_title("NVE energy drift")
axes[0].set_xlabel("time")
axes[0].set_ylabel("E(t) - E(0)")
axes[1].plot(time, np.asarray(lj.temperature))
axes[1].set_title("instantaneous temperature")
axes[1].set_xlabel("time")
axes[2].plot(time, np.asarray(lj.pair_count), label="pairs")
axes[2].plot(time, np.asarray(lj.rebuild_count), label="rebuilds")
axes[2].set_title("neighbor diagnostics")
axes[2].set_xlabel("time")
axes[2].legend()
fig.tight_layout()
"""
        ),
        markdown(
            r"""
## What this tells us

For this library, MD quality should be judged on three axes: force correctness,
energy/temperature stability, and pair-path scaling.  The tests enforce these
numerically; this notebook shows the same evidence visually.
"""
        ),
        markdown(
            r"""
### Practical interpretation

Use this notebook when changing force terms, neighbor-list logic, or integrator
details:

- if force errors jump, inspect the failing atom/axis before benchmarking;
- if NVE drift grows monotonically, reduce `dt` or inspect force consistency;
- if pair counts stay close to all-pairs counts, the cutoff/cell setup is not
  giving the neighbor list much opportunity to help;
- if rebuilds happen every step, the `skin` is too small for the timestep and
  temperature.
"""
        ),
    ]


def dft_density_scf() -> list[dict]:
    return [
        markdown(
            r"""
# DFT Foundations: grid, density, potentials, SCF

This is the DFT entry notebook.  It keeps the current limits explicit:
orthorhombic cell, plane-wave/grid representation, Γ point by default, and a
toy local pseudopotential path.

The Kohn-Sham equations replace the many-electron wavefunction with auxiliary
one-electron orbitals:

$$
\left[-\frac{1}{2}\nabla^2 + V_\mathrm{loc}(r) + V_H[\rho](r) + V_{xc}[\rho](r)\right]
\psi_i(r)=\epsilon_i\psi_i(r),
$$

with closed-shell density

$$
\rho(r)=2\sum_i |\psi_i(r)|^2.
$$
"""
        ),
        code(SETUP_CODE),
        markdown(
            r"""
## Grid representation and integration

The real-space grid stores fields such as \(\rho(r)\), \(V(r)\), and
\(\psi_i(r)\) on \(N_xN_yN_z\) points inside an orthorhombic periodic cell.
Integrals become weighted sums:

$$
\int_\Omega f(r)\,dr
\approx \Delta V\sum_g f(r_g),
\qquad
\Delta V = \frac{\Omega}{N_xN_yN_z}.
$$

For normalized orbitals,

$$
\int_\Omega |\psi_i(r)|^2\,dr = 1,
\qquad
\int_\Omega \rho(r)\,dr = N_e.
$$

The reciprocal grid supplies vectors \(G\).  In a plane-wave representation the
kinetic operator is diagonal:

$$
T\psi_G = \frac{1}{2}|G|^2\psi_G.
$$
"""
        ),
        markdown(
            r"""
## Hartree and exchange-correlation pieces

The Hartree potential is the classical electrostatic potential generated by the
electron density:

$$
\nabla^2 V_H(r) = -4\pi\rho(r).
$$

In reciprocal space this becomes

$$
V_H(G)=\frac{4\pi\rho(G)}{|G|^2},\qquad G\ne 0.
$$

The \(G=0\) term is set to zero in this periodic toy implementation, which
chooses a reference for the average electrostatic potential.

For exchange-correlation, this notebook uses an LDA-style functional:

$$
E_{xc}[\rho] = \int_\Omega \rho(r)\,
\epsilon_{xc}(\rho(r))\,dr,
\qquad
V_{xc}(r)=\frac{\delta E_{xc}}{\delta\rho(r)}.
$$
"""
        ),
        markdown(
            r"""
## Build and solve a two-center toy system

The SCF loop repeatedly builds the effective potential from the current density,
solves the Kohn-Sham eigenproblem, rebuilds the density, and mixes it with the
previous density.
"""
        ),
        markdown(
            r"""
### SCF fixed-point iteration

The Kohn-Sham map takes an input density and returns a new output density:

$$
\rho_\mathrm{out}
= \mathcal F[\rho_\mathrm{in}].
$$

Simple linear mixing forms the next input density as

$$
\rho_{n+1}
= (1-\alpha)\rho_n+\alpha\rho_\mathrm{out}.
$$

DIIS/Pulay mixing uses a short history of residuals to extrapolate a better
input density.  The residual plotted later is

$$
R_n = \|\rho_\mathrm{out}-\rho_\mathrm{in}\|.
$$
"""
        ),
        code(
            r"""
from mlx_atomistic.dft import DFTSystem, LDAExchangeCorrelation, SCFConfig, run_scf

system = DFTSystem.two_center(
    cell=(8.0, 8.0, 8.0),
    grid_shape=(8, 8, 8),
    centers=((3.3, 4.0, 4.0), (4.9, 4.0, 4.0)),
    electron_count=2.0,
    amplitudes=(-2.0, -2.0),
    widths=(0.85, 0.85),
    charges=(0.7, 0.7),
)
config = SCFConfig(
    max_iterations=12,
    tolerance=1e-6,
    mixing=0.35,
    solver="dense",
    mixer="diis",
    convergence_mode="either",
    seed=4,
)
result = run_scf(system, config=config, xc_functional=LDAExchangeCorrelation())
print(result.to_dict() | {"history": f"{len(result.history)} iterations"})
"""
        ),
        markdown(
            r"""
## Density and potential slices

A scalar density on a 3D grid is easier to reason about with slices.  The
mid-plane below shows where the electron density sits relative to the effective
potential landscape.
"""
        ),
        markdown(
            r"""
### What to look for in the slice

For this two-center toy system the density should concentrate near the two
attractive local wells.  The potential slice combines several terms:

$$
V_\mathrm{eff}(r)
= V_\mathrm{loc}(r)+V_H[\rho](r)+V_{xc}[\rho](r).
$$

The potential and density do not need to have the same visual shape.  The
density is determined by the occupied eigenvectors of the Hamiltonian built
from that potential.
"""
        ),
        code(
            r"""
density = np.asarray(result.density)
potential = np.asarray(result.effective_potential)
z = density.shape[2] // 2

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
im0 = axes[0].imshow(density[:, :, z].T, origin="lower", cmap="magma")
axes[0].set_title(r"density slice $\rho(x,y,z_\mathrm{mid})$")
fig.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(potential[:, :, z].T, origin="lower", cmap="coolwarm")
axes[1].set_title(r"effective potential slice $V_\mathrm{eff}$")
fig.colorbar(im1, ax=axes[1], fraction=0.046)
fig.tight_layout()
"""
        ),
        markdown(
            r"""
## SCF convergence diagnostics

Energy alone is not enough.  A useful SCF result should expose residuals,
energy deltas, orbital residuals, and orthonormality error.
"""
        ),
        markdown(
            r"""
### Energy consistency checks

The total energy is decomposed so we can see which physical term changed:

$$
E_\mathrm{total}
= E_\mathrm{kinetic}
+ E_\mathrm{local}
+ E_H
+ E_{xc}
+ E_\mathrm{center-center}.
$$

Orbital residuals check whether the reported orbitals solve the eigenproblem:

$$
r_i = \|H\psi_i-\epsilon_i\psi_i\|.
$$

Orthonormality checks whether the occupied orbitals still satisfy

$$
\langle\psi_i|\psi_j\rangle = \delta_{ij}.
$$
"""
        ),
        code(
            r"""
history = result.history
iterations = [row["iteration"] for row in history]
energies = [row["total"] for row in history]
residuals = [row["density_residual"] for row in history]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(iterations, energies, marker="o")
axes[0].set_title("SCF total energy")
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("energy / Ha")
axes[1].semilogy(iterations, residuals, marker="o")
axes[1].set_title("density residual")
axes[1].set_xlabel("iteration")
axes[1].set_ylabel(r"$||\rho_\mathrm{out}-\rho_\mathrm{in}||$")
fig.tight_layout()

print("energy terms:", result.energy_by_term)
orbital_residuals = (
    None if result.orbital_residuals is None else np.asarray(result.orbital_residuals)
)
print("orbital residuals:", orbital_residuals)
print("orthonormality error:", result.orthonormality_error)
"""
        ),
    ]


def dft_pseudopotentials_nonlocal() -> list[dict]:
    return [
        markdown(
            r"""
# Pseudopotentials and Nonlocal Projectors

Real pseudopotential files separate core electrons from valence electrons.  In
the current implementation:

$$
V_\mathrm{ps}=V_\mathrm{local}+\sum_{ij}|\beta_i\rangle D_{ij}\langle\beta_j|.
$$

The local part is applied on the real-space grid.  The separable nonlocal part
is represented and applied as a correctness-first projector operator.
"""
        ),
        code(SETUP_CODE),
        markdown(
            r"""
## Why pseudopotentials exist

An all-electron Hamiltonian must resolve rapidly oscillating core states near
the nucleus.  A pseudopotential replaces the nucleus plus frozen core electrons
with an effective ion potential seen by the valence electrons:

$$
V_\mathrm{ion}
\;\longrightarrow\;
V_\mathrm{ps}
= V_\mathrm{local}(r) + V_\mathrm{nonlocal}.
$$

This makes a plane-wave/grid calculation far cheaper because the valence
wavefunctions are smoother near the ion centers.  The tradeoff is that the
pseudopotential format and projector conventions become part of the numerical
contract.
"""
        ),
        markdown(
            r"""
## Parse UPF and GTH reference inputs

The vendored QE/CP2K trees are reference data only.  We parse the files directly
without importing or linking those packages.
"""
        ),
        markdown(
            r"""
### Local radial potentials

UPF files usually provide tabulated radial data:

$$
\{r_m, V_\mathrm{local}(r_m)\}_{m=1}^{M}.
$$

The code interpolates that radial function onto the periodic real-space grid by
using the minimum-image ion displacement:

$$
r_I(g)=|r_g-R_I|_\mathrm{MIC},
\qquad
V_\mathrm{local}(r_g)
= \sum_I V_I^\mathrm{local}(r_I(g)).
$$

GTH local potentials are compact analytic functions with Gaussian damping, so
they can be evaluated directly from the parsed coefficients.
"""
        ),
        code(
            r"""
from mlx_atomistic.dft import (
    DFTSystem,
    Ion,
    IonCollection,
    NonlocalPseudopotentialOperator,
    SCFConfig,
    read_gth,
    read_upf,
    run_scf,
)

upf = read_upf(ROOT / "vendors/quantum-espresso/pseudo/Si_r.upf")
gth = read_gth(ROOT / "vendors/quantum-espresso/pseudo/H-q1.gth", element="H")

metadata = [
    {
        "format": str(item.format),
        "element": item.element,
        "valence": item.valence_charge,
        "projectors": len(item.nonlocal_projectors),
        "nonlocal_available": item.nonlocal_available,
    }
    for item in (upf, gth)
]
metadata
"""
        ),
        code(
            r"""
r = np.linspace(1e-4, 4.0, 400)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(r, upf.local_potential(r), label="Si UPF local")
ax.plot(r, gth.local_potential(r), label="H GTH local")
ax.axhline(0.0, color="black", linewidth=1)
ax.set_title("radial local pseudopotentials")
ax.set_xlabel(r"$r$ / bohr")
ax.set_ylabel(r"$V_\mathrm{local}(r)$ / Ha")
ax.legend()
fig.tight_layout()
"""
        ),
        markdown(
            r"""
## Inspect projector action

The nonlocal operator is Hermitian by construction:

$$
V_\mathrm{NL}\psi = \sum_i |\beta_i\rangle D_i \langle\beta_i|\psi\rangle.
$$

The plot below shows the normalized real-space projector fields generated from
the parsed UPF metadata.
"""
        ),
        markdown(
            r"""
### Separable nonlocal energy

For occupied orbitals \(\psi_n\) with occupations \(f_n\), the diagonal
separable form used here contributes

$$
E_\mathrm{NL}
= \sum_n f_n
\sum_i D_i\left|\langle \beta_i|\psi_n\rangle\right|^2.
$$

The corresponding operator action is

$$
V_\mathrm{NL}\psi_n
= \sum_i |\beta_i\rangle D_i\langle\beta_i|\psi_n\rangle.
$$

The important implementation checks are: the projector fields are normalized on
the grid, the operator is Hermitian, and the energy appears exactly once in the
SCF decomposition.
"""
        ),
        code(
            r"""
ions = IonCollection([Ion("Si", (4.0, 4.0, 4.0), upf)])
system = DFTSystem(cell=(8.0, 8.0, 8.0), grid_shape=(4, 4, 4), ions=ions)
operator = NonlocalPseudopotentialOperator.from_ions(ions, system.grid)
print(operator.to_dict())

projectors = np.asarray(operator.projectors.projectors)
fig, axes = plt.subplots(1, min(operator.projectors.count, 4), figsize=(12, 3))
if operator.projectors.count == 1:
    axes = [axes]
z = projectors.shape[-1] // 2
for index, ax in enumerate(axes):
    im = ax.imshow(projectors[index, :, :, z].T, origin="lower", cmap="viridis")
    ax.set_title(f"projector {index}")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
"""
        ),
        markdown(
            r"""
## Local-only versus local + nonlocal SCF

This is a small-grid diagnostic, not a production calculation.  The useful
question is whether the nonlocal term appears exactly once in the energy
decomposition and changes the Hamiltonian consistently.
"""
        ),
        markdown(
            r"""
### Force provenance

Local ion forces can be evaluated from the local potential derivative:

$$
F_I^\mathrm{local}
= -\frac{\partial}{\partial R_I}
\int \rho(r)V_I^\mathrm{local}(|r-R_I|)\,dr.
$$

The nonlocal force path in this milestone is intentionally conservative: it
uses finite differences of the nonlocal energy rather than claiming a fully
optimized analytic projector-force implementation.  Diagnostics therefore
record force provenance as local analytic, nonlocal finite-difference, and
center-center contributions.
"""
        ),
        code(
            r"""
base_config = SCFConfig(max_iterations=2, solver="dense", seed=7, convergence_mode="either")
local_only = run_scf(
    system,
    config=SCFConfig(**(base_config.__dict__ | {"apply_nonlocal": False})),
)
with_nonlocal = run_scf(
    system,
    config=SCFConfig(**(base_config.__dict__ | {"apply_nonlocal": True})),
)

labels = ["local only", "local + nonlocal"]
energies = [local_only.total_energy, with_nonlocal.total_energy]
nonlocal_terms = [
    local_only.energy_by_term.get("nonlocal_pseudopotential", 0.0),
    with_nonlocal.energy_by_term.get("nonlocal_pseudopotential", 0.0),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(labels, energies)
axes[0].set_title("total energy")
axes[0].set_ylabel("Ha")
axes[1].bar(labels, nonlocal_terms, color="tab:orange")
axes[1].set_title("nonlocal energy term")
axes[1].set_ylabel("Ha")
fig.tight_layout()

print(
    "local-only diagnostics:",
    local_only.to_dict()["nonlocal_applied"],
    local_only.energy_by_term,
)
print(
    "with-nonlocal diagnostics:",
    with_nonlocal.to_dict()["nonlocal_applied"],
    with_nonlocal.energy_by_term,
)
"""
        ),
    ]


def dft_solvers_spin_kpoints() -> list[dict]:
    return [
        markdown(
            r"""
# Solvers, Occupations, Spin, k-Points, and Bands

This notebook collects the production-core abstractions added after the toy SCF
prototype:

- dense diagonalization as a tiny-grid reference,
- Davidson-style iterative solving for larger grids,
- fixed and Fermi-Dirac occupations,
- collinear spin-density utilities,
- k-point meshes and non-SCF band diagnostics.
"""
        ),
        code(SETUP_CODE),
        markdown(
            r"""
## Solver problem statement

At each SCF iteration we need the lowest occupied eigenpairs of

$$
H[\rho]\psi_i = \epsilon_i\psi_i.
$$

Dense diagonalization builds the full Hamiltonian matrix and solves it directly.
That is excellent for tiny reference grids, but the matrix size grows with the
number of grid points \(N_g=N_xN_yN_z\):

$$
H \in \mathbb C^{N_g\times N_g}.
$$

An iterative solver only needs repeated Hamiltonian applications \(H\psi\),
which is the path we eventually want to optimize for Apple Silicon.
"""
        ),
        markdown(
            r"""
## Dense reference versus Davidson-style solver

The dense path is valuable because it is simple to validate.  The iterative path
is the future practical path.  On a tiny grid they should agree closely.
"""
        ),
        markdown(
            r"""
### Residuals and preconditioning

The eigenpair residual is

$$
r_i = H\psi_i-\epsilon_i\psi_i.
$$

A Davidson-style method expands a small subspace using preconditioned residuals.
For a plane-wave/grid Hamiltonian, a simple kinetic preconditioner uses the
dominant kinetic diagonal:

$$
P^{-1}(G)
\approx
\frac{1}{\frac{1}{2}|G|^2-\epsilon_i+\eta}.
$$

This is not the final production solver, but it exposes the diagnostics needed
to judge solver quality: residual norms, subspace size, restart count, and
orthonormality.
"""
        ),
        code(
            r"""
from mlx_atomistic.dft import (
    BandPath,
    DFTSystem,
    EigensolverConfig,
    FermiDiracOccupations,
    FixedOccupations,
    KPointMesh,
    MonkhorstPackGrid,
    SCFConfig,
    magnetization_density,
    run_band_structure,
    run_scf,
    spin_density_from_orbitals,
)

system = DFTSystem.one_center(grid_shape=(4, 4, 4), electron_count=2.0)
dense = run_scf(
    system,
    config=SCFConfig(max_iterations=4, solver="dense", seed=3, convergence_mode="either"),
)
davidson = run_scf(
    system,
    config=SCFConfig(
        max_iterations=4,
        solver="davidson",
        seed=3,
        convergence_mode="either",
        eigensolver_config=EigensolverConfig(max_iterations=6, tolerance=1e-5),
    ),
)

print("dense energy:", dense.total_energy)
print("davidson energy:", davidson.total_energy)
print("solver metadata:", davidson.solver_metadata)
"""
        ),
        code(
            r"""
def residual_trace(result):
    return [row["density_residual"] for row in result.history]

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(residual_trace(dense), marker="o", label="dense")
ax.semilogy(residual_trace(davidson), marker="s", label="davidson")
ax.set_title("SCF residual trace by solver")
ax.set_xlabel("iteration")
ax.set_ylabel("density residual")
ax.legend()
fig.tight_layout()
"""
        ),
        markdown(
            r"""
## Occupations and spin diagnostics

Fractional occupations are needed for metallic or near-degenerate systems.
Collinear spin tracks two densities:

$$
\rho(r)=\rho_\uparrow(r)+\rho_\downarrow(r),
\qquad
m(r)=\rho_\uparrow(r)-\rho_\downarrow(r).
$$
"""
        ),
        markdown(
            r"""
### Fixed versus Fermi-Dirac occupations

For fixed occupations we directly specify \(f_i\).  For finite-temperature
occupations, the electron count is enforced by the chemical potential \(\mu\):

$$
f_i =
\frac{g}{\exp\left((\epsilon_i-\mu)/T_e\right)+1},
\qquad
\sum_i f_i = N_e.
$$

Here \(g=2\) for spin-unpolarized orbitals and \(g=1\) for each collinear spin
channel.  Larger \(T_e\) smooths the occupation step and can make difficult SCF
problems easier, but it also changes the electronic free-energy model.
"""
        ),
        code(
            r"""
eigenvalues = np.array([-0.40, -0.05, 0.02, 0.25], dtype=np.float64)
fixed = FixedOccupations([1.0, 1.0], spin_mode="polarized").resolve()
fermi_low = FermiDiracOccupations(2.0, temperature=0.02).resolve(eigenvalues)
fermi_high = FermiDiracOccupations(2.0, temperature=0.12).resolve(eigenvalues)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(eigenvalues, np.asarray(fermi_low.occupations), marker="o", label="T=0.02")
ax.plot(eigenvalues, np.asarray(fermi_high.occupations), marker="s", label="T=0.12")
ax.set_title("Fermi-Dirac occupations")
ax.set_xlabel("orbital energy / Ha")
ax.set_ylabel("occupation")
ax.legend()
fig.tight_layout()

rho_up, rho_down = spin_density_from_orbitals(
    dense.orbitals,
    dense.orbitals,
    system.grid,
    up_occupations=[1.0],
    down_occupations=[0.6],
)
mag = magnetization_density(rho_up, rho_down)
print("fixed polarized count:", fixed.to_dict())
print("fermi count:", fermi_low.to_dict())
print("magnetization integral:", float(np.asarray(mx.sum(mag) * system.grid.dv)))
"""
        ),
        markdown(
            r"""
## k-point mesh and non-SCF band path

At Γ, the kinetic term is \(0.5|G|^2\).  At a general k point it becomes
\(0.5|G+k|^2\).  Band diagnostics reuse a converged density and do not update
the SCF state.
"""
        ),
        markdown(
            r"""
### Weighted k-point density

For periodic systems the density sums over bands and k-points:

$$
\rho(r)
= \sum_k w_k\sum_n f_{nk}
|\psi_{nk}(r)|^2,
\qquad
\sum_k w_k = 1.
$$

The Γ-point implementation is just the one-point special case.  A band-structure
calculation then freezes the converged density and evaluates eigenvalues along a
path such as \(\Gamma\rightarrow X\).  It should not feed those path states back
into the SCF density.
"""
        ),
        code(
            r"""
gamma = KPointMesh.gamma()
mesh = MonkhorstPackGrid((2, 1, 1))
print("Γ mesh:", gamma.to_dict())
print("2x1x1 mesh:", mesh.to_dict())

path = BandPath.line((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), count=6, start_label="Γ", end_label="X")
bands = run_band_structure(system, dense, path, n_bands=1)
band_values = np.asarray(bands.eigenvalues)[:, 0]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(len(path.points)), band_values, marker="o")
ax.set_title("non-SCF band diagnostic")
ax.set_xlabel("k-path index")
ax.set_ylabel("eigenvalue / Ha")
ax.set_xticks([0, len(path.points) - 1], ["Γ", "X"])
fig.tight_layout()
"""
        ),
    ]


def dft_relaxation_reference() -> list[dict]:
    return [
        markdown(
            r"""
# Relaxation, Stress, Restart, and Reference Checks

Geometry workflows connect SCF forces to structural updates.  The current scope
is still intentionally bounded:

- ion relaxation is the main path,
- orthorhombic stress is finite-difference diagnostic support,
- dense restart persistence is for small systems,
- reference comparisons are static fixtures, not runtime QE/CP2K calls.
"""
        ),
        code(SETUP_CODE),
        markdown(
            r"""
## From SCF force to structure update

For fixed-cell ion relaxation, the optimizer treats the SCF total energy as a
function of ion positions:

$$
E(R_1,R_2,\ldots,R_M).
$$

The force on ion \(I\) is the negative gradient:

$$
F_I = -\frac{\partial E}{\partial R_I}.
$$

A descent method chooses a displacement \(\Delta R\) that should lower the
energy.  A line search then accepts a step only when the trial SCF result is
finite and energetically acceptable.
"""
        ),
        markdown(
            r"""
## Ion-position relaxation

The optimizer reuses the previous SCF density/orbitals as continuation input.
Each accepted step records energy, force norms, step length, SCF status, and
timing summary.
"""
        ),
        markdown(
            r"""
### L-BFGS intuition

Steepest descent uses the force direction directly:

$$
\Delta R \propto F.
$$

L-BFGS estimates an inverse Hessian from recent changes in position and
gradient:

$$
s_k = R_{k+1}-R_k,
\qquad
y_k = \nabla E_{k+1}-\nabla E_k.
$$

That history approximates curvature without storing a full Hessian matrix.  The
implementation falls back to conservative steepest descent when the curvature
update is not trustworthy.
"""
        ),
        code(
            r"""
from tempfile import TemporaryDirectory

from mlx_atomistic.dft import (
    GeometryOptimizationConfig,
    ReferenceDFTCase,
    SCFConfig,
    compare_reference_case,
    finite_difference_stress,
    geometry_demo_system,
    load_dense_scf_restart,
    optimize_geometry,
    run_scf,
    save_dense_scf_restart,
)

system = geometry_demo_system("gaussian-dimer", grid_shape=(4, 4, 4))
config = GeometryOptimizationConfig(
    max_steps=3,
    force_tolerance=1e-4,
    initial_step_size=0.06,
    scf_config=SCFConfig(max_iterations=8, solver="dense", seed=9, convergence_mode="either"),
)
relaxed = optimize_geometry(system, config=config)
print(relaxed.to_dict() | {"steps": f"{len(relaxed.steps)} accepted steps"})
"""
        ),
        code(
            r"""
history = relaxed.to_dict()["history"]
energies = [row["energy"] for row in history]
max_forces = [row["max_force"] for row in history]
steps = [row["index"] for row in history]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(steps, energies, marker="o")
axes[0].set_title("accepted relaxation energies")
axes[0].set_xlabel("geometry step")
axes[0].set_ylabel("energy / Ha")
axes[1].semilogy(steps, max_forces, marker="s")
axes[1].set_title("maximum force")
axes[1].set_xlabel("geometry step")
axes[1].set_ylabel("force / Ha bohr$^{-1}$")
fig.tight_layout()
"""
        ),
        markdown(
            r"""
## Reading the relaxation trace

For a clean relaxation, energy should trend downward and the maximum force
should fall toward the configured tolerance:

$$
\max_I |F_I| \le F_\mathrm{tol}.
$$

Small non-monotonic behavior can occur if the line search permits a marginal
step or if SCF noise dominates the force scale.  Persistent force growth means
the optimizer, timestep-like step size, or SCF convergence settings need
attention.
"""
        ),
        markdown(
            r"""
## Diagonal stress by finite difference

The stress diagnostic samples \(E(L_x\pm\delta)\), \(E(L_y\pm\delta)\), and
\(E(L_z\pm\delta)\).  It is slow but useful for validating future analytic
stress work.
"""
        ),
        markdown(
            r"""
### Orthorhombic stress convention

For an orthorhombic cell with volume

$$
\Omega = L_xL_yL_z,
$$

the diagonal stress estimate used here is

$$
\sigma_{\alpha\alpha}
=-\frac{L_\alpha}{\Omega}
\frac{\partial E}{\partial L_\alpha}.
$$

The derivative is evaluated by central finite difference:

$$
\frac{\partial E}{\partial L_\alpha}
\approx
\frac{
E(L_\alpha+\delta)-E(L_\alpha-\delta)
}{2\delta}.
$$

This validates the workflow and sign conventions before we introduce an
analytic stress tensor.
"""
        ),
        code(
            r"""
stress = finite_difference_stress(
    system,
    config=SCFConfig(max_iterations=2, solver="dense", seed=5, convergence_mode="either"),
    displacement=1e-2,
)
print(stress.to_dict())

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["σ_xx", "σ_yy", "σ_zz"], np.asarray(stress.stress))
ax.axhline(0.0, color="black", linewidth=1)
ax.set_title("finite-difference diagonal stress")
ax.set_ylabel("stress / Ha bohr$^{-3}$")
fig.tight_layout()
"""
        ),
        markdown(
            r"""
## Dense restart and static reference comparison

Restart files preserve the dense arrays needed for small-system continuation.
Reference comparisons use JSON fixtures so tests do not require QE or CP2K at
runtime.
"""
        ),
        markdown(
            r"""
### What a restart must preserve

A useful small-system DFT restart must preserve the objects needed to continue
SCF without changing the physical problem:

$$
\{\rho(r), \psi_i(r), f_i, R_I, \Omega, \text{k-points}, \text{spin mode}\}.
$$

The reference comparison is intentionally loose here.  It is plumbing for later
QE/CP2K fixture validation, not a claim that the current toy local/nonlocal
paths reproduce production DFT energies.
"""
        ),
        code(
            r"""
scf = run_scf(system, config=SCFConfig(max_iterations=3, solver="dense", seed=12))
with TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "dense-restart.npz"
    save_dense_scf_restart(
        path,
        scf,
        positions=np.asarray(system.centers),
        cell_lengths=np.asarray(system.cell.lengths),
        metadata={"notebook": "relaxation-reference"},
    )
    restart = load_dense_scf_restart(path)

case = ReferenceDFTCase(
    name="toy-one-center-4x4x4",
    source="mlx-atomistic-static-reference",
    expected_energy=-0.6709365248680115,
    energy_tolerance=1.0,
)
comparison = compare_reference_case(case, observed_energy=scf.total_energy)

print("restart:", restart.to_dict())
print("reference comparison:", comparison.to_dict())
"""
        ),
    ]


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook("01-md-molecular-mechanics.ipynb", md_core())
    write_notebook("02-md-validation-performance.ipynb", md_validation())
    write_notebook("03-dft-density-scf.ipynb", dft_density_scf())
    write_notebook("04-dft-pseudopotentials-nonlocal.ipynb", dft_pseudopotentials_nonlocal())
    write_notebook("05-dft-solvers-spin-kpoints.ipynb", dft_solvers_spin_kpoints())
    write_notebook("06-dft-relaxation-reference.ipynb", dft_relaxation_reference())


if __name__ == "__main__":
    main()
