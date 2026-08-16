"""Public self-consistent and fixed-density periodic DFT entry points."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from mlx_atomistic.dft._periodic_band import (
    run_periodic_band_structure as _run_periodic_band_structure,
)
from mlx_atomistic.dft._periodic_davidson import (
    solve_periodic_eigenproblem as solve_periodic_eigenproblem,
)
from mlx_atomistic.dft._periodic_hamiltonian import (
    PeriodicKohnShamOperator as PeriodicKohnShamOperator,
)
from mlx_atomistic.dft._periodic_models import (
    PeriodicBandPointResult as PeriodicBandPointResult,
)
from mlx_atomistic.dft._periodic_models import (
    PeriodicBandStructureResult as PeriodicBandStructureResult,
)
from mlx_atomistic.dft._periodic_models import (
    PeriodicDavidsonConfig as PeriodicDavidsonConfig,
)
from mlx_atomistic.dft._periodic_models import PeriodicDFTSystem as PeriodicDFTSystem
from mlx_atomistic.dft._periodic_models import (
    PeriodicEigenResult as PeriodicEigenResult,
)
from mlx_atomistic.dft._periodic_models import (
    PeriodicFrozenDensity as PeriodicFrozenDensity,
)
from mlx_atomistic.dft._periodic_models import (
    PeriodicKPointResult as PeriodicKPointResult,
)
from mlx_atomistic.dft._periodic_models import PeriodicSCFConfig as PeriodicSCFConfig
from mlx_atomistic.dft._periodic_models import PeriodicSCFResult as PeriodicSCFResult
from mlx_atomistic.dft._periodic_models import (
    _eigensolve_provenance as _periodic_eigensolve_provenance,
)
from mlx_atomistic.dft._periodic_scf_engine import _run_periodic_scf_controlled
from mlx_atomistic.dft._runtime_observer import RuntimeObserver
from mlx_atomistic.dft.kpoints import BandPath, KPointMesh
from mlx_atomistic.dft.xc import ExchangeCorrelationFunctional


def _eigensolve_provenance() -> dict[str, str]:
    """Preserve the periodic SCF provenance-helper import contract."""

    return _periodic_eigensolve_provenance()


def run_periodic_scf(
    system: PeriodicDFTSystem,
    *,
    cutoff_hartree: float,
    kpoint_mesh: KPointMesh,
    n_bands: int | None = None,
    config: PeriodicSCFConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    initial_density: mx.array | None = None,
    initial_coefficients: Sequence[mx.array] | None = None,
    observer: RuntimeObserver | None = None,
) -> PeriodicSCFResult:
    """Run weighted self-consistent periodic plane-wave DFT.

    Args:
        system: Periodic GTH system.
        cutoff_hartree: Kinetic cutoff in Hartree.
        kpoint_mesh: Weighted reduced-coordinate k-point mesh.
        n_bands: Number of occupied bands. Defaults to half the electron count.
        config: SCF controls. Defaults to `PeriodicSCFConfig`.
        xc_functional: Exchange-correlation functional. Defaults to production PBE.
        initial_density: Optional starting density on the FFT grid.
        initial_coefficients: Optional orbital stack per k-point.
        observer: Optional progress, synchronized timing, and work observer.

    Returns:
        Periodic SCF result with complete weighted k-point diagnostics.
    """

    return _run_periodic_scf_controlled(
        system,
        cutoff_hartree=cutoff_hartree,
        kpoint_mesh=kpoint_mesh,
        n_bands=n_bands,
        config=config,
        xc_functional=xc_functional,
        initial_density=initial_density,
        initial_coefficients=initial_coefficients,
        observer=observer,
    )


def run_periodic_band_structure(
    system: PeriodicDFTSystem,
    source: PeriodicSCFResult | PeriodicFrozenDensity,
    band_path: BandPath,
    *,
    n_bands: int | None = None,
    guard_bands: int = 0,
    config: PeriodicDavidsonConfig | None = None,
    xc_functional: ExchangeCorrelationFunctional | None = None,
    observer: RuntimeObserver | None = None,
) -> PeriodicBandStructureResult:
    """Solve production periodic bands on top of a converged frozen density.

    The ionic, Hartree, and exchange-correlation potentials are built once
    from ``source`` and reused unchanged at every path point.

    Args:
        system: Periodic GTH system matching the source SCF calculation.
        source: Converged periodic SCF result or validated portable density.
        band_path: Explicit reduced-coordinate k-point path.
        n_bands: Lowest bands to return. Defaults to occupied bands plus eight.
        guard_bands: Extra unpublished states around the requested boundary.
        config: Davidson controls. Defaults to ``PeriodicDavidsonConfig``.
        xc_functional: Exchange-correlation functional. Defaults to production PBE.
        observer: Optional progress and work observer.

    Returns:
        Fixed-density band energies, residuals, bases, and compact eigenstates.

    Raises:
        RuntimeError: If a path-point Davidson solve does not converge.
        TypeError: If ``source`` has an unsupported type.
        ValueError: If source, path, density, or band metadata are inconsistent.
    """

    return _run_periodic_band_structure(
        system,
        source,
        band_path,
        n_bands=n_bands,
        guard_bands=guard_bands,
        config=config,
        xc_functional=xc_functional,
        observer=observer,
    )
