"""DHFR benchmark readiness and MLX-side benchmark entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from mlx_atomistic.artifacts import (
    REQUIRED_ARRAYS,
    artifact_readiness_report,
    build_mlx_system_from_artifact,
    load_prepared_mlx_artifact,
)
from mlx_atomistic.benchmarks import (
    default_benchmark_command,
    get_hardware_info,
    normalize_benchmark_payload,
)
from mlx_atomistic.md import LangevinThermostat, SimulationConfig, simulate_nvt
from mlx_atomistic.neighbors import NeighborListManager
from mlx_atomistic.pme import PMEConfig, pme_readiness_report
from mlx_atomistic.prep.io import save_prepared_system
from mlx_atomistic.prep.topology_import import TopologyImportError, import_amber_prmtop
from mlx_atomistic.runtime import get_runtime_info

BENCHMARK_NAME = "dhfr"
COMMAND = default_benchmark_command(BENCHMARK_NAME)

DEFAULT_ARTIFACT_ROOT = Path("outputs/benchmarks/dhfr-artifacts")
GBSA_REQUIRED_ARRAYS = ("gbsa_radius", "gbsa_scale")
PME_REQUIRED_ARRAYS = (
    "pme_mesh_shape",
    "pme_alpha",
    "pme_real_cutoff",
    "pme_assignment_order",
    "pme_charge_tolerance",
    "pme_deconvolve_assignment",
    "pme_background_policy",
)


@dataclass(frozen=True)
class DHFRCaseSpec:
    """Input and semantic metadata for one DHFR benchmark case."""

    case: str
    fixture: str
    solvent_model: str
    electrostatics_model: str
    force_field_family: str
    timing_metric: str
    input_paths: tuple[Path, ...] = ()
    primary_structure_path: Path | None = None
    amber_topology_path: Path | None = None
    amber_coordinates_path: Path | None = None


CASE_SPECS = {
    "dhfr-implicit": DHFRCaseSpec(
        case="dhfr-implicit",
        fixture="dhfr_implicit",
        solvent_model="implicit",
        electrostatics_model="gbsa_obc",
        force_field_family="caller-provided-dhfr-gbsa",
        timing_metric="ns_per_day",
    ),
    "dhfr-explicit-pme": DHFRCaseSpec(
        case="dhfr-explicit-pme",
        fixture="dhfr_explicit_pme",
        solvent_model="explicit",
        electrostatics_model="pme",
        force_field_family="caller-provided-amber-pme",
        timing_metric="ns_per_day",
    ),
}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = build_payload(args)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_human_payload(payload))


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build the DHFR readiness payload for one case."""

    case_spec = _case_spec_from_args(args)
    if args.prepare:
        return prepare_payload(
            case_spec=case_spec,
            repo_root=args.repo_root,
            artifact_root=args.artifact_root,
            implicit_prep_script=args.implicit_prep_script,
        )
    if args.steps is not None:
        return runtime_payload(
            case_spec=case_spec,
            steps=args.steps,
            repo_root=args.repo_root,
            artifact_root=args.artifact_root,
            implicit_prep_script=args.implicit_prep_script,
        )
    if args.readiness:
        return readiness_payload(case_spec=case_spec, repo_root=args.repo_root)
    return readiness_payload(case_spec=case_spec, repo_root=args.repo_root)


def readiness_payload(*, case_spec: DHFRCaseSpec, repo_root: Path | None = None) -> dict[str, Any]:
    """Return normalized DHFR input-readiness metadata without running dynamics."""

    root = Path.cwd() if repo_root is None else Path(repo_root)
    input_status = _input_status(case_spec, root)
    atom_count = (
        _pdb_atom_count(root / case_spec.primary_structure_path)
        if case_spec.primary_structure_path is not None
        else None
    )
    amber_atom_count = (
        _amber_prmtop_atom_count(root / case_spec.amber_topology_path)
        if case_spec.amber_topology_path is not None
        else None
    )
    blocker = None
    if input_status["missing_input_paths"]:
        blocker = "missing DHFR input path(s): " + ", ".join(input_status["missing_input_paths"])
    payload: dict[str, Any] = {
        "case": case_spec.case,
        "comparison_pair_id": case_spec.case,
        "comparison_role": "mlx",
        "fixture": case_spec.fixture,
        "system": case_spec.case,
        "readiness_only": True,
        "input_status": input_status,
        "solvent_model": case_spec.solvent_model,
        "electrostatics_model": case_spec.electrostatics_model,
        "force_field_family": case_spec.force_field_family,
        "primary_structure_path": (
            None
            if case_spec.primary_structure_path is None
            else str(case_spec.primary_structure_path)
        ),
        "amber_topology_path": (
            None if case_spec.amber_topology_path is None else str(case_spec.amber_topology_path)
        ),
        "amber_coordinates_path": (
            None
            if case_spec.amber_coordinates_path is None
            else str(case_spec.amber_coordinates_path)
        ),
        "atom_count": amber_atom_count if amber_atom_count is not None else atom_count,
        "pdb_atom_count": atom_count,
        "amber_atom_count": amber_atom_count,
        "cell_metadata_available": case_spec.amber_coordinates_path is not None
        and (root / case_spec.amber_coordinates_path).exists(),
        "unsupported_terms": [],
        "raw_input_paths": [str(path) for path in case_spec.input_paths],
    }
    return normalize_benchmark_payload(
        payload,
        benchmark_name=BENCHMARK_NAME,
        fixture=case_spec.fixture,
        timing_metric=case_spec.timing_metric,
        hardware=get_hardware_info(),
        runtime=asdict(get_runtime_info()),
        atom_count=payload["atom_count"],
        status="blocked" if blocker else "ok",
        blocker=blocker,
        command=COMMAND,
        raw_output_path=(
            f"outputs/benchmarks/same-workload-openmm-comparison/mlx-{case_spec.case}.json"
        ),
    )


def prepare_payload(
    *,
    case_spec: DHFRCaseSpec,
    repo_root: Path | None = None,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    implicit_prep_script: Path | None = None,
) -> dict[str, Any]:
    """Prepare a DHFR artifact when inputs and compatibility metadata are available."""

    return _prepare_payload(
        case_spec=case_spec,
        repo_root=repo_root,
        artifact_root=artifact_root,
        implicit_prep_script=implicit_prep_script,
    )


def _prepare_payload(
    *,
    case_spec: DHFRCaseSpec,
    repo_root: Path | None = None,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    implicit_prep_script: Path | None = None,
) -> dict[str, Any]:
    """Prepare a DHFR artifact with explicit local reference inputs."""

    root = Path.cwd() if repo_root is None else Path(repo_root)
    payload = readiness_payload(case_spec=case_spec, repo_root=root)
    payload["readiness_only"] = False
    payload["prepare"] = True
    payload["artifact_status"] = "not_attempted"
    payload["artifact_path"] = str(artifact_root / case_spec.case)
    payload["required_arrays"] = list(REQUIRED_ARRAYS)
    payload["force_term_required_arrays"] = _force_term_required_arrays(case_spec)
    payload["unsupported_terms"] = []
    payload["artifact_readiness"] = None
    payload["gbsa_obc"] = None
    payload["pme"] = None
    if payload["status"] == "blocked":
        return payload
    if case_spec.solvent_model == "implicit":
        return _prepare_implicit_gbsa(
            case_spec=case_spec,
            repo_root=root,
            payload=payload,
            artifact_root=artifact_root,
            implicit_prep_script=implicit_prep_script,
        )
    return _prepare_explicit_pme(
        case_spec=case_spec,
        repo_root=root,
        payload=payload,
        artifact_root=artifact_root,
    )


def runtime_payload(
    *,
    case_spec: DHFRCaseSpec,
    steps: int,
    repo_root: Path | None = None,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    implicit_prep_script: Path | None = None,
) -> dict[str, Any]:
    """Return the DHFR runtime benchmark row or a concrete runtime blocker."""

    payload = _prepare_payload(
        case_spec=case_spec,
        repo_root=repo_root,
        artifact_root=artifact_root,
        implicit_prep_script=implicit_prep_script,
    )
    payload["prepare"] = False
    payload["runtime_attempted"] = False
    payload["step_count"] = steps
    payload["steps"] = steps
    payload["timing_value"] = None
    payload["timing_unit"] = "ns/day"
    payload["runtime_stage"] = "blocked"
    if payload["status"] == "blocked":
        payload["runtime_blocker_category"] = _runtime_blocker_category(payload)
        return payload
    return _run_prepared_artifact_runtime(
        case_spec=case_spec,
        repo_root=Path.cwd() if repo_root is None else Path(repo_root),
        steps=steps,
        payload=payload,
        artifact_root=artifact_root,
    )


def _prepare_implicit_gbsa(
    *,
    case_spec: DHFRCaseSpec,
    repo_root: Path,
    payload: dict[str, Any],
    artifact_root: Path,
    implicit_prep_script: Path | None,
) -> dict[str, Any]:
    artifact_dir = repo_root / artifact_root / case_spec.case
    if implicit_prep_script is None:
        blocker = "implicit DHFR preparation requires an explicit prep script path"
        payload.update(
            {
                "status": "blocked",
                "blocker": blocker,
                "artifact_status": "blocked",
                "unsupported_terms": ["gbsa_obc_artifact_prepare_unconfigured"],
                "gbsa_obc": {
                    "model": "OBC",
                    "required_arrays": list(GBSA_REQUIRED_ARRAYS),
                    "present_arrays": [],
                    "missing_arrays": list(GBSA_REQUIRED_ARRAYS),
                    "input_source": (
                        None
                        if case_spec.primary_structure_path is None
                        else str(case_spec.primary_structure_path)
                    ),
                    "blocker": blocker,
                },
            }
        )
        return payload
    try:
        _run_openmm_implicit_prep(
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            prep_script=implicit_prep_script,
        )
        artifact = load_prepared_mlx_artifact(artifact_dir, require_production=True)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as exc:
        blocker = _implicit_prep_error_message(exc)
        payload.update(
            {
                "status": "blocked",
                "blocker": blocker,
                "artifact_status": "blocked",
                "unsupported_terms": ["gbsa_obc_artifact_prepare_failed"],
                "gbsa_obc": {
                    "model": "OBC",
                    "required_arrays": list(GBSA_REQUIRED_ARRAYS),
                    "present_arrays": [],
                    "missing_arrays": list(GBSA_REQUIRED_ARRAYS),
                    "input_source": (
                        None
                        if case_spec.primary_structure_path is None
                        else str(case_spec.primary_structure_path)
                    ),
                    "blocker": blocker,
                },
            }
        )
        return payload

    arrays = artifact.arrays
    artifact_readiness = artifact_readiness_report(
        artifact.metadata,
        require_production=True,
        arrays=arrays,
    )
    gbsa_metadata = dict(
        artifact.metadata.get("protocol_metadata", {}).get("gbsa", {})
        or artifact.metadata.get("gbsa", {})
    )
    missing_gbsa_arrays = _missing_arrays(arrays, GBSA_REQUIRED_ARRAYS)
    blocker = None
    if artifact_readiness.status == "blocked":
        blocker = "; ".join(artifact_readiness.blockers)
    elif missing_gbsa_arrays:
        blocker = "missing GBSA/OBC artifact arrays: " + ", ".join(missing_gbsa_arrays)

    payload.update(
        {
            "status": "blocked" if blocker else "ok",
            "blocker": blocker,
            "atom_count": artifact.atom_count,
            "artifact_status": "saved",
            "artifact_path": str(artifact_root / case_spec.case),
            "artifact_files": [
                str(artifact_root / case_spec.case / "prepared_system.json"),
                str(artifact_root / case_spec.case / "prepared_system.npz"),
                str(artifact_root / case_spec.case / "view.pdb"),
            ],
            "required_arrays": _array_presence(arrays, REQUIRED_ARRAYS),
            "force_term_required_arrays": _array_presence(arrays, GBSA_REQUIRED_ARRAYS),
            "unsupported_terms": list(
                dict(artifact.metadata.get("compatibility_report", {})).get(
                    "unsupported_terms",
                    [],
                )
            ),
            "artifact_readiness": {
                "status": artifact_readiness.status,
                "blockers": list(artifact_readiness.blockers),
                "metadata": artifact_readiness.metadata,
            },
            "gbsa_obc": {
                "model": "OBC",
                "required_arrays": list(GBSA_REQUIRED_ARRAYS),
                "present_arrays": [
                    name for name in GBSA_REQUIRED_ARRAYS if _array_present(arrays.get(name))
                ],
                "missing_arrays": missing_gbsa_arrays,
                "input_source": (
                    None
                    if case_spec.primary_structure_path is None
                    else str(case_spec.primary_structure_path)
                ),
                "metadata": gbsa_metadata,
                "blocker": blocker,
            },
        }
    )
    return payload


def _run_openmm_implicit_prep(*, repo_root: Path, artifact_dir: Path, prep_script: Path) -> None:
    script = repo_root / prep_script
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--out",
            str(artifact_dir.relative_to(repo_root)),
            "--json",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )


def _implicit_prep_error_message(exc: Exception) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        return "OpenMM implicit DHFR artifact preparation blocked: " + details
    return f"OpenMM implicit DHFR artifact preparation blocked: {exc}"


def _runtime_blocker_category(payload: dict[str, Any]) -> str:
    unsupported = set(str(item) for item in payload.get("unsupported_terms", ()))
    blocker = str(payload.get("blocker") or "")
    if "gbsa_obc_parameters_missing" in unsupported or "gbsa_radius" in blocker:
        return "gbsa_parameter_gap"
    if "amber_10_12_nonbonded" in unsupported:
        return "amber_import_unsupported_terms"
    if "PME readiness blocked" in blocker:
        return "pme_readiness"
    if "missing DHFR input path" in blocker:
        return "input_absence"
    return "artifact_runtime_gap"


def _run_prepared_artifact_runtime(
    *,
    case_spec: DHFRCaseSpec,
    repo_root: Path,
    steps: int,
    payload: dict[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    artifact_dir = repo_root / artifact_root / case_spec.case
    setup_start = time.perf_counter()
    try:
        artifact = load_prepared_mlx_artifact(artifact_dir, require_production=True)
        system, force_terms, constraints = build_mlx_system_from_artifact(
            artifact,
            eager_nonbonded_pair_limit=0,
        )
        force_terms = _bind_runtime_pme_plan(force_terms, system.cell)
        neighbor_manager = _runtime_neighbor_manager(force_terms, system.cell)
    except (ValueError, FileNotFoundError) as exc:
        blocker = f"DHFR prepared artifact runtime setup blocked: {exc}"
        payload.update(
            {
                "status": "blocked",
                "blocker": blocker,
                "runtime_blocker_category": "artifact_runtime_gap",
            }
        )
        return payload

    setup_wall_s = time.perf_counter() - setup_start
    unit_system = artifact.unit_system
    dt_ps = 0.004
    config = SimulationConfig(
        dt=dt_ps,
        steps=steps,
        sample_interval=steps,
        diagnostic_interval=steps,
        kinetic_energy_scale=(
            1.0 if unit_system is None else unit_system.kinetic_energy_scale
        ),
        force_to_acceleration_scale=(
            1.0 if unit_system is None else unit_system.force_to_acceleration_scale
        ),
        boltzmann_constant=1.0 if unit_system is None else unit_system.boltzmann_constant,
        pressure_diagnostics=False,
        compile_force_evaluator=False,
    )
    runtime_start = time.perf_counter()
    try:
        result = simulate_nvt(
            system.positions,
            system.velocities,
            masses=system.masses,
            cell=system.cell,
            force_terms=force_terms,
            config=config,
            constraints=constraints,
            thermostat=LangevinThermostat(
                temperature=300.0,
                friction=91.0 if case_spec.solvent_model == "implicit" else 1.0,
                seed=17,
            ),
            neighbor_manager=neighbor_manager,
        )
        potential_energy = _last_scalar(result.potential_energy)
        kinetic_energy = _last_scalar(result.kinetic_energy)
        temperature = _last_scalar(result.temperature)
        constraint_max_error = _last_scalar(result.constraint_max_error)
    except (ValueError, FloatingPointError) as exc:
        blocker = f"DHFR bounded MLX runtime blocked: {exc}"
        payload.update(
            {
                "status": "blocked",
                "blocker": blocker,
                "runtime_blocker_category": "runtime_execution",
            }
        )
        return payload
    runtime_wall_s = time.perf_counter() - runtime_start
    simulated_ns = steps * dt_ps / 1000.0
    ns_per_day = simulated_ns / runtime_wall_s * 86400.0 if runtime_wall_s > 0.0 else 0.0
    finite = all(
        math.isfinite(value)
        for value in (
            ns_per_day,
            potential_energy,
            kinetic_energy,
            temperature,
            constraint_max_error,
        )
    )
    payload.update(
        {
            "status": "ok" if finite else "failed",
            "blocker": None if finite else "DHFR bounded MLX runtime produced non-finite values",
            "prepare": False,
            "runtime_attempted": True,
            "runtime_stage": "completed" if finite else "failed",
            "runtime_blocker_category": None if finite else "non_finite_runtime",
            "timing_value": ns_per_day if finite else None,
            "timing_unit": "ns/day",
            "ns_per_day": ns_per_day if finite else None,
            "dt_ps": dt_ps,
            "simulated_ns": simulated_ns,
            "wall_time_s": runtime_wall_s,
            "setup_wall_time_s": setup_wall_s,
            "potential_energy_kj_mol": potential_energy,
            "kinetic_energy_kj_mol": kinetic_energy,
            "temperature_k": temperature,
            "constraint_max_error": constraint_max_error,
            "force_term_count": len(force_terms),
            "force_terms": [
                str(getattr(term, "name", type(term).__name__))
                for term in force_terms
            ],
            "operation": "fixed_cell_nvt_step",
            "precision": "float32",
            "pme_parameters": _runtime_pme_parameters(force_terms),
            "pme_plan": _runtime_pme_plan(force_terms),
            "nonbonded_topology": _runtime_topology_report(force_terms),
            "neighbor_backend": (
                None if neighbor_manager is None else neighbor_manager.backend
            ),
            "neighbor_representation": (
                None
                if neighbor_manager is None or neighbor_manager.neighbor_list is None
                else neighbor_manager.neighbor_list.representation_kind
            ),
            "finite": finite,
        }
    )
    return payload


def _bind_runtime_pme_plan(force_terms: list[Any], cell: Any) -> tuple[Any, ...]:
    bound = []
    for term in force_terms:
        if getattr(term, "electrostatics", None) != "pme":
            bound.append(term)
            continue
        if cell is None:
            msg = "DHFR PME runtime requires a periodic fixed cell"
            raise ValueError(msg)
        bound.append(term.bind_pme_plan(cell))
    return tuple(bound)


def _runtime_neighbor_manager(
    force_terms: tuple[Any, ...],
    cell: Any,
) -> NeighborListManager | None:
    lazy_terms = [
        term
        for term in force_terms
        if getattr(getattr(term, "topology", None), "nonbonded_pair_policy", None)
        == "lazy"
    ]
    if not lazy_terms:
        return None
    if cell is None:
        msg = "DHFR lazy nonbonded runtime requires a periodic cell"
        raise ValueError(msg)
    cutoffs = {float(term.cutoff) for term in lazy_terms}
    if len(cutoffs) != 1:
        msg = "DHFR lazy nonbonded force terms require one shared cutoff"
        raise ValueError(msg)
    uses_pme = any(getattr(term, "electrostatics", None) == "pme" for term in lazy_terms)
    return NeighborListManager(
        cell,
        cutoff=cutoffs.pop(),
        skin=0.3,
        check_interval=1,
        sort_pairs=False,
        backend="mlx_cell_blocks" if uses_pme else "auto",
    )


def _runtime_pme_parameters(force_terms: tuple[Any, ...]) -> dict[str, Any] | None:
    for term in force_terms:
        config = getattr(term, "pme_config", None)
        if config is not None:
            return _pme_config_payload(config)
    return None


def _runtime_pme_plan(force_terms: tuple[Any, ...]) -> dict[str, Any] | None:
    for term in force_terms:
        plan = getattr(term, "pme_plan", None)
        if plan is not None:
            return plan.to_dict()
    return None


def _runtime_topology_report(force_terms: tuple[Any, ...]) -> dict[str, Any] | None:
    for term in force_terms:
        topology = getattr(term, "topology", None)
        if topology is None:
            continue
        return {
            "pair_policy": topology.nonbonded_pair_policy,
            "pair_cache_materialized": getattr(topology, "_nonbonded_pairs", None)
            is not None,
            "nonbonded_pair_count": topology.nonbonded_pair_count,
        }
    return None


def _last_scalar(values: Any) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return float(array.reshape(-1)[-1])


def _prepare_explicit_pme(
    *,
    case_spec: DHFRCaseSpec,
    repo_root: Path,
    payload: dict[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    if case_spec.amber_topology_path is None or case_spec.amber_coordinates_path is None:
        payload.update(
            {
                "status": "blocked",
                "blocker": "missing AMBER prmtop/inpcrd paths for explicit PME artifact import",
                "artifact_status": "blocked",
            }
        )
        return payload

    artifact_dir = repo_root / artifact_root / case_spec.case
    try:
        prepared = import_amber_prmtop(
            prmtop_path=repo_root / case_spec.amber_topology_path,
            coords_path=repo_root / case_spec.amber_coordinates_path,
        )
        prepared = _with_pme_metadata(prepared)
        save_prepared_system(prepared, artifact_dir)
        artifact = load_prepared_mlx_artifact(artifact_dir, require_production=True)
    except (TopologyImportError, ValueError, FileNotFoundError) as exc:
        unsupported_terms = _unsupported_terms_from_error(str(exc))
        payload.update(
            {
                "status": "blocked",
                "blocker": f"AMBER explicit PME artifact import blocked: {exc}",
                "artifact_status": "blocked",
                "unsupported_terms": unsupported_terms,
                "pme": {
                    "config": _pme_config_payload(_default_dhfr_pme_config()),
                    "coordinate_format": _amber_coordinate_format(
                        repo_root / case_spec.amber_coordinates_path
                    ),
                    "required_arrays": list(PME_REQUIRED_ARRAYS),
                    "present_arrays": [],
                    "missing_arrays": list(PME_REQUIRED_ARRAYS),
                    "blocker": f"AMBER explicit PME artifact import blocked: {exc}",
                },
            }
        )
        return payload

    arrays = artifact.arrays
    artifact_readiness = artifact_readiness_report(
        artifact.metadata,
        require_production=True,
        arrays=arrays,
    )
    pme_config = _pme_config_from_arrays(arrays)
    pme = pme_readiness_report(
        atom_count=artifact.atom_count,
        charges=arrays["charges"],
        cell_lengths=arrays.get("cell_lengths", np.asarray([])),
        config=pme_config,
        nonbonded_cutoff=pme_config.real_cutoff,
        exclusion_count=int(np.asarray(arrays["nonbonded_exception_pairs"]).shape[0]),
        one_four_count=int(
            dict(artifact.metadata.get("compatibility_report", {}))
            .get("term_counts", {})
            .get("amber_14_exceptions", 0)
        ),
        explicit_exception_count=int(
            np.asarray(arrays["nonbonded_exception_pairs"]).shape[0]
        ),
    )
    missing_pme_arrays = _missing_arrays(arrays, PME_REQUIRED_ARRAYS)
    blocker = None
    if artifact_readiness.status == "blocked":
        blocker = "; ".join(artifact_readiness.blockers)
    elif missing_pme_arrays:
        blocker = "missing PME artifact arrays: " + ", ".join(missing_pme_arrays)
    elif pme["status"] == "blocked":
        blocker = "PME readiness blocked: " + ", ".join(str(item) for item in pme["blockers"])

    payload.update(
        {
            "status": "blocked" if blocker else "ok",
            "blocker": blocker,
            "atom_count": artifact.atom_count,
            "artifact_status": "saved",
            "artifact_path": str(artifact_root / case_spec.case),
            "artifact_files": [
                str(artifact_root / case_spec.case / "prepared_system.json"),
                str(artifact_root / case_spec.case / "prepared_system.npz"),
                str(artifact_root / case_spec.case / "view.pdb"),
            ],
            "required_arrays": _array_presence(arrays, REQUIRED_ARRAYS),
            "force_term_required_arrays": _array_presence(arrays, PME_REQUIRED_ARRAYS),
            "unsupported_terms": list(
                dict(artifact.metadata.get("compatibility_report", {})).get(
                    "unsupported_terms",
                    [],
                )
            ),
            "artifact_readiness": {
                "status": artifact_readiness.status,
                "blockers": list(artifact_readiness.blockers),
                "metadata": artifact_readiness.metadata,
            },
            "pme": pme,
        }
    )
    return payload


def _with_pme_metadata(prepared: Any) -> Any:
    pme_config = _default_dhfr_pme_config()
    report = dict(prepared.metadata.compatibility_report)
    required_terms = list(report.get("required_terms", ()))
    supported_terms = list(report.get("supported_terms", ()))
    for terms in (required_terms, supported_terms):
        if "pme" not in terms:
            terms.append("pme")
    report.update(
        {
            "periodic_box_present": True,
            "electrostatics_model": "pme",
            "required_terms": required_terms,
            "supported_terms": supported_terms,
        }
    )
    hydrogen_masses = np.asarray(prepared.masses)[
        np.char.upper(np.asarray(prepared.symbols, dtype=str)) == "H"
    ]
    if hydrogen_masses.size and np.any(hydrogen_masses > 1.25):
        report["hydrogen_mass_repartitioning"] = "represented_by_masses"
    metadata = replace(
        prepared.metadata,
        compatibility_report=report,
        pme_config=_pme_config_payload(pme_config),
    )
    return replace(
        prepared,
        metadata=metadata,
        pme_mesh_shape=np.asarray(pme_config.mesh_shape, dtype=np.int32),
        pme_alpha=np.asarray([pme_config.alpha], dtype=np.float32),
        pme_real_cutoff=np.asarray([pme_config.real_cutoff], dtype=np.float32),
        pme_assignment_order=np.asarray([pme_config.assignment_order], dtype=np.int32),
        pme_charge_tolerance=np.asarray([pme_config.charge_tolerance], dtype=np.float32),
        pme_deconvolve_assignment=np.asarray([pme_config.deconvolve_assignment], dtype=bool),
        pme_background_policy=np.asarray([pme_config.background_policy], dtype=str),
    )


def _default_dhfr_pme_config() -> PMEConfig:
    return PMEConfig(
        mesh_shape=(64, 64, 64),
        alpha=0.35,
        real_cutoff=9.0,
        assignment_order=5,
        charge_tolerance=1e-5,
        deconvolve_assignment=True,
        background_policy="uniform_neutralizing_plasma",
    )


def _pme_config_payload(config: PMEConfig) -> dict[str, Any]:
    return {
        "mesh_shape": list(config.mesh_shape),
        "alpha": float(config.alpha),
        "real_cutoff": None if config.real_cutoff is None else float(config.real_cutoff),
        "assignment_order": int(config.assignment_order),
        "charge_tolerance": float(config.charge_tolerance),
        "deconvolve_assignment": bool(config.deconvolve_assignment),
        "background_policy": config.background_policy,
    }


def _pme_config_from_arrays(arrays: dict[str, np.ndarray]) -> PMEConfig:
    return PMEConfig(
        mesh_shape=tuple(int(item) for item in np.asarray(arrays["pme_mesh_shape"]).tolist()),
        alpha=float(np.asarray(arrays["pme_alpha"])[0]),
        real_cutoff=float(np.asarray(arrays["pme_real_cutoff"])[0]),
        assignment_order=int(np.asarray(arrays["pme_assignment_order"])[0]),
        charge_tolerance=float(np.asarray(arrays["pme_charge_tolerance"])[0]),
        deconvolve_assignment=bool(np.asarray(arrays["pme_deconvolve_assignment"])[0]),
        background_policy=str(np.asarray(arrays["pme_background_policy"])[0]),
    )


def _force_term_required_arrays(case_spec: DHFRCaseSpec) -> list[str]:
    if case_spec.solvent_model == "implicit":
        return list(GBSA_REQUIRED_ARRAYS)
    return list(PME_REQUIRED_ARRAYS)


def _array_presence(
    arrays: dict[str, np.ndarray],
    names: tuple[str, ...],
) -> dict[str, bool]:
    return {name: _array_present(arrays.get(name)) for name in names}


def _missing_arrays(arrays: dict[str, np.ndarray], names: tuple[str, ...]) -> list[str]:
    return [name for name, present in _array_presence(arrays, names).items() if not present]


def _array_present(array: np.ndarray | None) -> bool:
    return array is not None and np.asarray(array).size > 0


def _unsupported_terms_from_error(message: str) -> list[str]:
    return [
        item.removeprefix("unsupported_terms:")
        for item in message.replace(",", " ").replace(";", " ").split()
        if item.startswith("unsupported_terms:")
    ]


def _amber_coordinate_format(path: Path) -> str:
    if not path.exists():
        return "missing"
    return "netcdf" if path.read_bytes()[:3] == b"CDF" else "formatted_restart"


def _input_status(case_spec: DHFRCaseSpec, repo_root: Path) -> dict[str, Any]:
    existing: list[str] = []
    missing: list[str] = []
    if not case_spec.input_paths:
        missing.append("caller-provided DHFR input path(s)")
    if case_spec.electrostatics_model == "pme":
        if case_spec.amber_topology_path is None:
            missing.append("caller-provided AMBER topology path")
        if case_spec.amber_coordinates_path is None:
            missing.append("caller-provided AMBER coordinates path")
    for path in case_spec.input_paths:
        target = repo_root / path
        if target.exists():
            existing.append(str(path))
        else:
            missing.append(str(path))
    return {
        "existing_input_paths": existing,
        "missing_input_paths": missing,
        "all_inputs_present": not missing,
        "downloads_attempted": False,
    }


def _pdb_atom_count(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                count += 1
    return count


def _amber_prmtop_atom_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    in_pointers = False
    values: list[int] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("%FLAG "):
                if in_pointers:
                    break
                in_pointers = line.strip() == "%FLAG POINTERS"
                continue
            if not in_pointers or line.startswith("%FORMAT"):
                continue
            values.extend(int(item) for item in line.split())
            if values:
                return values[0]
    return None


def _format_human_payload(payload: dict[str, Any]) -> str:
    status = payload["status"]
    case = payload["case"]
    atom_count = payload.get("atom_count")
    if status == "blocked":
        return f"DHFR {case}: blocked ({payload['blocker']})"
    return f"DHFR {case}: ready, atom_count={atom_count}"


def _case_spec_from_args(args: argparse.Namespace) -> DHFRCaseSpec:
    spec = CASE_SPECS[args.case]
    amber_topology = args.amber_topology
    amber_coordinates = args.amber_coordinates
    input_paths: list[Path] = []
    if args.primary_structure is not None:
        input_paths.append(args.primary_structure)
    if amber_topology is not None:
        input_paths.append(amber_topology)
    if amber_coordinates is not None:
        input_paths.append(amber_coordinates)
    return replace(
        spec,
        input_paths=tuple(input_paths),
        primary_structure_path=args.primary_structure,
        amber_topology_path=amber_topology,
        amber_coordinates_path=amber_coordinates,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASE_SPECS), required=True)
    parser.add_argument("--readiness", action="store_true", help="report input readiness only")
    parser.add_argument("--prepare", action="store_true", help="prepare artifact when implemented")
    parser.add_argument("--steps", type=int, default=None, help="runtime steps for later slices")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--primary-structure", type=Path, default=None)
    parser.add_argument("--amber-topology", type=Path, default=None)
    parser.add_argument("--amber-coordinates", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--implicit-prep-script", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)
    return args


if __name__ == "__main__":
    main()
