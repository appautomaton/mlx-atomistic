"""Build/load/run workflow for the bundled production MLX ATP-pocket artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

import mlx_atomistic.prep.io as prep_io
import mlx_atomistic.prep.notebook as prep_notebook
import mlx_atomistic.prep.prepare as prep_prepare
import mlx_atomistic.prep.runner as prep_runner
from helpers.config import MDProtocol, NotebookPaths, PreviewSettings
from mlx_atomistic.artifacts import MLXCompatibilityError, load_prepared_mlx_artifact


@dataclass(frozen=True)
class ProductionLoadResult:
    """All notebook state produced by loading or generating a production trajectory."""

    real_trajectory_loaded: bool
    trajectory_source: str
    universe: object | None = None
    prepared_bundle: object | None = None
    prepared_artifact: object | None = None
    trajectory_record: object | None = None
    production_error: str | None = None
    generated_artifact_now: bool = False
    generated_trajectory_now: bool = False
    artifact_rebuild_reason: str | None = None
    trajectory_run_reason: str | None = None


def prepared_artifact_paths(paths: NotebookPaths):
    """Return required prepared artifact files."""

    return [
        paths.prepared_dir / "prepared_system.json",
        paths.prepared_dir / "prepared_system.npz",
        paths.prepared_dir / "view.pdb",
    ]


def missing_prepared_artifact_paths(paths: NotebookPaths):
    """Return missing prepared artifact files."""

    return [path for path in prepared_artifact_paths(paths) if not path.exists()]


def prepared_metadata_json(paths: NotebookPaths) -> dict:
    """Load prepared-system JSON metadata if present."""

    metadata_path = paths.prepared_dir / "prepared_system.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except Exception:
        return {}


def artifact_needs_rebuild(paths: NotebookPaths):
    """Return whether the bundled production artifact is missing or stale."""

    missing_paths = missing_prepared_artifact_paths(paths)
    if missing_paths:
        return True, "missing prepared artifact files"
    metadata = prepared_metadata_json(paths)
    report = metadata.get("compatibility_report", {})
    if not report.get("production_force_field", False):
        return True, "stale non-production prepared artifact"
    try:
        load_prepared_mlx_artifact(paths.prepared_dir, require_production=True)
    except MLXCompatibilityError as exc:
        return True, f"stored artifact failed production validation: {exc}"
    return False, None


def build_bundled_production_artifact(
    paths: NotebookPaths,
    *,
    cutoff_angstrom: float,
    reason: str | None,
):
    """Build and save the bundled production 4DW1 ATP-pocket artifact."""

    if not paths.atp_receptor_pdb.exists():
        msg = f"Missing bundled 4DW1 PDB: {paths.atp_receptor_pdb}"
        raise FileNotFoundError(msg)
    prepared = prep_prepare.prepare_p2x4_atp(
        pdb_path=paths.atp_receptor_pdb,
        cutoff_angstrom=cutoff_angstrom,
        backend="production_mlx",
    )
    prep_io.save_prepared_system(prepared, paths.prepared_dir)
    paths.prepared_trajectory.unlink(missing_ok=True)
    return prepared, reason


def trajectory_needs_run(paths: NotebookPaths, artifact, protocol: MDProtocol):
    """Return whether the trajectory is missing or mismatched to the protocol."""

    if not paths.prepared_trajectory.exists():
        return True, "missing MLX trajectory"
    try:
        record = prep_notebook.load_prepared_npz_trajectory(paths.prepared_trajectory)
    except Exception as exc:
        return True, f"trajectory could not be read: {type(exc).__name__}: {exc}"
    metadata = record.metadata
    if not metadata.get("production_force_field", False):
        return True, "stale non-production trajectory"
    if metadata.get("parameter_source") != artifact.metadata.get("parameter_source"):
        return True, "trajectory parameter source does not match prepared artifact"
    if record.sampled_positions.shape[1] != artifact.atom_count:
        return True, "trajectory atom count does not match prepared artifact"
    expected_protocol = {
        "steps": protocol.steps,
        "sample_interval": protocol.sample_interval,
        "dt": protocol.dt_ps,
        "temperature": protocol.temperature_k,
        "friction": protocol.friction_per_ps,
        "restraint_k": protocol.restraint_k,
        "minimize_steps": protocol.minimize_steps,
        "equilibration_steps": protocol.equilibration_steps,
        "constraint_max_iterations": protocol.constraint_max_iterations,
        "diagnostic_interval": protocol.diagnostic_interval,
    }
    for key, expected in expected_protocol.items():
        observed = metadata.get(key)
        if observed is None:
            return True, f"trajectory metadata is missing {key}"
        if isinstance(expected, float):
            if not np.isclose(float(observed), expected):
                return True, f"trajectory {key} does not match notebook protocol"
        elif int(observed) != int(expected):
            return True, f"trajectory {key} does not match notebook protocol"
    return False, None


def production_api_snippet(paths: NotebookPaths, protocol: MDProtocol) -> str:
    """Return a reproducible Python snippet for the bundled production path."""

    return "\n".join(
        [
            "from mlx_atomistic.prep.io import save_prepared_system",
            "from mlx_atomistic.prep.prepare import prepare_p2x4_atp",
            "from mlx_atomistic.prep.runner import run_mlx",
            "",
            f"prepared_dir = {str(paths.prepared_dir)!r}",
            "prepared = prepare_p2x4_atp(",
            f"    pdb_path={str(paths.atp_receptor_pdb)!r},",
            "    backend='production_mlx',",
            ")",
            "save_prepared_system(prepared, prepared_dir)",
            "run_mlx(",
            "    prepared_dir,",
            "    require_production=True,",
            f"    steps={protocol.steps},",
            f"    sample_interval={protocol.sample_interval},",
            f"    dt={protocol.dt_ps},",
            f"    temperature={protocol.temperature_k:g},",
            f"    friction={protocol.friction_per_ps:g},",
            f"    restraint_k={protocol.restraint_k:g},",
            f"    minimize_steps={protocol.minimize_steps},",
            f"    equilibration_steps={protocol.equilibration_steps},",
            f"    constraint_max_iterations={protocol.constraint_max_iterations},",
            f"    diagnostic_interval={protocol.diagnostic_interval},",
            ")",
        ]
    )


def load_production_bundle(paths: NotebookPaths):
    """Load prepared arrays, trajectory arrays, and the MDAnalysis universe."""

    bundle = prep_notebook.load_prepared_trajectory_bundle(paths.prepared_dir)
    return bundle.universe, bundle, bundle.prepared, bundle.trajectory


def ensure_production_trajectory(
    paths: NotebookPaths,
    protocol: MDProtocol,
    settings: PreviewSettings,
    *,
    run_mlx_on_the_fly: bool = True,
) -> ProductionLoadResult:
    """Ensure the bundled production artifact and trajectory exist, then load them."""

    production_error = None
    generated_artifact_now = False
    generated_trajectory_now = False
    artifact_rebuild_reason = None
    trajectory_run_reason = None
    try:
        needs_rebuild, artifact_rebuild_reason = artifact_needs_rebuild(paths)
        if needs_rebuild:
            _, artifact_rebuild_reason = build_bundled_production_artifact(
                paths,
                cutoff_angstrom=settings.prep_cutoff_angstrom,
                reason=artifact_rebuild_reason,
            )
            generated_artifact_now = True

        production_artifact = load_prepared_mlx_artifact(
            paths.prepared_dir, require_production=True
        )
        needs_run, trajectory_run_reason = trajectory_needs_run(
            paths, production_artifact, protocol
        )
        if needs_run:
            paths.prepared_trajectory.unlink(missing_ok=True)
            if run_mlx_on_the_fly:
                prep_runner.run_mlx(
                    paths.prepared_dir,
                    out=paths.prepared_trajectory,
                    steps=protocol.steps,
                    sample_interval=protocol.sample_interval,
                    dt=protocol.dt_ps,
                    temperature=protocol.temperature_k,
                    friction=protocol.friction_per_ps,
                    seed=protocol.seed,
                    restraint_k=protocol.restraint_k,
                    minimize_steps=protocol.minimize_steps,
                    equilibration_steps=protocol.equilibration_steps,
                    constraint_max_iterations=protocol.constraint_max_iterations,
                    diagnostic_interval=protocol.diagnostic_interval,
                    require_production=True,
                )
                generated_trajectory_now = True
            else:
                production_error = (
                    f"MLX run disabled and trajectory is not usable: {trajectory_run_reason}"
                )
        if production_error is None:
            universe, bundle, prepared, record = load_production_bundle(paths)
            return ProductionLoadResult(
                real_trajectory_loaded=True,
                trajectory_source="mlx_atomistic_production",
                universe=universe,
                prepared_bundle=bundle,
                prepared_artifact=prepared,
                trajectory_record=record,
                generated_artifact_now=generated_artifact_now,
                generated_trajectory_now=generated_trajectory_now,
                artifact_rebuild_reason=artifact_rebuild_reason,
                trajectory_run_reason=trajectory_run_reason,
            )
    except Exception as exc:
        production_error = f"{type(exc).__name__}: {exc}"

    return ProductionLoadResult(
        real_trajectory_loaded=False,
        trajectory_source="missing_production_mlx_artifact",
        production_error=production_error,
        generated_artifact_now=generated_artifact_now,
        generated_trajectory_now=generated_trajectory_now,
        artifact_rebuild_reason=artifact_rebuild_reason,
        trajectory_run_reason=trajectory_run_reason,
    )


def production_status_markdown(
    result: ProductionLoadResult, paths: NotebookPaths, protocol: MDProtocol
) -> str:
    """Return markdown describing the loaded or failed production trajectory."""

    if result.real_trajectory_loaded:
        prepared = result.prepared_artifact
        record = result.trajectory_record
        report = prepared.metadata.compatibility_report
        generated_artifact_text = "yes" if result.generated_artifact_now else "no"
        generated_trajectory_text = "yes" if result.generated_trajectory_now else "no"
        artifact_reason_text = result.artifact_rebuild_reason or "already valid"
        trajectory_reason_text = result.trajectory_run_reason or "already valid"
        elapsed = record.metadata.get("elapsed_wall_seconds")
        steps_per_second = record.metadata.get("integration_steps_per_second")
        elapsed_line = (
            f"- Last MLX run wall time: `{float(elapsed):.3f} s`\n"
            if elapsed is not None
            else ""
        )
        speed_line = (
            f"- Last MLX speed: `{float(steps_per_second):.1f} steps/s`\n"
            if steps_per_second is not None
            else ""
        )
        return (
            "Loaded production MLX trajectory.\n\n"
            f"- Artifact: `{paths.prepared_dir}`\n"
            f"- Rebuilt artifact now: `{generated_artifact_text}` ({artifact_reason_text})\n"
            "- Generated trajectory now: "
            f"`{generated_trajectory_text}` ({trajectory_reason_text})\n"
            f"- Frames: `{record.sampled_positions.shape[0]}`\n"
            f"- Simulated time: `{record.sampled_time[-1]:.3f} ps`\n"
            f"{elapsed_line}"
            f"{speed_line}"
            f"- Atoms: `{prepared.atom_count}`\n"
            f"- Hydrogens: `{report.get('hydrogen_count', 'unknown')}`\n"
            f"- Parameter source: `{prepared.metadata.parameter_source}`\n\n"
            "Fixed-topology semantics: no ATP hydrolysis, no bond breaking, "
            "no ligand docking/search."
        )

    return (
        "**Bundled production MLX MD did not load.**\n\n"
        f"Reason: `{result.production_error}`\n\n"
        "The notebook does not fall back to fake motion. "
        "Reproduce the same build/run path with:\n\n"
        f"```python\n{production_api_snippet(paths, protocol)}\n```"
    )
