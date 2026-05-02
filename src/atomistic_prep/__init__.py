"""Optional preparation layer for MLX-compatible atomistic systems."""

from atomistic_prep.gpcrmd import (
    GPCRMD_IMPORT_REPORT_NAME,
    GPCRmdCacheFileStatus,
    GPCRmdCacheInspection,
    GPCRmdFile,
    GPCRmdInspectionError,
    GPCRmdMLXCompatibilityReport,
    GPCRmdPreparedImportAttempt,
    GPCRmdTarget,
    GPCRmdTargetError,
    attempt_gpcrmd_prepared_artifact_import,
    default_gpcrmd_targets,
    gpcrmd_mlx_compatibility_report,
    gpcrmd_selection_reports,
    inspect_gpcrmd_cache,
    load_gpcrmd_targets,
    select_gpcrmd_target,
    write_gpcrmd_import_report,
    write_gpcrmd_targets,
)
from atomistic_prep.io import load_prepared_system, save_prepared_system, synthetic_prepared_system
from atomistic_prep.prepare import (
    MissingPrepDependencyError,
    ProductionPrepNotImplementedError,
    optional_prep_dependency_status,
    prepare_p2x4_atp,
    require_production_prep_dependencies,
)
from atomistic_prep.schema import ARTIFACT_VERSION, PreparedSystem, PreparedSystemMetadata
from atomistic_prep.solvated_example import (
    SOLVATED_LIGAND_RECEPTOR_PARAMETER_SOURCE,
    SolvatedExampleError,
    ensure_solvated_ligand_receptor_example,
    prepare_solvated_ligand_receptor_example,
    validate_complete_solvated_ligand_receptor_system,
)
from atomistic_prep.t4l_benzene import T4L_BENZENE_PARAMETER_SOURCE, prepare_t4l_benzene
from atomistic_prep.topology_import import (
    TopologyImportError,
    import_amber_prmtop,
    import_charmm_with_parmed,
)


def build_mlx_system(*args, **kwargs):
    """Lazily import the MLX runner so prep inspection does not initialize Metal."""

    from atomistic_prep.runner import build_mlx_system as _build_mlx_system

    return _build_mlx_system(*args, **kwargs)


def run_mlx(*args, **kwargs):
    """Lazily import the MLX runner so prep inspection does not initialize Metal."""

    from atomistic_prep.runner import run_mlx as _run_mlx

    return _run_mlx(*args, **kwargs)


def run_steered_mlx(*args, **kwargs):
    """Lazily import the MLX SMD runner so prep inspection does not initialize Metal."""

    from atomistic_prep.runner import run_steered_mlx as _run_steered_mlx

    return _run_steered_mlx(*args, **kwargs)


__all__ = [
    "ARTIFACT_VERSION",
    "GPCRMD_IMPORT_REPORT_NAME",
    "GPCRmdCacheFileStatus",
    "GPCRmdCacheInspection",
    "GPCRmdFile",
    "GPCRmdInspectionError",
    "GPCRmdMLXCompatibilityReport",
    "GPCRmdPreparedImportAttempt",
    "GPCRmdTarget",
    "GPCRmdTargetError",
    "MissingPrepDependencyError",
    "ProductionPrepNotImplementedError",
    "PreparedSystem",
    "PreparedSystemMetadata",
    "SOLVATED_LIGAND_RECEPTOR_PARAMETER_SOURCE",
    "SolvatedExampleError",
    "T4L_BENZENE_PARAMETER_SOURCE",
    "TopologyImportError",
    "attempt_gpcrmd_prepared_artifact_import",
    "build_mlx_system",
    "default_gpcrmd_targets",
    "ensure_solvated_ligand_receptor_example",
    "gpcrmd_mlx_compatibility_report",
    "gpcrmd_selection_reports",
    "import_amber_prmtop",
    "import_charmm_with_parmed",
    "inspect_gpcrmd_cache",
    "load_gpcrmd_targets",
    "load_prepared_system",
    "optional_prep_dependency_status",
    "prepare_p2x4_atp",
    "prepare_solvated_ligand_receptor_example",
    "prepare_t4l_benzene",
    "require_production_prep_dependencies",
    "run_mlx",
    "run_steered_mlx",
    "save_prepared_system",
    "select_gpcrmd_target",
    "synthetic_prepared_system",
    "validate_complete_solvated_ligand_receptor_system",
    "write_gpcrmd_import_report",
    "write_gpcrmd_targets",
]
