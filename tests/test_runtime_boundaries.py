from __future__ import annotations

import ast
import importlib.metadata as metadata
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ENGINE_ROOTS = {"openmm", "lammps"}
FORBIDDEN_REFERENCE_PATH_FRAGMENTS = (
    "vendors/",
    "quantum-espresso/",
    "cp2k/data",
    "scripts/prepare_openmm",
    "results/",
    "results/inputs/",
)
FORBIDDEN_UNIGNORED_OUTPUT_ROOTS = (
    "benchmark-output/",
    "pme-profile-output",
    "dhfr-artifacts",
)
IGNORED_GENERATED_OUTPUT_PREFIX = "outputs/benchmarks/"
REFERENCE_POLICY_TEXT_FILES = {
    Path("src/mlx_atomistic/runtime.py"),
    Path("src/mlx_atomistic/dft/references.py"),
}


def _python_files(root: Path, excluded_parts: set[str] | None = None) -> list[Path]:
    excluded_parts = excluded_parts or set()
    return sorted(
        path
        for path in root.rglob("*.py")
        if ".venv" not in path.parts and not (set(path.parts) & excluded_parts)
    )


def _import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def _import_roots(path: Path) -> set[str]:
    return {module.split(".", maxsplit=1)[0] for module in _import_modules(path)}


def _string_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _resolve_dependency_group(
    groups: dict, name: str, seen: set[str] | None = None
) -> set[str]:
    """Flatten a PEP 735 dependency group, resolving ``include-group`` entries."""
    seen = seen if seen is not None else set()
    if name in seen:
        return set()
    seen.add(name)
    resolved: set[str] = set()
    for entry in groups.get(name, []):
        if isinstance(entry, str):
            resolved.add(entry)
        elif isinstance(entry, dict) and "include-group" in entry:
            resolved |= _resolve_dependency_group(groups, entry["include-group"], seen)
    return resolved


def test_core_runtime_does_not_import_reference_engines():
    offenders = {
        path.relative_to(ROOT): sorted(_import_roots(path) & FORBIDDEN_ENGINE_ROOTS)
        for path in _python_files(ROOT / "src/mlx_atomistic")
        if _import_roots(path) & FORBIDDEN_ENGINE_ROOTS
    }

    assert offenders == {}


def test_installed_package_does_not_hardcode_reference_tree_paths():
    offenders: dict[Path, list[str]] = {}
    for path in _python_files(ROOT / "src/mlx_atomistic"):
        relative = path.relative_to(ROOT)
        if relative in REFERENCE_POLICY_TEXT_FILES:
            continue
        matches = sorted(
            {
                fragment
                for literal in _string_literals(path)
                for fragment in FORBIDDEN_REFERENCE_PATH_FRAGMENTS
                if fragment in literal
            }
        )
        if matches:
            offenders[relative] = matches

    assert offenders == {}


def test_installed_package_does_not_hardcode_unignored_output_roots():
    offenders: dict[Path, list[str]] = {}
    for path in _python_files(ROOT / "src/mlx_atomistic"):
        matches = sorted(
            {
                fragment
                for literal in _string_literals(path)
                for fragment in FORBIDDEN_UNIGNORED_OUTPUT_ROOTS
                if fragment in literal and not literal.startswith(IGNORED_GENERATED_OUTPUT_PREFIX)
            }
        )
        if matches:
            offenders[path.relative_to(ROOT)] = matches

    assert offenders == {}


def test_default_generated_benchmark_paths_are_gitignored():
    paths = [
        "outputs/benchmarks/pme-profile/pme-profile.json",
        "outputs/benchmarks/dhfr-artifacts/dhfr-implicit/prepared_system.json",
        "outputs/benchmarks/same-workload-openmm-comparison/mlx-dhfr-implicit.json",
        "outputs/benchmarks/same-workload-lj-scaling/summary.json",
    ]
    result = subprocess.run(
        ["git", "check-ignore", *paths],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ignored = set(result.stdout.splitlines())
    assert ignored == set(paths), result.stdout + result.stderr


def test_core_runtime_does_not_import_prep_layer():
    offenders: dict[Path, list[str]] = {}

    for path in _python_files(ROOT / "src/mlx_atomistic", excluded_parts={"benchmarks", "prep"}):
        imports = sorted(
            module
            for module in _import_modules(path)
            if module == "mlx_atomistic.prep" or module.startswith("mlx_atomistic.prep.")
        )
        if imports:
            offenders[path.relative_to(ROOT)] = imports

    assert offenders == {}


def test_importing_core_package_does_not_load_prep_layer():
    code = """
import sys
import mlx_atomistic

loaded = sorted(
    name for name in sys.modules
    if name == "mlx_atomistic.prep"
    or name.startswith("mlx_atomistic.prep.")
)
assert loaded == [], loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_prep_import_is_removed():
    code = """
import importlib.util

legacy_name = "atomistic" + "_prep"
assert importlib.util.find_spec(legacy_name) is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_prep_canonical_import_works():
    import mlx_atomistic.prep
    from mlx_atomistic.prep.runner import run_mlx as canonical_run_mlx

    assert "run_mlx" in mlx_atomistic.prep.__all__
    assert canonical_run_mlx is not None


def test_external_engine_imports_stay_in_documented_reference_scripts():
    allowed = {
        Path("scripts/benchmark_lammps_opencl.py"),
        Path("scripts/benchmark_m5max_reference.py"),
        Path("scripts/benchmark_openmm_dhfr.py"),
        Path("scripts/benchmark_openmm_opencl.py"),
        Path("scripts/openmm_mlx_parity.py"),
        Path("scripts/prepare_openmm_dhfr_implicit.py"),
        Path("scripts/run_openmm_mlx_npt_parity.py"),
        Path("scripts/run_openmm_gpcrmd_preview.py"),
        Path("scripts/run_openmm_gpcrmd_charmm_md.py"),
    }
    scanned_roots = [ROOT / "src", ROOT / "scripts", ROOT / "tests"]
    offenders: dict[Path, list[str]] = {}
    observed: set[Path] = set()

    for root in scanned_roots:
        for path in _python_files(root):
            relative = path.relative_to(ROOT)
            imports = _import_roots(path) & FORBIDDEN_ENGINE_ROOTS
            if not imports:
                continue
            observed.add(relative)
            if relative not in allowed:
                offenders[relative] = sorted(imports)

    assert offenders == {}
    assert observed == allowed


def test_engine_dependencies_are_not_core_runtime_dependencies():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    groups = data["dependency-groups"]
    core_dependencies = set(data["project"]["dependencies"])
    dev_dependencies = _resolve_dependency_group(groups, "dev")
    scripts = data["project"]["scripts"]

    assert "openmm>=8.5.1" not in core_dependencies
    assert "lammps>=2025.7.22.4.0" not in core_dependencies
    assert "openmm>=8.5.1" in dev_dependencies
    assert "lammps>=2025.7.22.4.0" in dev_dependencies
    # Engines reach `dev` via the `reference` group and must never leak into the
    # light `test` group that powers the fast CI lane (no LAMMPS build there).
    assert "openmm>=8.5.1" in _resolve_dependency_group(groups, "reference")
    assert "lammps>=2025.7.22.4.0" not in _resolve_dependency_group(groups, "test")
    assert "openmm>=8.5.1" not in _resolve_dependency_group(groups, "test")
    legacy_command = "atomistic" + "-prep"
    assert legacy_command not in scripts
    assert legacy_command not in {
        entry_point.name for entry_point in metadata.entry_points(group="console_scripts")
    }


def test_runtime_boundary_docs_label_reference_surfaces():
    runtime_doc = (ROOT / "docs/runtime-boundaries.md").read_text()
    notebook_doc = (ROOT / "notebooks/README.md").read_text()
    ligand_doc = (ROOT / "notebooks/ligand-receptor-motion/README.md").read_text()
    gitignore = (ROOT / ".gitignore").read_text()

    assert "primary trajectory generator" in runtime_doc
    assert "OpenMM is a reference and preview engine" in runtime_doc
    assert "LAMMPS is a reference engine" in runtime_doc
    assert "`vendors/` contains local reference source trees only" in runtime_doc
    assert "openmm-reference" in notebook_doc
    assert "openmm-reference" in ligand_doc
    assert "not production runtime output" in ligand_doc
    assert "generated MLX and OpenMM reference outputs are ignored" in gitignore
