"""Build the Phase 3 production-MD blocker matrix and readiness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
TAXONOMY_CATEGORIES = (
    "artifact_source",
    "preparation",
    "topology_terms",
    "forcefield_terms",
    "constraints_hmr_virtual_sites",
    "electrostatics_pme",
    "npt_barostat",
    "integrator_protocol",
    "stability_finiteness",
    "parity_tolerance",
    "performance_runtime",
    "output_restart",
    "dependency_boundary",
)


def build_blocker_matrix(
    *,
    candidate: dict[str, Any],
    openmm: dict[str, Any],
    mlx: dict[str, Any],
) -> dict[str, Any]:
    """Normalize fixture, OpenMM, and MLX evidence into the blocker taxonomy."""

    fixture_id = (
        candidate.get("fixture", {}).get("id")
        or openmm.get("fixture_id")
        or mlx.get("fixture", {}).get("id")
    )
    entries = {category: _base_entry(category, fixture_id) for category in TAXONOMY_CATEGORIES}

    _apply_candidate(entries, candidate)
    _apply_openmm(entries, openmm)
    _apply_mlx(entries, mlx)
    _finalize_defaults(entries, candidate, openmm, mlx)

    ordered = [entries[category] for category in TAXONOMY_CATEGORIES]
    status = "blocked" if any(item["prevents_bounded_pass"] for item in ordered) else "passed"
    return {
        "schema_version": SCHEMA_VERSION,
        "change": "production-md-readiness-fixture-probe",
        "fixture_id": fixture_id,
        "status": status,
        "bounded_pass": status == "passed",
        "summary": {
            "candidate_status": candidate.get("status"),
            "openmm_status": openmm.get("status"),
            "mlx_status": mlx.get("status"),
            "blocking_categories": [
                item["category"] for item in ordered if item["prevents_bounded_pass"]
            ],
        },
        "entries": ordered,
    }


def write_blocker_matrix(matrix: dict[str, Any], out: Path) -> None:
    """Write blocker matrix as stable JSON."""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")


def build_readiness_report(matrix: dict[str, Any]) -> str:
    """Build a compact human-readable readiness report."""

    lines = [
        "# Production MD Readiness Fixture Probe",
        "",
        f"- fixture: `{matrix['fixture_id']}`",
        f"- status: `{matrix['status']}`",
        f"- bounded pass: `{str(matrix['bounded_pass']).lower()}`",
        "",
        "## Blocking Categories",
        "",
    ]
    blocking = [entry for entry in matrix["entries"] if entry["prevents_bounded_pass"]]
    if blocking:
        for entry in blocking:
            lines.extend(
                [
                    f"- `{entry['category']}`: {entry['observed_result']}",
                    f"  - command: `{entry['command']}`",
                    f"  - next: {entry['next_implementation_decision']}",
                ]
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Category Matrix", ""])
    lines.append("| Category | Status | Prevents Pass | Observed Result |")
    lines.append("| --- | --- | --- | --- |")
    for entry in matrix["entries"]:
        observed = str(entry["observed_result"]).replace("|", "\\|")
        lines.append(
            f"| `{entry['category']}` | `{entry['status']}` | "
            f"`{str(entry['prevents_bounded_pass']).lower()}` | {observed} |"
        )
    lines.extend(
        [
            "",
            "## Production Claim Boundary",
            "",
            "This report is one bounded fixture probe. It is not broad production MD "
            "certification.",
            "",
        ]
    )
    return "\n".join(lines)


def write_readiness_report(matrix: dict[str, Any], out: Path) -> None:
    """Write the Markdown readiness report."""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_readiness_report(matrix))


def _base_entry(category: str, fixture_id: str | None) -> dict[str, Any]:
    return {
        "category": category,
        "status": "deferred",
        "fixture": fixture_id,
        "command": "not evaluated in this bounded probe",
        "observed_result": "not evaluated in this bounded probe",
        "smallest_reproduction_context": "not applicable",
        "affected_acceptance_criteria": [],
        "next_implementation_decision": "none",
        "prevents_bounded_pass": False,
    }


def _apply_candidate(entries: dict[str, dict[str, Any]], candidate: dict[str, Any]) -> None:
    if candidate.get("selected"):
        entries["artifact_source"].update(
            {
                "status": "passed",
                "command": candidate.get("fixture", {}).get(
                    "source_reproduction_command",
                    "select_production_md_fixture",
                ),
                "observed_result": (
                    "selected local GPCRmd cache fixture with "
                    f"{candidate.get('scale', {}).get('atom_count')} atoms"
                ),
                "smallest_reproduction_context": candidate.get("fixture", {}).get(
                    "source_path", "candidate evidence"
                ),
                "affected_acceptance_criteria": ["AC2", "AC7"],
                "next_implementation_decision": "use selected fixture evidence",
            }
        )
    for blocker in candidate.get("blockers", []):
        _merge_blocker(entries, blocker, default_ac=["AC2", "AC7"])
    for blocker in candidate.get("known_pre_execution_blockers", []):
        if blocker.get("status") == "blocked":
            continue
        _merge_blocker(entries, blocker, default_ac=["AC2"], prevents=False)


def _apply_openmm(entries: dict[str, dict[str, Any]], openmm: dict[str, Any]) -> None:
    if openmm.get("status") == "ran":
        entries["parity_tolerance"].update(
            {
                "status": "partial",
                "command": openmm.get("command", {}).get("command", "OpenMM reference probe"),
                "observed_result": (
                    "OpenMM reference ran with finite outputs; comparison is bounded "
                    "by documented protocol divergences"
                ),
                "smallest_reproduction_context": openmm.get("evidence_source", {}).get(
                    "run_report", "openmm-reference.json"
                ),
                "affected_acceptance_criteria": ["AC3", "AC6"],
                "next_implementation_decision": "compare against MLX probe once MLX run passes",
            }
        )
        entries["stability_finiteness"].update(
            {
                "status": "partial",
                "command": openmm.get("command", {}).get("command", "OpenMM reference probe"),
                "observed_result": openmm.get("finite_output_checks", {}).get(
                    "observed_result",
                    "OpenMM finite-output evidence recorded",
                ),
                "smallest_reproduction_context": openmm.get("evidence_source", {}).get(
                    "preview_summary", "openmm-reference.json"
                ),
                "affected_acceptance_criteria": ["AC3", "AC6"],
                "next_implementation_decision": "MLX run must produce energies for parity",
            }
        )
        entries["dependency_boundary"].update(
            {
                "status": "passed",
                "command": "OpenMM reference evidence writer",
                "observed_result": "OpenMM remains reference-only/dev evidence",
                "smallest_reproduction_context": "openmm-reference.json",
                "affected_acceptance_criteria": ["AC3", "AC8"],
                "next_implementation_decision": "preserve reference-only boundary",
            }
        )
    for blocker in openmm.get("blockers", []):
        _merge_blocker(entries, blocker, default_ac=["AC3", "AC7"])


def _apply_mlx(entries: dict[str, dict[str, Any]], mlx: dict[str, Any]) -> None:
    stages = mlx.get("stages", {})
    if stages.get("prep", {}).get("status") == "passed":
        entries["preparation"].update(
            {
                "status": "passed",
                "command": stages["prep"].get("command", "MLX prep probe"),
                "observed_result": "prepared artifact exported for selected fixture",
                "smallest_reproduction_context": mlx.get("fixture", {}).get(
                    "prep_reproduction_command", "mlx-probe.json"
                ),
                "affected_acceptance_criteria": ["AC4", "AC6"],
                "next_implementation_decision": "use prepared artifact for runtime blocker work",
            }
        )
    compatibility = stages.get("prep", {}).get("compatibility_report", {})
    if compatibility:
        required_terms = compatibility.get("required_terms", [])
        entries["forcefield_terms"].update(
            {
                "status": "passed" if required_terms else "partial",
                "command": stages.get("prep", {}).get("command", "MLX prep probe"),
                "observed_result": (
                    f"prepared terms represented: {len(required_terms)} required term families"
                ),
                "smallest_reproduction_context": "mlx-probe.json:stages.prep",
                "affected_acceptance_criteria": ["AC4", "AC6"],
                "next_implementation_decision": "keep term coverage tied to runtime support",
            }
        )
        unsupported = compatibility.get("unsupported_physics", [])
        if unsupported:
            entries["constraints_hmr_virtual_sites"].update(
                {
                    "status": "partial",
                    "command": stages.get("prep", {}).get("command", "MLX prep probe"),
                    "observed_result": ", ".join(str(item) for item in unsupported),
                    "smallest_reproduction_context": "mlx-probe.json:stages.prep",
                    "affected_acceptance_criteria": ["AC4", "AC5", "AC7"],
                    "next_implementation_decision": (
                        "verify HMR/virtual-site policy before production-length runs"
                    ),
                }
            )
    readiness = stages.get("readiness", {}).get("reports", {})
    if readiness:
        entries["integrator_protocol"].update(
            {
                "status": "partial",
                "command": "protocol_readiness_report",
                "observed_result": "NVT short proof protocol is accepted; NPT is not required",
                "smallest_reproduction_context": "mlx-probe.json:stages.readiness",
                "affected_acceptance_criteria": ["AC4", "AC6"],
                "next_implementation_decision": "keep NPT/barostat out of this NVT fixture claim",
            }
        )
        entries["npt_barostat"].update(
            {
                "status": "passed",
                "command": "protocol_readiness_report",
                "observed_result": "selected fixture protocol is NVT; no barostat required",
                "smallest_reproduction_context": "mlx-probe.json:stages.readiness",
                "affected_acceptance_criteria": ["AC5"],
                "next_implementation_decision": "do not claim NPT coverage from this fixture",
            }
        )
    finite = mlx.get("finite_checks", {})
    if finite.get("positions") and finite.get("velocities"):
        entries["stability_finiteness"].update(
            {
                "status": "partial",
                "command": mlx.get("command", "MLX probe"),
                "observed_result": (
                    "MLX prep produced finite positions and velocities; energies "
                    "are unavailable because bounded run blocked"
                ),
                "smallest_reproduction_context": "mlx-probe.json:finite_checks",
                "affected_acceptance_criteria": ["AC6", "AC7"],
                "next_implementation_decision": "rerun finite energy checks after runtime blocker",
            }
        )
    runtime = mlx.get("runtime_performance", {})
    if runtime:
        entries["performance_runtime"].update(
            {
                "status": "blocked"
                if not runtime.get("bounded_run_completed")
                else "passed",
                "command": mlx.get("command", "MLX probe"),
                "observed_result": (
                    "bounded run attempted but did not complete"
                    if not runtime.get("bounded_run_completed")
                    else "bounded run completed"
                ),
                "smallest_reproduction_context": "mlx-probe.json:runtime_performance",
                "affected_acceptance_criteria": ["AC4", "AC6", "AC7"],
                "next_implementation_decision": "fix runtime blocker before timing claims",
                "prevents_bounded_pass": not runtime.get("bounded_run_completed"),
            }
        )
    if mlx.get("dependency_boundary", {}).get("status") == "passed":
        entries["dependency_boundary"].update(
            {
                "status": "passed",
                "command": "MLX probe dependency boundary",
                "observed_result": "MLX probe imported no reference engines or vendors",
                "smallest_reproduction_context": "mlx-probe.json:dependency_boundary",
                "affected_acceptance_criteria": ["AC8"],
                "next_implementation_decision": "preserve product-runtime boundary",
            }
        )
    for blocker in mlx.get("taxonomy_blockers", []):
        _merge_blocker(entries, blocker, default_ac=["AC4", "AC6", "AC7", "AC8"])


def _finalize_defaults(
    entries: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    openmm: dict[str, Any],
    mlx: dict[str, Any],
) -> None:
    del openmm
    if entries["electrostatics_pme"]["status"] == "deferred":
        entries["electrostatics_pme"].update(
            {
                "status": "partial",
                "command": "candidate fixture and readiness evidence",
                "observed_result": (
                    "selected fixture requires periodic PME-scale electrostatics; "
                    "OpenMM reference used PME while MLX readiness reports cutoff"
                ),
                "smallest_reproduction_context": "candidate-fixture.json and mlx-probe.json",
                "affected_acceptance_criteria": ["AC5", "AC7"],
                "next_implementation_decision": (
                    "decide whether next wave implements PME runtime path for this fixture"
                ),
            }
        )
    if entries["output_restart"]["status"] == "deferred":
        entries["output_restart"].update(
            {
                "status": "blocked" if mlx.get("status") == "blocked" else "partial",
                "command": "MLX probe output check",
                "observed_result": (
                    "no trajectory, checkpoint, or restart output because bounded MLX run blocked"
                ),
                "smallest_reproduction_context": "mlx-probe.json:stages.run",
                "affected_acceptance_criteria": ["AC5", "AC7"],
                "next_implementation_decision": (
                    "record output/restart behavior after runtime blocker is fixed"
                ),
                "prevents_bounded_pass": mlx.get("status") == "blocked",
            }
        )
    if entries["topology_terms"]["status"] == "deferred" and candidate.get("selected"):
        entries["topology_terms"].update(
            {
                "status": "partial",
                "command": "selected fixture topology inspection",
                "observed_result": (
                    "large CHARMM topology selected; runtime topology path not proven"
                ),
                "smallest_reproduction_context": "candidate-fixture.json",
                "affected_acceptance_criteria": ["AC5", "AC7"],
                "next_implementation_decision": "use MLX runtime blocker evidence",
            }
        )


def _merge_blocker(
    entries: dict[str, dict[str, Any]],
    blocker: dict[str, Any],
    *,
    default_ac: list[str],
    prevents: bool | None = None,
) -> None:
    category = str(blocker.get("category", "preparation"))
    if category not in entries:
        category = "preparation"
    entry = entries[category]
    status = str(blocker.get("status", "blocked"))
    prevents_bounded_pass = (
        bool(blocker.get("prevents_bounded_pass", status == "blocked"))
        if prevents is None
        else prevents
    )
    if entry["status"] == "blocked" and status != "blocked":
        return
    entry.update(
        {
            "status": status,
            "command": str(blocker.get("command", entry["command"])),
            "observed_result": str(
                blocker.get(
                    "observed_result",
                    blocker.get("observed", entry["observed_result"]),
                )
            ),
            "smallest_reproduction_context": str(
                blocker.get(
                    "smallest_reproduction_context",
                    blocker.get("context", entry["smallest_reproduction_context"]),
                )
            ),
            "affected_acceptance_criteria": list(
                blocker.get("affected_acceptance_criteria", default_ac)
            ),
            "next_implementation_decision": str(
                blocker.get(
                    "next_implementation_decision",
                    blocker.get("next_decision", entry["next_implementation_decision"]),
                )
            ),
            "prevents_bounded_pass": prevents_bounded_pass,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--openmm", required=True, type=Path)
    parser.add_argument("--mlx", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    matrix = build_blocker_matrix(
        candidate=_read_json(args.candidate),
        openmm=_read_json(args.openmm),
        mlx=_read_json(args.mlx),
    )
    write_blocker_matrix(matrix, args.out)
    write_readiness_report(matrix, args.report)


if __name__ == "__main__":
    main()
