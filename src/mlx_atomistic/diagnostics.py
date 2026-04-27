"""Small diagnostic summaries for MD results."""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize_md_result(
    result: Any,
    *,
    ensemble: str | None = None,
) -> dict[str, Any]:
    """Return notebook- and CLI-friendly scalar diagnostics for an MD result."""

    temperature = np.array(result.temperature)
    total_energy = np.array(result.total_energy)
    summary: dict[str, Any] = {
        "ensemble": ensemble or ("nvt" if hasattr(result, "target_temperature") else "nve"),
        "steps": int(total_energy.shape[0] - 1),
        "initial_temperature": float(temperature[0]),
        "final_temperature": float(temperature[-1]),
        "mean_temperature": float(np.mean(temperature)),
        "initial_total_energy": float(total_energy[0]),
        "final_total_energy": float(total_energy[-1]),
        "max_energy_drift": float(np.max(np.abs(total_energy - total_energy[0]))),
    }

    if hasattr(result, "target_temperature"):
        target_temperature = float(result.target_temperature)
        summary["target_temperature"] = target_temperature
        summary["final_temperature_error"] = float(temperature[-1] - target_temperature)
        summary["mean_temperature_error"] = float(np.mean(temperature) - target_temperature)

    if hasattr(result, "pair_count"):
        summary["final_pair_count"] = int(np.array(result.pair_count)[-1])
    if hasattr(result, "rebuild_count"):
        summary["final_rebuild_count"] = int(np.array(result.rebuild_count)[-1])
    if hasattr(result, "potential_energy_by_term"):
        summary["final_potential_energy_by_term"] = {
            name: float(np.array(series)[-1])
            for name, series in result.potential_energy_by_term.items()
        }
        summary["mean_potential_energy_by_term"] = {
            name: float(np.mean(np.array(series)))
            for name, series in result.potential_energy_by_term.items()
        }

    return summary
