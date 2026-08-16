"""Canonical local paths for generated benchmark output."""

from pathlib import Path

BENCHMARK_RESULTS_ROOT = Path("results")
SAME_WORKLOAD_OUTPUT_ROOT = BENCHMARK_RESULTS_ROOT / "same-workload-openmm-comparison"
LJ_SCALING_OUTPUT_ROOT = BENCHMARK_RESULTS_ROOT / "same-workload-lj-scaling"
PME_PROFILE_OUTPUT_ROOT = BENCHMARK_RESULTS_ROOT / "pme-profile"
DHFR_ARTIFACT_ROOT = BENCHMARK_RESULTS_ROOT / "dhfr-artifacts"
