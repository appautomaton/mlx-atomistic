# Dual-runtime baselines on Apple M5 Max

This page is the durable control record for maintainers optimizing the existing MLX/Metal DFT and molecular-dynamics runtimes. It reports only evidence generated from the current repository and the local artifacts named below. The complete suite passed its provenance, numerical, workload, host, memory, timeout, and integrity gates.

## Control contract

- One clean Git commit must identify every accepted timing artifact.
- Metal runs execute serially in separate bounded processes.
- Clean product wall time stays separate from synchronized profiling.
- A blocked provenance, numerical, route, power, timeout, memory, or integrity gate remains blocked.

Every accepted timing artifact uses control commit `c55c2651fd9c29c902fd11d7c83417d209a4318d`. DFT fixed-density and full-SCF artifacts share protocol fingerprint `32282c7b99b53f93a6064d6189066a575f5830378b943f9135461b28560f8a39`, runtime fingerprint `419e43254748fd9ff0eab9c287878eabb1fcc908def09d7ff6e0cf2ec431086d`, and workload fingerprint `d4cbb4abe682895be010362078629354201570377d0ae3c3194ced39d7ad4426`.

## Host and fixture admission

The admitted host is Apple M5 Max with 128 GB unified memory, macOS 26.5.2 build 25F84, MLX 0.31.2, Battery Power, and Low Power Mode active. Thermal pressure was unavailable from the read-only host query and is recorded as `null`; this did not weaken the required chip, power-source, or power-mode checks. The molecular-dynamics fixture is the prepared 23,558-atom 5DFR system under `results/dhfr-npt-closure/prepared`.

## Raw artifact namespace

Generated evidence lives under `results/dual-runtime-baselines/2026-08-10/`. The directory is gitignored and is not a package input or runtime dependency.

`suite.json` binds the selected reports, captures, graph, publication attestation, source identity, evidence-helper hashes, and summary below. The earlier `dft-fixed-density` attempt with `--seal` has status `blocked` because that option audits a historical architecture baseline; it is preserved locally but excluded from this current-commit suite.

## DFT fixed-density

After one warmup, five fresh samples took 0.290112, 0.293308, 0.322057, 0.279346, and 0.288371 seconds. The median was 0.290112 seconds and the full-range relative dispersion was 14.72%. `Hpsi` was the largest measured phase at 0.105465 seconds, or 35.65% of accounted phase time, followed by orthogonalization at 0.053490 seconds. Peak MLX temporary memory was 56,061,289 bytes and unified-memory high water was 402,547,791 bytes.

## DFT full SCF

The process-cold, non-resumed calculation converged in 14 SCF iterations and 65.405795 seconds, below the 300-second bound. Its final total energy was -31.5087886561 Hartree, density residual was 8.01382e-7, energy delta was 4.86771e-7 Hartree, and maximum orthonormality error was 1.67014e-6. `Hpsi` consumed 31.045308 seconds, or 48.23% of phase time; orthogonalization was next at 12.108094 seconds. The raw run and its sibling publication attestation are both included in the suite inventory.

## MD clean 5DFR

Two separate-process controls, each with 10 warmups and 75 measured steps, took 0.120709 and 0.122596 seconds. Their median was 0.121653 seconds, relative spread was 1.55%, and throughput was 616.508 steps/s, approximately 213.07 ns/day at the 0.004 ps timestep. Both runs performed two neighbor rebuilds, stayed below the 40 GB process-tree limit, and passed the scientific and route checks.

## MD structural profile

The synchronized route profile took 0.336186 seconds and reconciled 0.331021 seconds of named routes. It is structural evidence, not clean product wall time. The leading routes were `direct_spatial_tiles` at 0.052573 seconds, `integration_thermostat` at 0.049095 seconds, `neighbor_update_rebuild` at 0.036097 seconds, and `reciprocal_pme` at 0.028104 seconds.

The separate graph capture contains 79 MLX primitives, including 9 `CustomKernel` nodes, with 83 runtime asynchronous submissions and one blocking materialization. MLX peak memory was 962,225,373 bytes.

## Bottleneck decision

For DFT, `Hpsi` remains the first optimization target: it leads both the fixed-density and full-SCF phase rankings. For MD, the evidence replaces the earlier neighbor-rebuild/synchronization hypothesis with `direct_spatial_tiles`. The next implementation slice should therefore profile and optimize direct spatial-tile force evaluation first, while preserving the existing clean-control protocol and numerical checks.

<!-- dual-runtime-summary:start -->
```json
{
  "artifact_paths": {
    "dft_fixed_density": "results/dual-runtime-baselines/2026-08-10/dft-fixed-density-admitted",
    "dft_full_scf": "results/dual-runtime-baselines/2026-08-10/dft-full-scf",
    "dft_full_scf_publication": "results/dual-runtime-baselines/2026-08-10/dft-full-scf.publication",
    "dft_workload": "results/dual-runtime-baselines/2026-08-10/dft-workload",
    "graph": "results/dual-runtime-baselines/2026-08-10/md-current-step.dot",
    "graph_summary": "results/dual-runtime-baselines/2026-08-10/md-current-step.summary.json",
    "md_binding": "results/dual-runtime-baselines/2026-08-10/md-source-binding.json",
    "md_control_after": "results/dual-runtime-baselines/2026-08-10/md-control-c2-capture.json",
    "md_control_before": "results/dual-runtime-baselines/2026-08-10/md-control-c1-capture.json",
    "md_control_spread": "results/dual-runtime-baselines/2026-08-10/md-control-spread.json",
    "md_graph": "results/dual-runtime-baselines/2026-08-10/md-graph-capture.json",
    "md_profile": "results/dual-runtime-baselines/2026-08-10/md-profile-capture.json"
  },
  "bottleneck_decision": {
    "hpsi_remains_target": true,
    "neighbor_rebuild_or_synchronization_remains_target": false,
    "next_dft_target": "hpsi",
    "next_md_target": "direct_spatial_tiles"
  },
  "control_commit": "c55c2651fd9c29c902fd11d7c83417d209a4318d",
  "dft_fixed_density": {
    "median_seconds": 0.29011245793662965,
    "memory": {
      "coefficient_payload_bytes": 827008,
      "fft_workspace_bytes": 44957696,
      "hpsi_fft_workspace_bytes": 44957696,
      "hpsi_peak_temporary_bytes": 56061289,
      "peak_temporary_bytes": 56061289,
      "persistent_coefficient_bytes": 827008,
      "persistent_projector_bytes": 2067520,
      "process_high_water_bytes": 212140032,
      "projector_payload_bytes": 2067520,
      "projector_traffic_bytes": 1267389760,
      "shared_full_grid_bytes": 2809856,
      "unified_memory_high_water_bytes": 402547791
    },
    "phases": {
      "median_seconds": {
        "cpu_small_solve": 0.03246244927868247,
        "density": 0.0,
        "eigensolver_control": 0.0,
        "hpsi": 0.10546516813337803,
        "mixing": 0.0,
        "orthogonalization": 0.053489914862439036,
        "persistence": 0.0,
        "rayleigh_ritz": 0.018870927393436432,
        "setup": 0.05155199998989701,
        "unaccounted": 0.03402237989939749
      },
      "ranking": [
        "hpsi",
        "orthogonalization",
        "setup",
        "unaccounted",
        "cpu_small_solve",
        "rayleigh_ritz",
        "density",
        "eigensolver_control",
        "mixing",
        "persistence"
      ],
      "shares": {
        "cpu_small_solve": 0.10972127938494645,
        "density": 0.0,
        "eigensolver_control": 0.0,
        "hpsi": 0.35646642306012644,
        "mixing": 0.0,
        "orthogonalization": 0.1807929476459046,
        "persistence": 0.0,
        "rayleigh_ritz": 0.06378268870020129,
        "setup": 0.17424290278240573,
        "unaccounted": 0.11499375842641552
      }
    },
    "relative_dispersion": 0.14722221948685385,
    "sample_count": 5,
    "walls_seconds": [
      0.29011245793662965,
      0.2933077921625227,
      0.3220568329561502,
      0.27934583299793303,
      0.2883707501459867
    ],
    "work_counters": {
      "cpu_bridge_calls": 34,
      "cpu_bridge_elements": 63494,
      "davidson_hv_new_vectors": 274,
      "davidson_hv_reused_vectors": 1098,
      "device_materializations": 106,
      "device_materialized_arrays": 178,
      "fft_submissions": 72,
      "fft_vector_equivalents": 612,
      "hpsi_calls": 36,
      "hpsi_lane_padding_vector_equivalents": 0,
      "hpsi_submitted_vector_equivalents": 576,
      "hpsi_vector_equivalents": 306,
      "hpsi_vector_padding_equivalents": 270,
      "kpoint_lane_solves": 0,
      "orthogonalization_vectors": 258,
      "padding_elements": 0,
      "partner_reconstructions": 0,
      "projected_old_old_rebuilds": 0,
      "projector_cache_hits": 1120,
      "projector_cache_misses": 32,
      "projector_elements_generated": 258440,
      "projector_elements_loaded": 158165280,
      "projector_traffic_elements": 158423720,
      "representative_lane_solves": 0
    }
  },
  "dft_full_scf": {
    "density_residual": 8.013816454877087E-7,
    "elapsed_seconds": 65.40579512482509,
    "electron_count_error": 0.0,
    "energy_delta_hartree": 4.867712632972143E-7,
    "maximum_orthonormality_error": 0.0000016701433740777084,
    "memory": {
      "coefficient_payload_bytes": 89162240,
      "fft_workspace_bytes": 359661568,
      "hpsi_fft_workspace_bytes": 359661568,
      "hpsi_peak_temporary_bytes": 448675760,
      "peak_temporary_bytes": 448675760,
      "persistent_coefficient_bytes": 89162240,
      "persistent_projector_bytes": 222905600,
      "process_high_water_bytes": null,
      "projector_payload_bytes": 16565440,
      "projector_traffic_bytes": 529000236160,
      "shared_full_grid_bytes": 11239424,
      "unified_memory_high_water_bytes": null
    },
    "phases": {
      "median_seconds": {
        "cpu_small_solve": 4.665874325204641,
        "density": 2.415372996823862,
        "eigensolver_control": 6.702527114190161,
        "hpsi": 31.045308244181797,
        "mixing": 0.05574458185583353,
        "orthogonalization": 12.10809419117868,
        "persistence": 0.20343387499451637,
        "rayleigh_ritz": 2.965853748843074,
        "setup": 3.132554959040135,
        "unaccounted": 1.0732831726782024
      },
      "ranking": [
        "hpsi",
        "orthogonalization",
        "eigensolver_control",
        "cpu_small_solve",
        "setup",
        "rayleigh_ritz",
        "density",
        "unaccounted",
        "persistence",
        "mixing"
      ],
      "shares": {
        "cpu_small_solve": 0.07248743013836395,
        "density": 0.03752441003812966,
        "eigensolver_control": 0.10412817235900167,
        "hpsi": 0.4823093070290534,
        "mixing": 0.0008660287871533743,
        "orthogonalization": 0.18810721648686937,
        "persistence": 0.0031604792100342113,
        "rayleigh_ritz": 0.04607649101445484,
        "setup": 0.04866630408825858,
        "unaccounted": 0.01667416084868094
      }
    },
    "scf_iterations": 14,
    "total_energy_hartree": -31.508788656058726,
    "work_counters": {
      "cpu_bridge_calls": 507,
      "cpu_bridge_elements": 13205538,
      "davidson_hv_new_vectors": 97216,
      "davidson_hv_reused_vectors": 217558,
      "device_materializations": 2565,
      "device_materialized_arrays": 43707,
      "fft_submissions": 3146,
      "fft_vector_equivalents": 280384,
      "hpsi_calls": 1475,
      "hpsi_lane_padding_vector_equivalents": 808,
      "hpsi_submitted_vector_equivalents": 138748,
      "hpsi_vector_equivalents": 128096,
      "hpsi_vector_padding_equivalents": 9844,
      "kpoint_lane_solves": 1512,
      "orthogonalization_vectors": 73024,
      "padding_elements": 340054,
      "partner_reconstructions": 0,
      "projected_old_old_rebuilds": 0,
      "projector_cache_hits": 342144,
      "projector_cache_misses": 3456,
      "projector_elements_generated": 27863200,
      "projector_elements_loaded": 66097166320,
      "projector_traffic_elements": 66125029520,
      "representative_lane_solves": 1512
    }
  },
  "host_signature": {
    "chip": "Apple M5 Max",
    "low_power_mode": 1,
    "macos": {
      "BuildVersion": "25F84",
      "ProductName": "macOS",
      "ProductVersion": "26.5.2"
    },
    "mlx_version": "0.31.2",
    "power_source": "Battery Power"
  },
  "md_clean": {
    "median_seconds": 0.12165285367518663,
    "process_peak_bytes": [
      958415712,
      1103299472
    ],
    "rebuild_counts": [
      2,
      2
    ],
    "relative_spread": 0.01551237727758818,
    "steps_per_second": 616.5083492430861,
    "walls_seconds": [
      0.12070929119363427,
      0.122596416156739
    ]
  },
  "md_graph": {
    "blocking_materialization_count": 1,
    "mlx_active_memory_bytes": 72,
    "mlx_peak_memory_bytes": 962225373,
    "primitive_categories": {
      "Add": 1,
      "And": 1,
      "Arange": 8,
      "AsType": 1,
      "Broadcast": 12,
      "CustomKernel": 9,
      "Divide": 3,
      "Equal": 2,
      "ErfInv": 1,
      "ExpandDims": 1,
      "Gather": 6,
      "LogicalNot": 1,
      "LogicalOr": 2,
      "Max": 1,
      "Minimum": 1,
      "Multiply": 5,
      "NotEqual": 1,
      "RandomBits": 2,
      "Round": 1,
      "Sqrt": 1,
      "Squeeze": 12,
      "Subtract": 4,
      "Sum": 3
    },
    "primitive_count": 79,
    "runtime_async_submission_count": 83
  },
  "md_profile": {
    "accounted_route_seconds": 0.33102107886224985,
    "instrumented_wall_seconds": 0.33618574985302985,
    "neighbor": {
      "backend": "mlx_cell_tiles",
      "candidate_count": 44160881,
      "candidate_waste_count": 29461430,
      "candidate_waste_fraction": 0.6671386379270831,
      "compact_pair_count": 14699451,
      "compaction_backend": "metal_spatial_tile_prefix_scan",
      "cutoff": 9.0,
      "estimated_candidate_memory_bytes": 21314800,
      "estimated_cell_list_memory_bytes": 484984,
      "estimated_compact_pair_memory_bytes": 117595608,
      "estimated_pair_memory_bytes": 129371112,
      "fallback_reason": null,
      "force_evaluation_wall_seconds": 0.15084150270558894,
      "manager_backend": "mlx_cell_tiles",
      "measured_rebuild_wall_seconds": 0.01589270797558129,
      "measured_update_wall_seconds": 0.03569025476463139,
      "neighbor_rebuild_wall_seconds": 0.07963762385770679,
      "neighbor_update_wall_seconds": 0.12677171151153743,
      "pair_count": 14699451,
      "rebuild_count": 2,
      "representation": "tiles",
      "representation_kind": "tiles",
      "runtime_materialization_checkpoint_count": 0,
      "runtime_materialization_checkpoint_wall_seconds": 0.0,
      "runtime_materialization_diagnostic_count": 0,
      "runtime_materialization_diagnostic_wall_seconds": 0.0,
      "runtime_materialization_explicit_user_output_count": 1,
      "runtime_materialization_explicit_user_output_wall_seconds": 0.0,
      "runtime_materialization_failure_check_count": 0,
      "runtime_materialization_failure_check_wall_seconds": 0.0,
      "runtime_materialization_final_state_count": 1,
      "runtime_materialization_final_state_wall_seconds": 0.0,
      "runtime_materialization_reporter_count": 0,
      "runtime_materialization_reporter_wall_seconds": 0.0,
      "runtime_materialization_total_count": 2,
      "runtime_materialization_total_wall_seconds": 0.0,
      "runtime_sync_checkpoint_count": 0,
      "runtime_sync_checkpoint_wall_seconds": 0.0,
      "runtime_sync_diagnostic_count": 1,
      "runtime_sync_diagnostic_wall_seconds": 0.00028849998489022255,
      "runtime_sync_explicit_user_output_count": 1,
      "runtime_sync_explicit_user_output_wall_seconds": 0.000001125037670135498,
      "runtime_sync_failure_check_count": 2,
      "runtime_sync_failure_check_wall_seconds": 0.0000025420449674129486,
      "runtime_sync_final_state_count": 1,
      "runtime_sync_final_state_wall_seconds": 0.0000012081582099199295,
      "runtime_sync_reporter_count": 0,
      "runtime_sync_reporter_wall_seconds": 0.0,
      "runtime_sync_total_count": 5,
      "runtime_sync_total_wall_seconds": 0.0002933752257376909,
      "skin": 5.5
    },
    "residual_seconds": 0.005164670990779996,
    "route_ranking": [
      "direct_spatial_tiles",
      "integration_thermostat",
      "neighbor_update_rebuild",
      "reciprocal_pme",
      "force_aggregation",
      "shake_clusters_position",
      "shake_clusters_velocity",
      "shake_clusters_pre_force_velocity",
      "settle_position",
      "settle_velocity",
      "pme_exceptions_corrections",
      "bonded_fused",
      "diagnostics_reporting",
      "neighbor_force_binding",
      "constraint_validation"
    ],
    "route_seconds": {
      "bonded_fused": 0.015089167281985283,
      "constraint_validation": 0.0014426237903535366,
      "diagnostics_reporting": 0.013536250218749046,
      "direct_spatial_tiles": 0.052572837099432945,
      "force_aggregation": 0.025212588720023632,
      "integration_thermostat": 0.04909476405009627,
      "neighbor_force_binding": 0.00425740797072649,
      "neighbor_update_rebuild": 0.036097087897360325,
      "pme_exceptions_corrections": 0.015304338652640581,
      "reciprocal_pme": 0.02810438210144639,
      "settle_position": 0.017644706647843122,
      "settle_velocity": 0.017075833631679416,
      "shake_clusters_position": 0.019423504592850804,
      "shake_clusters_pre_force_velocity": 0.0178878391161561,
      "shake_clusters_velocity": 0.018277747090905905
    },
    "throughput": {
      "comparison_status": "not_reported_without_matching_runtime_manifest",
      "ns_per_day": 77.00646287623412,
      "openmm_ratio": null,
      "simulated_ns": 0.0003,
      "steps_per_second": 222.81962637799225
    }
  },
  "schema_version": "mlx_atomistic.dual_runtime_summary.v1"
}
```
<!-- dual-runtime-summary:end -->
