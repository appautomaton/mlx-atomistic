# DFT Material Validation

This is the committed scientific ledger for the periodic
`PeriodicDFTSystem`/`run_periodic_scf` path. It summarizes accepted generated
reports without promoting their gitignored raw artifacts or rerunning any DFT
calculation.

## Results

| Material and workload | Main result | Reference comparison | Status and boundary |
| --- | --- | --- | --- |
| Diamond Si, 8 atoms, PBE/GTH-q4, 25 Ha, 6×6×6 | `a₀ = 5.460859 Å`, `B₀ = 88.306 GPa`, `B₀′ = 4.3052`, Δ = `1.942 meV/atom` | All-electron PBE: lattice `0.166%`, bulk modulus `0.232%`, and `B₀′` `0.153%` relative error | Verified for this bulk-Si EOS workload. Broader chemistry and MLX-versus-QE PWscf parity remain proof-level or diagnostic. |
| Diamond Si fixed-cell relaxation, 8 atoms, PBE/GTH-q4, 25 Ha, 6×6×6 | The atom displaced by `(+0.04, -0.03, +0.02) Å` converged in 7 accepted steps; final maximum force `8.832e-5 Ha/bohr`, RMS force `2.437e-5 Ha/bohr`, and final maximum step `1.480e-3 bohr` | Translation-aligned maximum error from the source-defined ideal diamond positions: `0.000312 Å` | Verified for this fixed-cell relaxation. Exact restart equivalence is CPU-tested; the Metal workload verifies accepted-step checkpoint publication. This row alone does not establish variable-cell relaxation. |
| 2H-Si full-rank fixed-cell relaxation, 4 atoms, PBE/GTH-q4, 25 Ha, 6×6×4 | The ideal `P63/mmc` lonsdaleite prototype converged in 3 accepted steps; final energy `-15.76211219 Ha`, maximum force `1.715e-5 Ha/bohr`, and final maximum step `1.659e-4 bohr` | Source prototype `A_hP4_194_f-001`, `z = 1/16`; maximum translation-aligned internal relaxation `0.003351 Å` | Verified for this hexagonal fixed-cell workflow and complete-matrix identity. It does not establish stress, variable-cell relaxation, or broad Silicon polytype accuracy. |
| 2H-Si variable-cell relaxation, 4 atoms, PBE/GTH-q4, 25 Ha, 6×6×4 | Starting at `0.995` of the accepted scale, the cell accepted one step and converged at `0.9981142`; final pressure was `2.66553e-6 Ha/bohr³`, lattice error `0.1886%`, and relaxation wall time 22.707 s. | A fresh 35 Ha SCF gave `2.88255e-6 Ha/bohr³`; the `2.17019e-7 Ha/bohr³` drift passed the locked `5e-6` Pulay gate in 8.248 s. Complete validation wall was 30.956 s. | Current-verified on Apple M5 Max, AC power, normal power mode. Analytic electronic stress, float64 analytic Ewald stress, deterministic cell/coupled/restart gates, and the bounded material trajectory pass. This is not broad variable-cell transferability. |
| Diamond Si frozen-density bands, Γ-X-W-K-Γ-L | Indirect gap `0.578 eV`, valence width `11.959 eV` | Published PBE/GTH ordering and high-symmetry energies; PBE gap is not compared with the experimental `1.12 eV` gap | Accepted band-path validation from a converged 6×6×6 density; not a general band-structure certification. |
| Diamond Si Gamma phonons, 2-atom primitive, PBE/GTH-q4, 25 Ha, full 4×4×4 electronic mesh | At `0.01 bohr`, acoustic modes `3.739`, `4.269`, `7.517 cm⁻¹`; optical triplet `509.325`, `512.097`, `512.103 cm⁻¹`; no ASR correction | Quantum ESPRESSO tutorial optical context `516.175 cm⁻¹`; local mean error `4.999 cm⁻¹`. The `0.02→0.01 bohr` maximum drift is `2.379 cm⁻¹` | Current-verified on Apple M5 Max. Raw reciprocity, translational residual, acoustic overlap, stability, displacement convergence, reference, restart, wall, and memory gates pass. This is a bounded Gamma-point result, not phonon dispersion or LO-TO validation. |
| Diamond C, 8 atoms, PBE/GTH-q4, 40 Ha, 6×6×6, 7 volumes | `a₀ = 3.574441 Å`, `B₀ = 438.100 GPa`, Δ = `1.268 meV/atom` | All-electron PBE: lattice `0.075%` and bulk modulus `1.080%` relative error | Verified for the accepted 40 Ha workload. A 40→50 Ha central three-volume screen changed the curve by `0.438 meV/atom`; no full 50 Ha curve is required. |
| Rock-salt MgO, 8 atoms, PBE, Mg-q2/O-q6, 70 Ha, 6×6×6, 7 volumes | `a₀ = 4.259503 Å`, `B₀ = 146.914 GPa`, Δ = `1.060 meV/atom` | All-electron PBE: lattice `0.123%` and bulk modulus `1.387%` relative error | Core EOS properties validated. The strict whole-report gate remains failed because `B₀′ = 3.39995` differs from `4.09093` by `16.89%`, above the locked `15%` threshold; this is retained as a likely Mg-q2 transferability limit. |
| MgO periodic forces at the accepted EOS cell | 21 of 24 atom/axis comparisons pass `1e-4 Ha/bohr`; maximum deviation `2.246e-4 Ha/bohr` | Analytic force versus 48 reconverged ±0.01 bohr SCFs | Accepted with a known float32 total-energy precision limit. The three failures are O 6-x and O 7-y/z; the threshold was not weakened. |
| Rock-salt MgO primitive Mg-q10/O-q6 feasibility, 2 atoms, 40 Ha, density-reconstructed reduced 4×4×4 | Cold wall `39.892 s`; direct-Rayleigh seeded wall `42.780 s`; peak temporary memory `63.8 MB`; both reached 80 SCF iterations | Numerical diagnostic only; no EOS reference claim | Rejected: the cold direct orbital residual was `3.570e-6`; refreshing the direct Rayleigh quotient improved the seeded residual to `3.013e-6`, still above the locked `2e-6` gate. The matched full oracle and seven-point EOS were correctly skipped. |
| fcc Al, 4 atoms, PBE/GTH-q3, Fermi-Dirac `0.00225 Ha`, 15 Ha, reduced 15×15×15, 11 bands, 7 volumes | `a₀ = 4.039885 Å`, `B₀ = 76.631 GPa`, `B₀′ = 4.58384`, Δ = `0.230 meV/atom` | All-electron PBE: lattice `0.024%`, bulk modulus `1.137%`, and `B₀′` `0.851%` relative error | Verified for this metallic EOS workload. The accepted mesh has 120 weighted representatives; the result does not establish broad Aluminum chemistry or metal transferability. |
| bcc Fe primitive cell, spin-PBE/GTH-q16, Fermi-Dirac `0.01 Ha`, unfolded 4×4×4, 10 bands | The density-reconstructed reduced result gives `2.34075 μB/atom` in `11.375 s`; the matched full mesh gives `2.33818 μB/atom` in `53.104 s`. | Published PBE context `2.33 μB/atom`; the full-mesh moment error is `0.00818 μB/atom`. | Current-verified method oracle on Apple M5 Max, AC power, normal mode. Reduction changes free energy by `4.0505e-6 Ha/atom` and moment by `0.002563 μB/atom`, passing the locked `5e-5` and `0.02` gates. Older cutoff/k-point evidence still requires refreshed v2 identities. |

## Evidence Boundary

The compact reference bundles are committed under
`src/mlx_atomistic/benchmarks/data/`; tests lock the fit and completion
semantics. Raw SCF, EOS, band, and force reports remain under gitignored
`results/`. Quantum ESPRESSO and CP2K are reference families only and are not
installed, imported, or executed by the MLX runtime or routine CI.

These rows prove the listed fixed workloads. Aluminum closes one simple-metal
case and Iron closes one collinear-magnetic q16 case, but the rows do not
establish broad metallic, magnetic, or pseudopotential transferability. The
retained Fe q8 failure makes that boundary explicit. The 2H-Si row closes one bounded
variable-cell path; it does not establish universal periodic DFT accuracy or
broad cell-relaxation transferability.

The periodic runtime has a numerically validated Fermi-Dirac occupation and
free-energy path, including weighted k-points, odd electron counts,
checkpoints, and analytic-force occupation propagation. The Aluminum row adds
a source-bound material validation for that path. Capability and broader
material accuracy remain separate claims.
