# DFT Material Validation

This is the committed scientific ledger for the periodic
`PeriodicDFTSystem`/`run_periodic_scf` path. It summarizes accepted generated
reports without promoting their gitignored raw artifacts or rerunning any DFT
calculation.

## Results

| Material and workload | Main result | Reference comparison | Status and boundary |
| --- | --- | --- | --- |
| Diamond Si, 8 atoms, PBE/GTH-q4, 25 Ha, 6×6×6 | `a₀ = 5.460859 Å`, `B₀ = 88.306 GPa`, `B₀′ = 4.3052`, Δ = `1.942 meV/atom` | All-electron PBE: lattice `0.166%`, bulk modulus `0.232%`, and `B₀′` `0.153%` relative error | Verified for this bulk-Si EOS workload. Broader chemistry and MLX-versus-QE PWscf parity remain proof-level or diagnostic. |
| Diamond Si frozen-density bands, Γ-X-W-K-Γ-L | Indirect gap `0.578 eV`, valence width `11.959 eV` | Published PBE/GTH ordering and high-symmetry energies; PBE gap is not compared with the experimental `1.12 eV` gap | Accepted band-path validation from a converged 6×6×6 density; not a general band-structure certification. |
| Diamond C, 8 atoms, PBE/GTH-q4, 40 Ha, 6×6×6, 7 volumes | `a₀ = 3.574441 Å`, `B₀ = 438.100 GPa`, Δ = `1.268 meV/atom` | All-electron PBE: lattice `0.075%` and bulk modulus `1.080%` relative error | Verified for the accepted 40 Ha workload. A 40→50 Ha central three-volume screen changed the curve by `0.438 meV/atom`; no full 50 Ha curve is required. |
| Rock-salt MgO, 8 atoms, PBE, Mg-q2/O-q6, 70 Ha, 6×6×6, 7 volumes | `a₀ = 4.259503 Å`, `B₀ = 146.914 GPa`, Δ = `1.060 meV/atom` | All-electron PBE: lattice `0.123%` and bulk modulus `1.387%` relative error | Core EOS properties validated. The strict whole-report gate remains failed because `B₀′ = 3.39995` differs from `4.09093` by `16.89%`, above the locked `15%` threshold; this is retained as a likely Mg-q2 transferability limit. |
| MgO periodic forces at the accepted EOS cell | 21 of 24 atom/axis comparisons pass `1e-4 Ha/bohr`; maximum deviation `2.246e-4 Ha/bohr` | Analytic force versus 48 reconverged ±0.01 bohr SCFs | Accepted with a known float32 total-energy precision limit. The three failures are O 6-x and O 7-y/z; the threshold was not weakened. |

## Evidence Boundary

The compact reference bundles are committed under
`src/mlx_atomistic/benchmarks/data/`; tests lock the fit and completion
semantics. Raw SCF, EOS, band, and force reports remain under gitignored
`results/`. Quantum ESPRESSO and CP2K are reference families only and are not
installed, imported, or executed by the MLX runtime or routine CI.

These rows prove the listed fixed workloads. They do not establish metallic or
spin-polarized chemistry, broad pseudopotential transferability, cell
relaxation, or universal periodic DFT accuracy.
