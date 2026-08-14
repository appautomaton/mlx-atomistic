---
title: "Retained-stack Phase 5 confirmation on Apple M5 Max"
---

Phase 5 confirms the deferred exact-pair MLX/Metal molecular-dynamics route
across 5DFR and JAC, records why the next atomic-reduction Metal kernel is not
worth implementing, and refreshes the manifest-bound OpenMM reference context.

## Result

| Evidence | Result |
| --- | --- |
| 5DFR clean 75-step tile median | 0.104837 s; all profile checks passed |
| 5DFR sustained | 750 steps twice; 1.626131 s and 1.456710 s; both memory plateaus passed |
| JAC sustained | 750 steps twice plus 1,500 steps; all science checks passed; 45-rebuild memory trace settled near 4.68 GB |
| Diagnostic exact pairs | Never materialized; zero compact-pair bytes |
| Current MLX/OpenMM JAC ratio | `5.2701x` by two-run medians, down from historical `9.7586x` |

The synchronized 5DFR profile assigns 15.90% to direct spatial tiles, 15.59%
to integration and thermostat, 8.91% to reciprocal Particle Mesh Ewald (PME),
8.85% to neighbor work, and 8.26% to force aggregation. Combined SETTLE and
SHAKE constraint routes account for 27.44%. These are instrumented ordering
diagnostics, not uninstrumented wall shares.

## Next-kernel decision

Packing four force groups exposes repeated left blocks, but right blocks almost
never repeat. A bounded cross-group reducer can eliminate at most 15.19% of all
endpoint atomic writes. Since all atomic writes account for only about 15% of
direct-kernel work, the ideal bound is roughly 2.3% of the direct kernel before
new barriers and scratch. The candidate is therefore `no-go` without code.

The two Phase 2 density-functional theory (DFT) Metal candidates were removed,
so no new Carbon or MgO transfer claim is made. The next DFT target is FFT work
volume and shape scheduling, not another narrow boundary kernel.

Raw reports remain gitignored under `results/md-post-pair-elision/`.
