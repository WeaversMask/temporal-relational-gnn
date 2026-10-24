# Trained inference investigation: independent reproduction, 2026-10-24

Subsequent [GPU diagnostic findings](SPIKE_DIAGNOSTIC_STATUS.md) directly observed
full Python collection inside similar quarter-second online pauses. The original
percentages below remain unchanged observations of this original protocol;
they are not corrected estimates from a collection-controlled benchmark.

The fixed trained model's execution benefit reproduced in a fresh T4 session:
blocks reduced graph-loaded online latency versus HGAM by **8.91% in the first
session and 8.77% in the second** at the predefined primary batch size 32.
Both estimates and their within-run intervals fall below the 10% practical
target. This supports a repeatable, modest improvement for this frozen workload;
it does not establish the target or a general performance claim.

The second run passed **171 T4 tests**, all 336 query batches and all 6,720
paired trials. The weights stayed unchanged and predictions matched within the
original numerical tolerances. No training or baseline refit occurred.

## Question, workload and preserved evidence

The question was whether destination-restricted bipartite blocks save complete
online inference time beyond existing PyG HGAM while preserving the same trained
GraphSAGE, features and temporal sampled occurrences. GraphSAGE, HGAM and block
execution are established methods; this project evaluates their execution
schedules and occurrence semantics, rather than claiming those methods as new.
See the [execution contract](../docs/EXECUTION.md) and
[fixed measurement protocol](../docs/TRAINED_INFERENCE.md).

Both runs used the notebook 05 checkpoint, constants plus relative time,
64 channels, two layers, fanouts `[16, 8]`, FP32 and the same 8,192 validation
occurrences. Each session measured sizes 32/128/512, with 256/64/16 query batches,
20 paired repetitions per batch and five untimed warmups per mode. Mode order
was randomized using the fixed seed schedule. Each session contains 20,160
whole online passes and 20,160 separate stage replays.

| Evidence | First session | Independent reproduction |
| --- | --- | --- |
| Run ID | `20261024T082321Z-4a43ed` | `20261024T102607Z-6593a4` |
| Clean source | `0d5720f970dd62f1470bb09da21019d5589b4326` | `b069b581d2ba0672e638411d497504ac4c002cca` |
| GPU identity | T4 UUID ending `ada7e1eab4f9` | T4 UUID ending `8e8eea01dd20` |
| Correctness suite | 171 passed | 171 passed |
| Complete fresh query batches | 336 | 336 |
| Raw archive | [First provenance](colab/20261024T082321Z-4a43ed/PROVENANCE.json) | [Repeat provenance](colab/20261024T102607Z-6593a4/PROVENANCE.json) |

The reproduction used `resume=false`, a new session ID and a different device
UUID; no prior measurement sidecars were reused. The complete benchmark
contracts match exactly, including source code hashes, checkpoint/input
contracts, runtime dependencies, hardware class and protocol. Both used
Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, driver 580.82.07 and one CPU
thread. Matching recorded hardware class does not imply identical host load.

The new archive preserves **352 unchanged original files**, including 336 raw
sidecars, totaling 53,044,131 bytes. File and canonical payload hashes,
query coverage, paired seeds and summaries were independently checked against
the export. The checkpoint is preserved outside Git; its SHA-256 remains
`92d02f7d2007c0ab560629e9c4e0d48886a1d149930e68e494a861b85659e1d6`.
Original raw exports from both sessions remain unchanged.

All 6,720 measured sampler fingerprints, sampled counts, visibility audits,
query indices, seeds and mode orders match between sessions, as do all 256
saved-validation replay fingerprints. The reproduction replayed all 8,192
saved predictions before timing, with a maximum logit difference of
`3.5762786865234375e-7` and exactly reproduced source ROC-AUC/AP. Maximum
cross-mode logit/loss differences were `9.5367431640625e-7` and
`5.960464477539063e-8`, within unchanged `atol=1e-5`, `rtol=1e-4` tolerances.
Equality is numerical, not bitwise. The before/after model-state digest remains
`fe95a7aaabb1efa78de2e7324a2a3c5a4ae7739eb3b467651c8b9c218e1dc5d9`.

## Primary and secondary latency results

| Batch size | First reduction vs HGAM | First within-run 95% interval | Repeat reduction vs HGAM | Repeat within-run 95% interval |
| --- | ---: | ---: | ---: | ---: |
| **32, primary** | **8.91%** | 8.47–9.36% | **8.77%** | 8.34–9.18% |
| 128 | 8.02% | 7.07–8.98% | 8.64% | 7.86–9.42% |
| 512 | 10.46% | 7.00–15.05% | 11.82% | 8.97–16.01% |

Reduction is `100 * (1 - summed blocks time / summed HGAM time)`, retaining
every trial. It is not calculated from the displayed latency medians below.
Intervals use 2,000 paired whole-query-batch bootstrap draws, preserving all
repeats and mode pairs. They quantify variation within each run. Two sessions
do not establish a population confidence interval across GPUs, hosts or dates;
no pooled cross-session interval is claimed.

The online timer covers serial sampling, gathering, transfer, feature/time
encoding, GNN execution including block construction, and readout on a resident
graph. Setup, graph construction, saved-validation replay, correctness audits,
reporting, DataLoader scheduling and serialization are excluded. Telemetry
observer overhead is present. These are conditional online pipeline timings,
rather than cold-start or serving-system latency.

| Repeat batch size | Reference median ms | HGAM median ms | Blocks median ms | Blocks p95 ms | Blocks queries/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 26.795 | 25.854 | 23.579 | 30.172 | 1,315 |
| 128 | 27.607 | 26.704 | 24.284 | 30.021 | 5,114 |
| 512 | 29.227 | 28.280 | 25.925 | 32.356 | 19,181 |

Blocks had lower median online latency than both baselines on all 336 query
batches in the reproduction. Against HGAM they won 4,342/5,120, 1,092/1,280 and
274/320 individual paired trials, respectively. Their aggregate reductions
versus the full reference were 11.75%, 12.14% and 11.10%. At size 512, HGAM had
a lower median than reference but slightly greater summed time; HGAM's p95 was
40.805 ms versus reference's 37.849 ms. A lower median alone does not establish
an aggregate improvement.

### Recurring secondary timing spike

The largest HGAM observation occurred at the same planned position in both
sessions: batch size 512, zero-based batch 4, repeat 13, sampler seed 2,000,135,
with mode order reference → HGAM → blocks. Its duration was **248.378 ms in the
first session and 252.555 ms in the reproduction**. The next slowest repeat-run
HGAM observation was 54.978 ms. The spike therefore recurred on a different T4
at the same workload/order position; it should not be dismissed as arbitrary
session noise. Its cause remains unresolved.

Both observations remain in the predefined all-data results. The first run's
explicit diagnostic excluding that entire paired trial gives 8.35% reduction.
The corresponding second-run diagnostic gives **9.73%**; its ratio of pooled
medians gives 8.33%. The
[reproduction audit](analysis/20261024T102607Z-6593a4-audit.json) records these
calculations. These diagnostics do not replace either headline estimate.
Separate synchronized replay did not reproduce the same
large duration and cannot identify which stage caused the uninstrumented spike.
Only 16 query-batch clusters contribute to each size-512 interval. Its exact
tail-sensitive benefit remains less settled than the primary comparison.

A subsequent [CPU lifecycle probe](analysis/512-fingerprint-gc-investigation.json)
established that the original fingerprint helper leaves collectible recursive
closure cycles. The completed [notebook 07 diagnostic](SPIKE_DIAGNOSTIC_STATUS.md)
then observed four full collections lasting 228–240 ms inside similarly slow
online passes at other modes/positions. The historical size-512 position was
ordinary in that run. Collection cadence is now a strong explanation of the
earlier events, while their exact attribution remains retrospective inference.
The short cycle-free control does not establish elimination of all long pauses.

## Why the complete saving is smaller than the GNN-stage saving

In the reproduction, HGAM-to-block message-passing replay medians were
13.347→11.112 ms, 13.689→11.353 ms and 13.836→11.478 ms: approximately 17%
reductions. Every query batch's median message-passing replay favored blocks.
Feature encoding remained around 8.6–8.7 ms; other stage differences were small
and mixed. The unchanged work limits the benefit seen in complete online
inference. Stage measurements are separate synchronized replays; their medians
must not be summed into online time or treated as exact fractions of it.

Batching changes throughput much more than execution mode in this workload:
blocks processed 1,315 versus 19,181 queries/s at sizes 32 and 512. The same
cohort is regrouped and sampler draws can differ across sizes, which ran in
ascending order. This is measured workload scaling. It does not establish a
comparison of identical sampled graphs or a universal batching factor.

## Memory, activity and cost

Allocator maxima match the first session exactly. Peak allocated HGAM/blocks
memory was 16.163/15.376 MiB, 17.940/16.393 MiB and 24.021/20.084 MiB at the
three sizes. Reserved memory was 30/32/40 MiB in all modes and includes the warm
cache. Host process-lifetime peak RSS was 2.650 GiB, including setup.

The repeat telemetry has 1,453 one-second observations; 1,430 fall inside
collection windows. Device-wide memory medians were 161/163/169 MiB, with a
sampled maximum of 171 MiB. GPU activity medians were 21%/22%/28%, with sampled
maxima 23%/26%/32%. Median power was 35.80/36.09/37.08 W. Windows include
warmup, audits and stage replays, so these observations cannot be attributed
to individual modes, treated as continuous peaks, or used to prove a hardware
bottleneck. They support the user's observation of low memory use.

Collection windows summed to 1,439.065 seconds, while summed online timing
across every mode and repeat was 533.708 seconds. Neither is a single request
or the complete session duration. The recorded isolated budget interval was
**197.90→197.46 units**, a **0.44-unit decrease**, at an observed
**1.07 units/hour**. The ending balance leaves **147.46 units above the
50-unit reserve**. These account readings do not assign a cost to individual
modes. See the [budget record](budget/20261024-notebook06-reproduction-observations.json).

## Conclusion and remaining scope

For this frozen trained temporal relational workload, the incremental online
saving beyond HGAM reproduced at **8.77–8.91%** in two fresh T4 sessions, with
numerically equivalent predictions and unchanged weights. The predefined 10%
primary practical target was not met. The stronger message-passing improvement
and small memory saving are useful secondary findings; the recurring size-512
tail requires explanation before making a specific tail-performance claim.

This completes the planned independent repeat of the bounded execution
comparison. Another unchanged GPU run is not required to document these
findings. Explaining the recurring spike would require a separately identified
trace experiment; model or schema extensions would be separate investigations.

The limits remain one trained checkpoint/seed, one schema, one validation date,
one fixed cohort and seed/order schedule, two T4 sessions, and constants/time
features under an exported-snapshot contract. Historical foreign-key changes,
merges, deletions and survivorship remain unresolved. On the source validation
cohort, the trained GNN's ROC-AUC/AP (0.83194/0.15439) were below the train-fitted
activity baseline's (0.85721/0.27248); the test split remains untouched. Execution
speed does not establish predictive superiority or resolve historical validity.
See the [training report](TRAINING_RUN_STATUS.md) and
[first-session report](TRAINED_INFERENCE_STATUS.md) for their original evidence.
