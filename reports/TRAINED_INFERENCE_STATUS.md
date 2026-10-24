# Trained online inference: verified 2026-10-24

Notebook 06 completed all three batch sizes in run
`20261024T082321Z-4a43ed`. Blocks reduced measured online latency versus HGAM
by **8.91% at the primary batch size 32**, **8.02% at 128**, and **10.46% at
512**, while passing numerical equivalence checks. The primary result is below
the predefined 10% practical target. The secondary batch-512 aggregate is
sensitive to one slow HGAM observation, detailed below.

These are measured trained-model results from one T4 session, on a resident
graph. They support a modest incremental benefit over existing HGAM for this
workload. They do not establish a cross-session, cross-model or cross-schema
performance claim.

## Preserved evidence and correctness

The archive contains **352 unchanged original files**: 16 root reports/logs,
including device telemetry, and 336 raw measurement sidecars. Their SHA-256
hashes and sizes are in
[PROVENANCE.json](colab/20261024T082321Z-4a43ed/PROVENANCE.json).
The source revision was clean `0d5720f970dd62f1470bb09da21019d5589b4326`.
All **171 tests passed on the T4**, and preflight passed.

The fixed notebook 05 checkpoint SHA-256 remains
`92d02f7d2007c0ab560629e9c4e0d48886a1d149930e68e494a861b85659e1d6`.
Its binary is preserved outside Git. Reconstruction verified original source,
dependency, input, cohort and feature-policy contracts. No training occurred
and no baseline was refitted. All 8,192 saved validation cases and 256 sampler
fingerprints replayed successfully; the maximum replay logit difference was
`2.384185791015625e-7`. Source validation ROC-AUC/AP were reproduced exactly.

Each size covered the same 8,192 selected validation occurrences, grouped into
256, 64 and 16 query batches. Twenty paired repetitions per query batch yielded
**6,720 paired trials**, with 20,160 complete online passes and 20,160 separate
stage replays. Five warmups per mode per query batch were outside these trials.
Mode order was randomized within paired trials.

All recorded occurrence audits passed. Real repeated global IDs appeared in
2/256 query batches at size 32, 5/64 at 128 and 12/16 at 512, with occurrences
kept separate. Maximum cross-mode logit and loss differences were
`9.5367431640625e-7` and `5.960464477539063e-8`, within the unchanged
`atol=1e-5`, `rtol=1e-4` tolerances. Equality is numerical, not bitwise.
The model-state digest before and after was unchanged:
`fe95a7aaabb1efa78de2e7324a2a3c5a4ae7739eb3b467651c8b9c218e1dc5d9`.

Independent review checked every sidecar file/hash/checksum, protocol, query
coverage, seed schedule, model digest and source checkpoint hash. Recomputed
summaries, including all 2,000 bootstrap draws, matched the exported report.
The runtime's audited assertions establish cross-mode/stage sampler equality;
raw sampler tensors are not exported for an independent tensor replay.

## Main online latency result

| Query batch | Reference median ms | HGAM median ms | Blocks median ms | Blocks reduction vs HGAM | Within-run 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| **32, primary** | 26.577 | 25.650 | 23.353 | **8.91%** | 8.47–9.36% |
| 128 | 27.093 | 26.150 | 23.852 | 8.02% | 7.07–8.98% |
| 512 | 29.179 | 28.102 | 25.864 | 10.46% | 7.00–15.05% |

The online boundary includes serial sampling, feature gathering, transfer,
feature/time encoders, message passing including block construction, and
readout. Graph/model setup, replay verification, correctness audits, reporting,
DataLoader scheduling and serialization are excluded. The one-second telemetry
observer ran during collection, with its overhead present.

The predefined reduction is `100 * (1 - summed blocks time / summed HGAM time)`.
It is not the ratio of the displayed medians. The 95% percentile intervals use
paired bootstrap resampling of whole query batches, retaining all repeats and
mode pairs. They describe within-run query-batch variation, not independent
GPU-session, device or training-seed uncertainty. Blocks reduced total online
time versus the full reference by 11.91%, 11.45% and 11.49%, respectively.

The primary batch-32 estimate and interval fall below the 10% practical target.
Keep that primary comparison when interpreting the investigation; the secondary
batch-512 point estimate does not replace it.

### Batch-512 sensitivity

HGAM had a **248.378 ms** observation at zero-based `batch_index=4`,
`repeat_index=13`, in
[the raw sidecar](colab/20261024T082321Z-4a43ed/batches/batch-000512-000004.json).
The next slowest HGAM observation at this size was 53.781 ms. Its cause is
unknown. The complete observation remains in the headline 10.46% result.

As an explicit sensitivity diagnostic, removing that entire paired trial
across all three modes gives **8.35%** reduction; the ratio of pooled medians
gives **7.96%**. Neither diagnostic replaces the predefined all-data estimator.
The 512 result has only 16 query-batch clusters and a wider interval, so the
exact secondary percentage needs independent reproduction.

## Throughput, stages and memory

| Batch | Reference queries/s | HGAM queries/s | Blocks queries/s | Blocks p95 batch ms |
| --- | ---: | ---: | ---: | ---: |
| 32 | 1,170 | 1,209 | 1,328 | 29.756 |
| 128 | 4,573 | 4,750 | 5,164 | 31.643 |
| 512 | 17,066 | 17,264 | 19,280 | 31.765 |

Throughput divides actual measured query occurrences by summed online time.
Blocks throughput was **14.52 times** higher at size 512 than 32. Grouping and
sampling differ across sizes, and sizes ran in ascending order, so this is a
workload-level scaling observation in one session. It does not establish
identical sampled graphs or controlled timing across independent sessions.

The consistent execution benefit lies in message passing. Separate synchronized
HGAM-to-block replay medians were 13.187→10.918 ms, 13.553→11.199 ms and
13.715→11.423 ms, respectively: approximately 17% reductions. Every one of the
336 query-batch message-passing medians favored blocks. Encoding stayed around
8.5–8.75 ms; sampling, gathering, transfer and readout showed no corresponding
consistent improvement. These stage medians must not be summed into online
latency or treated as exact proportions of its total.

| Batch | Peak allocated HGAM / blocks MiB | HGAM-to-block allocation difference | Peak reserved, all modes MiB |
| --- | ---: | ---: | ---: |
| 32 | 16.163 / 15.376 | 0.787 MiB | 30 |
| 128 | 17.940 / 16.393 | 1.547 MiB | 32 |
| 512 | 24.021 / 20.084 | 3.937 MiB | 40 |

These are PyTorch allocator maxima, with a 14.477 MiB model/baseline allocation.
Reserved memory includes the warm cache. Process-lifetime host peak RSS was
2.691 GiB. They are distinct from device-wide readings.

Of 1,440 one-second telemetry observations, 1,413 fell inside query-batch
collection windows. Device memory medians were **161, 163 and 169 MiB**, with
a sampled maximum of 171 MiB. GPU activity medians were **21%, 22% and 29%**;
sampled maxima were 24%, 25% and 33%. Median power was approximately 34.02,
34.42 and 35.28 W. Windows include warmups, online trials, audits and separate
stage replays, so these counters cannot be assigned to individual modes or
used as continuous peaks. They confirm a small memory footprint and modest
collection-period GPU activity; they do not identify a hardware bottleneck.

Collection windows totaled 1,425.94 seconds. Summed online timings over all
repeats/modes totaled 528.57 seconds. Neither is a single inference request or
the complete Colab session duration. The
[derived audit](analysis/20261024T082321Z-4a43ed-audit.json) records telemetry,
throughput scaling and the outlier sensitivity calculations.

## Budget, limits and next step

The original budget export transposed the starting balance and hourly rate,
and omitted the ending balance. The user explicitly confirmed **198.50 units
at start, 198.00 at end, and 1.07 units/hour**. The corrected **0.50-unit balance
decrease** is attributed using those user-reported run boundaries and the
exported isolation flag (`true`); it is not a raw exported cost calculation.
The ending balance leaves **148 units above reserve**. Original files remain
unchanged, with the correction in a
[separate budget record](budget/20261024-notebook06-observations.json).

Scope remains one T4 session, one checkpoint/training seed, one schema and
one October 1 validation date, with constant/time features and unresolved
exported-snapshot history assumptions. The activity baseline remains the
stronger predictor on the source validation cohort. This result concerns
execution efficiency of the fixed GNN.

The next step is a clean independent T4 reproduction with the same fixed
protocol and checkpoint, followed by the final report. Preserve the primary
batch-32 comparison, retain outliers transparently, and assess between-session
stability before generalizing. Additional model/configuration extensions are
separate research questions.
