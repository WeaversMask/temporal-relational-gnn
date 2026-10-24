# Natural-GC trained inference: first revised run

Notebook 08 completed on 2026-10-24 UTC in run `20261024T215147Z-bd9574`,
at clean source `56d82fb6d783a680e0cb122afd94c15ffc106368`. All **240 tests
passed on the Tesla T4**, including the revised executor's CUDA check and the
compiled temporal sampler. All **336 query batches** completed 20 paired repeats
in one fresh session, with no reused measurement sidecars. This is the first
hosted version-2 measurement. Its independent repeat has since completed,
with a primary reduction of 8.59%; see the
[two-run report](GC_AWARE_INFERENCE_REPRODUCTION.md). The sections below retain
the first run's measurements.

The primary blocks-versus-HGAM online latency reduction is **8.38%**. All three
sizes show a positive reduction with their within-run intervals above zero.
The two historical notebook 06 sessions remain separate old-protocol results;
this run does not replace their raw measurements or supply a pooled interval.

## Measured online results

| Query batch size | Query batches | Blocks reduction vs HGAM | Within-run 95% interval | Blocks reduction vs reference | Blocks queries/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| **32, primary** | 256 | **8.38%** | **7.75–8.90%** | 11.55% | 1,319 |
| 128 | 64 | 8.12% | 7.25–8.94% | 12.18% | 5,205 |
| 512 | 16 | 8.21% | 6.41–9.93% | 12.16% | 19,299 |

Each size covers the same selected 8,192 validation queries, repeated 20 times:
163,840 measured query occurrences per mode and size. There are 6,720 paired
trials and 20,160 online mode executions overall. Reductions use
`100 * (1 - sum(blocks online ms) / sum(HGAM online ms))`, retaining every
observation. Throughput divides measured query occurrences by summed online
seconds. The 2,000-draw paired bootstrap resamples whole query batches while
preserving their repeats and modes. Its intervals describe query-batch variation
within this session; they do not measure cross-session or training variability.

| Size | Reference median ms | HGAM median ms | Blocks median ms | HGAM p95 ms | Blocks p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 26.709 | 25.709 | 23.457 | 32.381 | 29.491 |
| 128 | 26.978 | 26.093 | 23.829 | 31.694 | 29.812 |
| 512 | 29.168 | 28.134 | 25.674 | 34.489 | 31.869 |

These medians explain typical batch duration; ratios of these medians are not
the headline estimator. Separate synchronized message-passing replays show
approximately 16.6–17.3% lower median duration for blocks versus HGAM. Sampling,
gathering, transfer, encoding and readout reduce that advantage across the whole
online boundary. Separate stage medians must not be summed into online latency.

## Fixed model, inputs and equivalence

The run used the original notebook 05 `training.pt`, SHA-256
`92d02f7d2007c0ab560629e9c4e0d48886a1d149930e68e494a861b85659e1d6`,
with no training, baseline refit or test-query read. The unchanged model-state
hash is `fe95a7aaabb1efa78de2e7324a2a3c5a4ae7739eb3b467651c8b9c218e1dc5d9`.
All saved validation predictions and 256 sampled fingerprints replayed; maximum
saved-logit error was `2.384185791015625e-7`. Across measured modes/sizes, maximum
logit error was `9.5367431640625e-7` and maximum loss error was
`5.960464477539063e-8`, within the original `atol=1e-5`, `rtol=1e-4`.

The model remains two-layer, 64-channel heterogeneous GraphSAGE on
`rel-stack/user-badge`, with fanouts `[16, 8]`, mean neighbor/sum relation
aggregation, FP32 and constant/relative-time features. Temporal per-query
occurrences, empty-relation contributions, sampled counts, query indices,
seeds, execution orders and visibility audits remain part of the contract.
Every measured case is bound to the archived notebook 06 source cohort.

The version-2 contract digest is
`8863a37d502df4eb267e1bb3eeb3ce5561f0e92210d7cb6740df1098410f6b6f`.
It records cycle-free, byte-identical fingerprinting during replay and
measurement, automatic GC enabled, thresholds `[2000, 10, 10]`, and hashes of
the new measurement implementation. The old checkpoint-bound source remains
unchanged. The source contract comparison recorded no differences.

## Cleanup pauses remain visible and included

This run does **not** establish elimination of full garbage collections. Case
callbacks observed **7,395 collections**: 6,775 generation 0, 617 generation 1,
and three generation 2. The three full collections occurred inside these online
passes; batch/repeat indices below are zero-based.

| Size / batch / repeat | Mode | Online duration ms | Full-GC overlap ms |
| --- | --- | ---: | ---: |
| 32 / 84 / 15 | Blocks | 242.118 | 217.209 |
| 32 / 201 / 13 | Blocks | 232.779 | 209.390 |
| 128 / 23 / 4 | Reference | 231.613 | 204.681 |

Those are exactly the three online observations exceeding 100 ms. Their cleanup
cost stays in the estimator, including the two pauses received by blocks.
Other slower passes remain unexplained by GC: the largest zero-overlap pass was
66.031 ms (reference, size 32, batch 2, repeat 2). No observation is removed.

At the historical size-512 batch-4 repeat-13 position, HGAM took **28.221 ms**
with **0.735 ms** minor-collection overlap. No size-512 online pass exceeded
100 ms; its largest was 53.620 ms. The old two roughly 250 ms HGAM observations
remain untraced historical events. This run is consistent with changed collection
cadence, but cannot retrospectively prove their cause or guarantee future tails.

Observed case GC durations total **5.511 seconds**: **5.068 seconds** overlap
online timers and **0.443 seconds** lie elsewhere within observed cases. These
are portions of already measured time and must not be added or subtracted again.
Program-boundary GC statistics record **four** full collections, one more than
the three inside callback-covered cases. One full collection therefore occurred
outside the observed case windows; its duration was not separately recorded.
It remains inside the **1,461.210-second collection-program wall time**.

That program wall includes warmups, correctness audits, stage replays, observer
processing, progress serialization, summaries and monitor teardown after setup
and saved-validation replay. It excludes final summary serialization and is not
a serving-throughput denominator. Case callbacks exclude reporting gaps; no
collection duration is invented for those gaps. Minimal observer overhead is
present and can change allocation/collection cadence. Automatic GC stays enabled,
no forced collection is used, and no GC cost or slow trial is subtracted.

## Historical comparison and interpretation

| Protocol/session | Primary size 32 reduction | Size 128 | Size 512 |
| --- | ---: | ---: | ---: |
| Original notebook 06, first | 8.91% | 8.02% | 10.46% |
| Original notebook 06, repeat | 8.77% | 8.64% | 11.82% |
| Revised notebook 08, first | 8.38% | 8.12% | 8.21% |

The broad benefit remains about 8% in this first revised run. The new size-512
estimate is lower than the old spike-influenced estimates. Session variation
and the protocol change prevent treating their difference as an isolated causal
effect of the hash correction. The predeclared 10% planning target remains
unmet; it is not a publication or scientific-validity threshold.

The independent fresh T4 repeat of the **same notebook 08** is now verified,
with no reused sidecars, unchanged contract and a different physical GPU.
The primary result is 8.38%/8.59% across these two sessions. Size 512 has wider
uncertainty in the repeat, including non-GC tails. See the
[two-run report](GC_AWARE_INFERENCE_REPRODUCTION.md) before making broader claims.
No further unchanged GPU run is planned; the
[run instructions](../docs/GC_AWARE_INFERENCE.md) remain available for deliberate
reproduction.

## Resources, evidence and scope

The runtime recorded Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, PyG 2.8.1,
RelBench 3.0.2, PyTorch Frame 0.3.0 and driver 580.82.07, using one CPU thread.
All package observations remain in the original environment/preflight reports.

The device-wide telemetry contains 1,451 one-second samples, 1,422 matched to
observed case windows. Median/p95/maximum GPU utilization was 21/22/24% at size
32, 22/24/26% at 128 and 29/31/32% at 512. Sampled device memory ranged from
161 to 171 MiB. These observations include driver/other-process memory, miss
short events and do not identify a particular execution mode or bottleneck.
Maximum PyTorch allocated memory was 25.720 MiB reference, 24.021 MiB HGAM
and 20.084 MiB blocks; maximum reserved memory was 40 MiB. Process-lifetime
peak host RSS was 2.671 GiB, including imports and graph preparation. Allocator
peaks, sampled total device memory and host RSS describe different resources.

The isolated exported account interval was **195.07 to 194.63 units**, consuming
**0.44 units** at an observed **1.07 units/hour**. The ending balance leaves
**144.63 units above the 50-unit reserve**. The earlier 195.36-to-195.07 account
gap is separate and unattributed. See the
[budget record](budget/20261024-notebook08-observations.json).

The [archive](colab/20261024T215147Z-bd9574/PROVENANCE.json) preserves 16 unchanged
root files and 336 byte-identical batch files in a deterministic compressed
archive, with individual hashes and sizes. The checkpoint stays outside Git.
The [independent audit](analysis/20261024T215147Z-bd9574-audit.json) checks raw
integrity, fixed paired inputs, summaries, intervals and collection accounting.

This remains one checkpoint/training seed, schema, validation date and restricted
feature policy. The stronger activity baseline, unresolved exported-snapshot
assumptions and untouched test split remain explicit. GraphSAGE, HGAM and block
execution are established methods; the findings concern their implementation
and evaluation in this workload.
