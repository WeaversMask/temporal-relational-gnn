# Natural-GC trained inference: independent reproduction

The version-2 primary result reproduces in a fresh T4 session: blocks reduced
summed online latency versus HGAM by **8.38% in the first run and 8.59% in the
repeat**, using the same fixed checkpoint, sampled inputs and measurement
contract. Each session passed **240 hosted T4 tests** and completed all
**336 query batches**, with 20 paired repeats each and no reused sidecars.

This completes the bounded version-2 execution comparison and its independent
repeat. No additional unchanged GPU run is required for this result. The work
still covers one checkpoint/training seed, schema, validation date and restricted
feature policy. Larger-workload or publication claims require separate evidence.

## Independent sessions and controlled inputs

| | First version-2 run | Independent repeat |
| --- | --- | --- |
| Run ID | `20261024T215147Z-bd9574` | `20261024T223434Z-dd7297` |
| Clean source commit | `56d82fb6d783a680e0cb122afd94c15ffc106368` | `0fde659166bd616c0b414da94e3dba586e358918` |
| Measurement session | `33bd9de14d6b4706941035a15517b2be` | `94ee97d2331e41b48356f18effa435d6` |
| T4 UUID | `GPU-6f4c0876-3702-8ced-2308-92df18f69053` | `GPU-ad92af85-4639-b8e4-50cc-956e943e68e4` |
| Completed cases / paired trials | 336 / 6,720 | 336 / 6,720 |
| Resumed / mixed-session | No / no | No / no |
| Hosted correctness suite | 240 passed | 240 passed |

The commits differ in archived results and documentation, including notebook
markdown; the executable notebook cells, measurement code and frozen settings
are unchanged. Both runs use benchmark contract SHA-256
`8863a37d502df4eb267e1bb3eeb3ce5561f0e92210d7cb6740df1098410f6b6f`.
The runtime observations match: Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0,
PyG 2.8.1, RelBench 3.0.2, PyTorch Frame 0.3.0, driver 580.82.07 and one CPU
thread. Full package observations remain in the original exports.

The original notebook 05 checkpoint SHA-256 is
`92d02f7d2007c0ab560629e9c4e0d48886a1d149930e68e494a861b85659e1d6`;
model-state SHA-256 remained
`fe95a7aaabb1efa78de2e7324a2a3c5a4ae7739eb3b467651c8b9c218e1dc5d9`
in both sessions. All 8,192 saved validation queries and 256 sampled fingerprints
replayed before each measurement. Every query plan, measured sampler seed,
execution order, sampled fingerprint/count and visibility audit must match the
fixed source and the first version-2 run. Predictions agree within the original
`atol=1e-5`, `rtol=1e-4`; numerical equality is not bitwise identity.

No training, baseline refit or test-query read occurred. The model remains the
same two-layer, 64-channel heterogeneous GraphSAGE with fanouts `[16, 8]`, mean
neighbor/sum relation aggregation, FP32 and constant/relative-time features.
Temporal occurrences are preserved separately, including repeated global IDs;
empty relationships retain their reference root/bias behavior.

## Results, retaining every observation

| Query batch | First reduction vs HGAM | First within-run 95% interval | Repeat reduction vs HGAM | Repeat within-run 95% interval |
| --- | ---: | ---: | ---: | ---: |
| **32, primary** | **8.38%** | **7.75–8.90%** | **8.59%** | **8.06–9.10%** |
| 128 | 8.12% | 7.25–8.94% | 8.42% | 7.56–9.30% |
| 512 | 8.21% | 6.41–9.93% | 7.65% | 2.07–11.49% |

The primary estimates differ by about **0.21 percentage points** and show a
consistent benefit across these two sessions. Both primary intervals lie above
zero and below the original 10% planning target; that target is not a scientific
validity or publication threshold. Secondary size 128 also agrees closely.
Size 512 remains less stable, with a substantially wider interval in the repeat.

The estimator is `100 * (1 - summed blocks online ms / summed HGAM online ms)`.
Each size uses all 8,192 queries over 20 repeats: 163,840 measured occurrences
per mode. Intervals use 2,000 paired whole-query-batch bootstrap draws, preserving
all repeats and modes. They measure within-session query-batch variation, not
cross-session variability. They remain separate; no pooled cross-session
confidence interval or statistical claim about their difference is made.

| Batch size | First blocks queries/s | Repeat blocks queries/s | First HGAM → blocks message-passing median ms | Repeat HGAM → blocks message-passing median ms |
| --- | ---: | ---: | ---: | ---: |
| 32 | 1,319 | 1,330 | 13.291 → 10.990 | 13.265 → 10.984 |
| 128 | 5,205 | 5,044 | 13.499 → 11.188 | 13.878 → 11.598 |
| 512 | 19,299 | 17,524 | 13.746 → 11.469 | 14.329 → 12.005 |

Separate message-passing replays show about **16.2–17.3%** median reduction across
these sessions/sizes. Unchanged sampling, gathering, transfer, encoders and readout
limit the broader online benefit. Stage medians come from separate synchronized
replays and must not be summed into online latency. Increasing batch size changes
grouping and sampler draws; its throughput scaling is not an execution-mode
speedup claim.

## Cleanup and the remaining size-512 tails

Automatic GC remains enabled at thresholds `[2000, 10, 10]`. Cycle-free
fingerprinting is byte-identical to the original sampler serialization, during
both saved-validation replay and measurement. Lightweight callbacks and exact
timer metadata remain present; their overhead is not subtracted. No collection
is explicitly forced between passes and no observation is filtered.

Each session observed 7,395 collections within case callbacks: 6,775 generation
0, 617 generation 1 and three generation 2. Each program's counter difference
records four full collections, including one outside the observed case windows.
Its separate duration is unknown, while program wall retains the cost. The
cycle-free helper therefore does not eliminate all full collections.

The repeat's observed full collections occurred inside these online passes;
indices are zero-based:

| Size / batch / repeat | Mode | Online duration ms | Full-GC overlap ms |
| --- | --- | ---: | ---: |
| 32 / 85 / 4 | Reference | 240.484 | 211.302 |
| 32 / 201 / 13 | Blocks | 236.064 | 212.815 |
| 128 / 23 / 4 | Reference | 242.635 | 214.237 |

All costs remain included, including the blocks pause. The first run instead
received two full-GC pauses on blocks and one on the reference. Collection
cadence can assign cleanup to different modes even with the same sampled inputs.

The repeat also has **six size-512 online observations exceeding 100 ms**, in
batches 10–11. None overlaps generation-2 collection. Four reference/HGAM passes
have only 1.324–7.632 ms of minor-GC overlap despite total durations of
113.398–140.693 ms. Two blocks passes (batch 11, repeats 8 and 16) take
117.935 and 120.677 ms with **zero GC overlap**. Their causes are unresolved;
these events cannot be explained by recorded full GC. They remain in the
headline estimator and bootstrap, alongside all surrounding observations.

These cases cluster around `23:05:05–23:05:19 UTC`. Several one-second GPU
activity samples fall to 7–15% while sampled memory remains 171 MiB; separate
stage replays slow across multiple stages and all modes. This identifies a
localized period of runtime variation, not its cause. Stage replays cannot
localize a previous online pause, and clocks, temperatures and CPU scheduling
were not traced. No hardware-throttling or specific CPU-bottleneck diagnosis
is established.

The old size-512 batch-4 repeat-13 HGAM position took **28.221 ms** in the first
version-2 run and **32.118 ms** in the repeat, each with less than 1 ms minor-GC
overlap. Neither repeats the historical roughly 250 ms pause. Historical
notebook 06 events have no collection traces, so their specific causes remain
unproved. The new non-GC tails further limit any claim that cleanup explains
all slow passes or that size-512 performance is uniformly stable.

| Accounting boundary | First run | Repeat |
| --- | ---: | ---: |
| Observed case GC duration | 5.511 s | 5.565 s |
| GC already overlapping online timers | 5.068 s | 5.115 s |
| GC elsewhere within observed cases | 0.443 s | 0.450 s |
| Collection-program wall | 1,461.210 s | 1,473.237 s |

These GC durations are parts of already measured time, never extra costs to add
again or amounts to subtract. Case callbacks exclude reporting/gaps; no missing
GC duration is invented. Collection-program wall includes warmups, audits,
stage replays, observer processing, serialization, summaries and monitor teardown
after setup/replay, excluding final summary serialization. It is not a serving
throughput denominator.

## Interpretation, resources and next steps

The bounded finding is a reproduced **roughly 8.4–8.6% primary online latency
reduction versus HGAM** for this fixed temporal relational model/workload, while
preserving numerical predictions and sampled temporal occurrences. The original
notebook 06 observations of 8.91%/8.77% remain separate old-protocol results;
GC-aware measurements do not retroactively alter them or isolate the causal
effect of changing the audit helper.

Maximum measured logit/loss errors remain `9.5367431640625e-7` and
`5.960464477539063e-8` across both sessions. The repeat's saved-prediction replay
maximum logit error is `4.76837158203125e-7`. Allocator maxima remain unchanged:
25.720 MiB reference, 24.021 MiB HGAM, 20.084 MiB blocks, and 40 MiB reserved.
Device-wide sampled memory and utilization are separate from allocator peaks;
one-second samples cannot identify the cause of individual timing spikes.
The repeat recorded 1,463 device samples, 1,439 matched to case windows.
Median utilization was 21%, 22% and 28% at sizes 32/128/512; sampled maximum
was 31%, with device memory 161–171 MiB. Process-lifetime peak host RSS was
2.684 GiB, including imports and graph preparation.

The repeat's isolated account interval was **194.54 to 194.15 units**:
**0.39 units consumed** at an observed **1.07 units/hour**, leaving
**144.15 units above the 50-unit reserve**. The earlier 194.63-to-194.54 gap
is a separate unattributed 0.09-unit interval. See the
[budget observations](budget/20261024-notebook08-reproduction-observations.json).

The [first run's report](GC_AWARE_INFERENCE_STATUS.md),
[first provenance](colab/20261024T215147Z-bd9574/PROVENANCE.json),
[repeat provenance](colab/20261024T223434Z-dd7297/PROVENANCE.json),
[first audit](analysis/20261024T215147Z-bd9574-audit.json) and
[repeat audit](analysis/20261024T223434Z-dd7297-audit.json) retain the evidence.
Both exports preserve all 16 root files unchanged and all 336 batch files as
byte-identical entries in deterministic compressed archives. Checkpoint binaries
remain outside Git.

Consolidate the bounded report before choosing a separate extension. A deliberate
second schema/task or deeper workload could test generality; it is not measured
here and no new GPU experiment is prepared automatically. The activity baseline
remains the stronger predictor, exported-snapshot historical assumptions remain
unresolved and the test split remains untouched. GraphSAGE, HGAM and block
execution are established methods; this project adapts and evaluates them.
