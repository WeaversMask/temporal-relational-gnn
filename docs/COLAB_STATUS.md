# Colab feasibility and inference: verified 2026-10-25

Notebook 08's first revised trained-inference run completed at clean source
`56d82fb`, with **240 hosted T4 tests passed** and all **336 query batches**.
The original checkpoint, saved predictions and sampled inputs matched. Blocks
reduced summed online latency versus HGAM by **8.38% at batch 32** (primary,
within-run 95% interval **7.75–8.90%**), **8.12% at 128** and **8.21% at 512**.
Independent repetition of this version-2 measurement remains pending.

Natural cleanup stays enabled and its actual online cost remains in every
observation. Three observed full collections landed inside online passes;
program counters record one more outside case windows, without a separately
measured duration. The old size-512 spike position was ordinary in this run.
See the [new measured result](../reports/GC_AWARE_INFERENCE_STATUS.md),
[completed diagnostic](../reports/SPIKE_DIAGNOSTIC_STATUS.md),
[original two-run report](../reports/TRAINED_INFERENCE_REPRODUCTION.md) and
[current steps](NEXT_RUN.md). Historical input assumptions and the stronger
activity baseline remain part of the scope.

## Initial dependency and synthetic checks: 2026-10-23

Two independent exported runs are preserved verbatim with SHA-256 manifests in
`reports/colab/`. The second export includes the subsequently completed sampler
check from the same runtime. Original project revision: `4199f13`.

- CPU-only runtime: 17 tests passed; one CUDA test skipped.
- T4 runtime: 18 tests passed, including CUDA seed-output equivalence.
- Actual PyG temporal disjoint sampler: passed the small two-query visibility test.
- Pinned database/task manifests: retrieved successfully.
- GPU stack: Python 3.13.16, PyTorch 2.11.0+cu130, PyG 2.8.1,
  `pyg_lib` 0.9.0+pt211cu130, PyTorch Frame 0.3.0, RelBench 3.0.2.
- Reported Colab package versions are observations, not an independently tested
  environment installation lockfile.
- Compute-unit consumption was not recorded; the budget values are null.
- Drive mounting failed; the user completed checks with temporary storage and
  exported the results manually.

On the larger synthetic T4 batch, the median core latency was 5.363 ms for the
reference, 4.775 ms for HGAM, and 4.383 ms for blocks. Blocks reduced median
latency by 8.2% and extra peak allocated memory by 93.3% versus HGAM in this one
configuration. These are untrained synthetic message-passing measurements,
excluding sampling, transfer, encoders, and readout. Extra allocation is not total
VRAM. They do not establish real-data or end-to-end improvement.

## Hosted relational batches: completed 2026-10-23 UTC

Notebook 02 passed in the user-exported run `20261023T223033Z-4cb94d`, recorded
at project revision `31d6b15910cb537d8f8024d11e97fa87d3fd8d34` with a clean
working tree. All seven original files are archived verbatim with SHA-256
hashes in [`reports/colab/20261023T223033Z-4cb94d/`](../reports/colab/20261023T223033Z-4cb94d/).
The correctness suite passed 26 tests on the T4, including CUDA equivalence,
and the compiled sampler preflight passed.

The pinned `rel-stack` database, schema/task manifests, and validation labels
downloaded successfully. The projected historical graph retained seven tables,
22 directed relationships, and 4,121,747 rows at the maximum selected query
timestamp, `2020-10-01`. Three batches of eight queries passed eligibility,
sampled time visibility, disjoint ownership, BFS prefix, logit, and loss checks.
The largest block logit error was `1.1920928955078125e-7`; HGAM matched exactly.

| Batch | Reference ms | HGAM ms | Blocks ms | Blocks latency reduction vs HGAM |
| --- | ---: | ---: | ---: | ---: |
| 1 | 21.268 | 20.138 | 18.292 | 9.17% |
| 2 | 21.684 | 21.050 | 18.349 | 12.83% |
| 3 | 20.037 | 19.207 | 17.276 | 10.05% |

These are ten-repetition forward medians from one untrained, constant-feature
model with relative-time encoding, including the actual RelBench encoders and
readout. The sampled batches contained only 60, 130, and 68 nodes. Extra peak
allocated CUDA memory fell by 19.37%, 38.26%, and 36.29% versus HGAM, which
corresponds to small absolute savings of 18.5, 88, and 45 KiB. These are not
total-VRAM savings. No repeated global IDs occurred in the measured batches.

Timing covers the forward pass on identical already sampled/transferred batches.
Neighbor sampling and gathering are recorded separately, and this is explicitly
not a complete inference throughput result. Transfer and peak host RAM were
unmeasured in this notebook 02 export, and its budget observations were null.
The later notebook 03 run below adds the pipeline and memory measurements.

## Hosted feature-policy audit and pipeline profile: completed 2026-10-23 UTC

Notebook 03 passed in run `20261023T225501Z-efdf72`, at clean project revision
`c3f5f6e5e64d13dc3fdfb57abdbfd704f0078ed7`. All eleven exported source files
are preserved verbatim with [SHA-256 hashes](../reports/colab/20261023T225501Z-efdf72/PROVENANCE.json).
All 36 tests passed on the Tesla T4, including CUDA and compiled sampler checks.
The runtime also confirmed compatibility with pandas 2.2.3 and PyArrow 23.0.1.

All three paired batches passed the sampler fingerprint, visibility, ownership,
BFS prefix, logit and loss checks over ten trials each. The maximum block logit
error remained `1.1920928955078125e-7`; HGAM matched exactly.

| Batch | Reference ms | HGAM ms | Blocks ms | Blocks reduction vs HGAM |
| --- | ---: | ---: | ---: | ---: |
| 1 | 28.098 | 27.662 | 24.430 | 11.68% |
| 2 | 26.904 | 27.712 | 24.661 | 11.01% |
| 3 | 27.230 | 25.233 | 22.953 | 9.04% |

These medians cover serial sampling, gathering, transfer, encoding, message
passing and readout from an already built graph with a resident model. Setup,
DataLoader scheduling and serialization are excluded. Graph construction took
54.509 seconds. Peak process-lifetime host RSS was 2.0415 GiB; allocated CUDA
peaks were 14.564–14.755 MiB, which are not total VRAM. Stage costs come from
separate synchronized replays, not parts whose medians can be summed into the
measured pipeline total. Raw outliers and precise boundaries are documented in
[the detailed report](../reports/PIPELINE_PROFILE_STATUS.md).

The feature-policy metadata audit passed, approving only constants and relative
record time; `training_ready` remains `false`. All 24 queries share the timestamp
`2020-10-01`, and all repeated-global-ID counts were zero across the 30 trials.
These remain preliminary small-workload, untrained results. Historical feature
and relationship semantics, the representative frozen workload, training and
predictive evaluation remain pending.

The observed rate was 1.07 Colab units/hour and the initial remaining balance
was 198.78 units. The user later reported 198.67 remaining and a notebook 02
runtime left active in the background. The 0.11-unit account decrease covers
the observed interval, rather than an isolated notebook 03 cost. The user
believes the session rate remained 1.07 units/hour; a combined rate is
unmeasured. See the [budget follow-up](../reports/budget/20261024-notebook03-followup.json).
That earlier balance left 148.67 units above the planned 50-unit reserve.

## Hosted reference-training pilot: completed 2026-10-24 UTC

Notebook 04 passed in run `20261024T051434Z-948a5d`, at clean source revision
`78feba6ba26622b77e1530ae3ae93315a7857109`. All 69 tests passed on the Tesla
T4. Thirteen original report/configuration/log files are archived verbatim with
[SHA-256 provenance](../reports/colab/20261024T051434Z-948a5d/PROVENANCE.json).
The checkpoint binary stays outside Git; its hash and size are recorded there.

All 100 optimizer updates completed with finite gradients and passed temporal
occurrence audits. The 3,200-query training cohort covered 39 timestamps and
3,177 unique users; repeated global post IDs occurred in two training updates
(three extra occurrences total). The final one-batch comparison had zero logit
and loss differences across reference, HGAM and blocks. Trained inference
speedup was not measured.

| Predictor on the same 512-query validation subset | ROC-AUC | Average precision | BCE |
| --- | ---: | ---: | ---: |
| Initial GNN | 0.47807 | 0.03259 | 1.02604 |
| GNN after 100 updates | 0.85419 | 0.13373 | 0.13122 |
| Historical-activity baseline | 0.85585 | 0.14655 | 0.12523 |
| Constant training-prevalence baseline | 0.50000 | 0.03320 | 0.14812 |

Only 17 validation examples were positive. The activity baseline's slightly
better numerical metrics do not establish statistical superiority, and these
subset results are not finished predictive quality. Inputs and snapshot
assumptions stayed unchanged; broader historical feature approval, final
workload selection and full evaluation remain pending.

The exported start/end balance and hourly rate fields were null despite an
`isolated_runtime_confirmed: true` flag. A later user follow-up reported 198.63
units remaining. The decrease from the prior 198.67 observation is 0.04 units
over that interval and does not establish isolated notebook 04 cost. The
[separate follow-up](../reports/budget/20261024-notebook04-followup.json) preserves
those distinctions without rewriting the export. That earlier balance
left 148.63 units above the 50-unit reserve.

## Hosted larger training run: completed 2026-10-24 UTC

[Notebook 05](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/05_training_run.ipynb)
passed in run `20261024T073010Z-1f9800` at clean source commit
`c9879d3bcb5cd76814bd9bc088a9ea4481aa0a40`. All **81 tests passed on T4**.
All 64,000 selected queries were consumed once across 2,000 finite-gradient
updates with passed occurrence audits. Eighteen training batches had repeated
global IDs, which remained distinct occurrences. The checkpoint is preserved
outside Git, with [export hashes](../reports/colab/20261024T073010Z-1f9800/PROVENANCE.json).

| Same 8,192-query validation cohort, 229 positives | ROC-AUC | AP | BCE |
| --- | ---: | ---: | ---: |
| Trained GNN | 0.83194 | 0.15439 | 0.11392 |
| Historical-activity baseline | 0.85721 | 0.27248 | 0.09880 |

The activity baseline scored better on all three metrics. Different cohorts
prevent interpreting the earlier pilot's scores as a controlled quality trend.
One trained validation batch again passed exact three-mode logits/loss parity;
trained inference speed remains unmeasured.

Training-call wall time was 214.262 seconds, with 185.104 seconds of timed
work. Graph construction took 56.046 seconds separately. Training allocator
peaks were 53.742 MiB allocated and 58 MiB reserved; process-lifetime host
peak RSS reached 2.481 GiB. Memory headroom does not establish GPU compute
utilization, and allocator figures are not total device-memory use.

The export recorded 198.63 to 198.59 units remaining at 1.07 units/hour.
Isolation was not confirmed, so the 0.04-unit account interval remains
unattributed. That earlier balance left 148.59 units above the reserve.
See the [full training report](../reports/TRAINING_RUN_STATUS.md),
[budget summary](../reports/budget/20261024-notebook05-observations.json), and
[next trained-inference requirements](NEXT_RUN.md).

## Hosted trained inference comparison: completed 2026-10-24 UTC

[Notebook 06](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/06_trained_inference.ipynb)
passed in run `20261024T082321Z-4a43ed` at clean source revision
`0d5720f970dd62f1470bb09da21019d5589b4326`, with **171 T4 tests passed**.
All 8,192 saved validation cases replayed before measurement. The checkpoint
and model state stayed unchanged; no training or baseline refit occurred.
All 336 query batches and 20 paired repetitions each completed, with passed
occurrence audits and numerical equivalence at the original tolerances. The
archive preserves 352 original files with
[SHA-256 provenance](../reports/colab/20261024T082321Z-4a43ed/PROVENANCE.json).

| Query batch | Blocks reduction vs HGAM | Within-run 95% interval | Blocks queries/s |
| --- | ---: | ---: | ---: |
| **32, primary** | **8.91%** | 8.47–9.36% | 1,328 |
| 128 | 8.02% | 7.07–8.98% | 5,164 |
| 512 | 10.46% | 7.00–15.05% | 19,280 |

The primary estimate and interval are below the predefined 10% practical
target. One slow HGAM trial influences the secondary batch-512 result; removing
the entire paired trial as a sensitivity diagnostic gives 8.35%. The headline
retains all data. These reductions use summed online times, rather than a ratio
of medians, and the intervals describe query-batch variation within this run.

The graph-loaded online boundary covers serial sampling, gathering, transfer,
encoders, message passing including block construction, and readout. Setup,
audits, reporting and DataLoader scheduling are excluded. Throughput rose
14.52-fold from batch 32 to 512 for blocks, with different grouping and sampler
draws across sizes. One-second device memory samples were 161–171 MiB;
these are separate from allocator peaks and cannot identify a hardware
bottleneck. Equality is numerical, not bitwise. See the
[full report](../reports/TRAINED_INFERENCE_STATUS.md) for stage evidence,
memory definitions, outlier sensitivity and limitations.

The raw budget export swapped the apparent start balance and hourly rate and
omitted the ending balance. The user explicitly confirmed **198.50 at start,
198.00 at end and 1.07 units/hour**. The 0.50-unit decrease is attributed using
those user-reported boundaries and the exported isolation flag. That earlier
balance left **148 units above reserve**. Originals remain unchanged; the
[separate corrected record](../reports/budget/20261024-notebook06-observations.json)
preserves their provenance.

## Independent trained inference reproduction: completed 2026-10-24 UTC

Notebook 06 passed again in run `20261024T102607Z-6593a4` at clean source
revision `b069b581d2ba0672e638411d497504ac4c002cca`. All **171 T4 tests passed**.
The fresh run reused no measurement sidecars, used the same frozen benchmark
contract/checkpoint and ran on a different physical T4 UUID. All 336 query
batches completed 20 paired repetitions each. Source validation replay passed,
weights stayed fixed and all occurrence/numerical equivalence checks passed.
Its [archive](../reports/colab/20261024T102607Z-6593a4/PROVENANCE.json) preserves
352 original files with hashes and sizes.

| Query batch | Blocks reduction vs HGAM | Within-run 95% interval | Blocks queries/s |
| --- | ---: | ---: | ---: |
| **32, primary** | **8.77%** | 8.34–9.18% | 1,315 |
| 128 | 8.64% | 7.86–9.42% | 5,114 |
| 512 | 11.82% | 8.97–16.01% | 19,181 |

The primary reduction reproduces the first run's 8.91% benefit; both point
estimates and within-run intervals remain below 10%. Every second-run
query-batch median favored blocks over both HGAM and the reference. Separate
message-passing replays show roughly 17% improvement, reduced to roughly 9%
over the broader online boundary.

The secondary batch-512 HGAM extreme recurred at the same zero-based query
batch 4, repeat 13: 248.378 ms in the first session and 252.555 ms in the
second. Its cause is unresolved. Keep every trial in headline results and
report sensitivity separately; recurrence alone does not identify its cause.
The [two-run report](../reports/TRAINED_INFERENCE_REPRODUCTION.md) records the
comparison and remaining limitations. Confidence intervals stay per-run;
no pooled cross-session confidence interval is claimed.

The reproduction export recorded **197.90 to 197.46 units** at **1.07 units/hour**
with isolation confirmed, yielding **0.44 units consumed** and **147.46 above
reserve**. Its [budget summary](../reports/budget/20261024-notebook06-reproduction-observations.json)
retains original observations without changing the files.

The bounded execution comparison and its report are complete, keeping the
stronger predictive activity baseline and unresolved exported-snapshot
assumptions visible. Two sessions still cover one checkpoint,
training seed, schema, validation date and constant/time feature policy. The
same [fixed protocol](TRAINED_INFERENCE.md) remains available for deliberate
reproduction; there is no planned automatic third run or configuration expansion.

## Follow-up diagnostic: completed 2026-10-24 UTC

[Notebook 07](SPIKE_DIAGNOSTIC.md) passed in run `20261024T121852Z-4747d5`
at clean source `da47db9d008d63046c5c35559765b199d455c174`, with **200 tests
passed in the T4 runtime**. All 332 cases completed: 326 original-history
cases, three forced-collection controls and three cycle-free-hash controls.
The original weights, benchmark contract and archived sampled-input
fingerprints matched. No training occurred. The archive preserves 16 unchanged
root files and 664 diagnostic case files; the latter are byte-identical entries
in a deterministic compressed archive, with all hashes and sizes in
[provenance](../reports/colab/20261024T121852Z-4747d5/PROVENANCE.json).

Four new automatic-condition slow online passes across different modes and
sizes took **251.483–266.993 ms**, with **227.667–239.758 ms** generation-2
collection overlap. The historical size-512 batch-4 repeat-13 HGAM position
instead took **28.770 ms with zero GC overlap**. Collection directly explains
most extra time in the new traced passes and is strongly consistent with the
old spike pattern. The old runs contain no event traces, so their specific
causes remain unproved. Two early 108.55/133.98 ms passes had no collection;
not every pause is a GC pause.

Both the original-hashing automatic target window (512 batches 3–5) and the
short cycle-free automatic control had zero generation-2 collections. These
controls cannot show that changing the helper fixed the spike; other cycles
remain. The forced condition performed **405 full collections outside executor
timers**, costing **88.656 seconds**. It relocated cleanup cost rather than
removing it, and supplies no free performance improvement. All instrumented
timings remain separate from the old 8.77–8.91% primary observations.

The [diagnostic report](../reports/SPIKE_DIAGNOSTIC_STATUS.md) records evidence
and limits. This diagnostic led to the revised
[notebook 08 protocol](GC_AWARE_INFERENCE.md), with cycle-free replay/auditing,
natural cleanup retained in actual timings, and separate collection-overlap and
whole-program wall accounting. Its first hosted measurement is recorded below.

Budget exports recorded **195.94 to 195.36 units**, an isolated **0.58-unit**
interval at an observed **1.07 units/hour**, leaving **145.36 above reserve**.
The [budget summary](../reports/budget/20261024-notebook07-observations.json)
keeps the earlier account gap separate and preserves original observations.


## Revised natural-GC trained inference: first run completed 2026-10-24 UTC

[Notebook 08](GC_AWARE_INFERENCE.md) passed in run `20261024T215147Z-bd9574`
at clean source `56d82fb6d783a680e0cb122afd94c15ffc106368`, with **240 tests
passed on T4**. All 336 query batches completed 20 paired repeats in one fresh
session with no reused sidecars. All 8,192 saved validation predictions and
256 source sampler fingerprints replayed; every measured case matched the
original query indices, sampled inputs, seeds, mode orders and occurrence audits.
The original notebook 05 checkpoint and model state stayed unchanged.

| Query batch | Blocks reduction vs HGAM | Within-run 95% interval | Blocks queries/s |
| --- | ---: | ---: | ---: |
| **32, primary** | **8.38%** | **7.75–8.90%** | 1,319 |
| 128 | 8.12% | 7.25–8.94% | 5,205 |
| 512 | 8.21% | 6.41–9.93% | 19,299 |

Case callbacks observed 7,395 collections, including three full collections
inside online passes. Two affected blocks at size 32 and one affected the
reference at 128; all costs remain in the estimator. Program counters show
four full collections, so one occurred outside observed case windows. Its
separate duration is unmeasured, while program wall retains the cost.
The old size-512/HGAM position was 28.221 ms, and no size-512 pass exceeded
100 ms. This does not establish elimination of all future pauses.

The collection-program wall was 1,461.210 seconds, including warmups, audits,
stage replays, reporting and monitor teardown, separately from online throughput.
Maximum cross-mode logit error was `9.5367431640625e-7` and maximum loss error
was `5.960464477539063e-8`, within the original tolerances. Device-wide one-second
samples recorded 161–171 MiB memory. These measurements retain observer overhead
and describe different boundaries from the old notebook 06 observations.

The [verified report](../reports/GC_AWARE_INFERENCE_STATUS.md),
[independent audit](../reports/analysis/20261024T215147Z-bd9574-audit.json) and
[archive provenance](../reports/colab/20261024T215147Z-bd9574/PROVENANCE.json)
preserve this first version-2 result. Repeat the same notebook in a fresh T4
session with `RESUME = False` before describing it as independently reproduced.
No new training or broader-scope claim is implied.

The isolated export recorded **195.07 to 194.63 units** at an observed
**1.07 units/hour**, consuming **0.44 units** and leaving **144.63 above reserve**.
The earlier 195.36-to-195.07 account interval stays separate and unattributed.
See the [budget observations](../reports/budget/20261024-notebook08-observations.json).
