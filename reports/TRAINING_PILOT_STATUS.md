# First hosted training pilot: verified 2026-10-24

Notebook 04 passed in run `20261024T051434Z-948a5d`, completing all **100
reference optimizer updates**. The thirteen exported report, configuration and
log files are preserved verbatim in
[`colab/20261024T051434Z-948a5d/`](colab/20261024T051434Z-948a5d/), with a
[SHA-256 provenance manifest](colab/20261024T051434Z-948a5d/PROVENANCE.json).
The recorded source revision is `78feba6ba26622b77e1530ae3ae93315a7857109`,
with a clean working tree. All **69 tests passed** on the Tesla T4, and sampler
preflight passed. The runtime used Python 3.13.16 and PyTorch 2.11.0+cu130 with
one CPU thread.

## What was trained and checked

The frozen pilot used 64 channels, two layers, fanouts `[16, 8]`, batches of 32,
Adam with learning rate 0.001, and seed 42. GNN inputs stayed constant features
plus relative record time. Literal constant statistics were independent of
validation data. The historical-activity logistic baseline fitted its scaler
and coefficients on training queries only. The test split remained untouched.

The label-independent training cohort contained **3,200 query occurrences**,
151 positives, 3,177 unique users and 39 prediction timestamps from
`2010-10-14` through `2020-07-02`. The target horizon is 91 days. All 3,200
selected rows were consumed exactly once across the 100 updates; every update
recorded finite gradients and passed sampled time visibility, disjoint ownership
and BFS prefix checks. Official label-file hashes, selected row IDs and the
input contract are preserved in the report.

Training audits found repeated global post IDs in **two updates**, totaling
three extra post occurrences. All other node types' repeated-global-ID counts
were zero. This adds a small amount of actual repeated-occurrence coverage to
the earlier zero-repeat hosted diagnostics. The distinct training timestamps and
repeated users describe the full cohort; they do not prove every batch contained
the same user at differing query times. Validation and the final parity sample
recorded no repeated global IDs.

After training, one validation batch used the same checkpoint and sampled
occurrences across reference, HGAM and blocks. All reported absolute logit and
loss differences were **zero**, passing the unchanged tolerances. This is one
trained-batch inference equivalence result. **No trained inference speedup was
measured.**

## Held-out pilot subset

Before/after GNN metrics and both baselines use the same **512 official validation
queries**, with **17 positives (3.32%)**, at `2020-10-01`. Metrics were
independently recomputed from the aligned exported labels and probabilities.

| Predictor | ROC-AUC | Average precision | Binary cross-entropy |
| --- | ---: | ---: | ---: |
| Initial random-weight GNN | 0.47807 | 0.03259 | 1.02604 |
| GNN after 100 updates | 0.85419 | 0.13373 | 0.13122 |
| Train-fitted historical-activity logistic baseline | 0.85585 | 0.14655 | 0.12523 |
| Constant training-prevalence baseline | 0.50000 | 0.03320 | 0.14812 |

The trained GNN's subset metrics improved over initialization and the constant
baseline. The activity baseline had slightly better numerical metrics than the
GNN on this subset. With only 17 positive examples, one seed, a short training
budget and no uncertainty analysis, these numbers do not establish a reliable
ranking or statistical superiority between the two learned models. They are
not full-validation scores or finished predictive-quality results.

## Timing and checkpoint evidence

The training call took 14.861 seconds in this attempt, after setup. Its median
recorded step time was 85.507 ms, and the sum of its two timed work regions
across 100 updates was 9.252 seconds. Those regions cover sampling/gathering and
transfer/forward/loss/backward/gradient checks/optimizer work. Query scheduling,
visibility/fingerprint audits, checkpoint writes and reporting are excluded
from `step_ms`, so it is not the complete wall time. Graph construction took
55.044 seconds separately. These observations do not establish trained inference
latency or a compute-unit cost.

The final checkpoint records step 100 and has SHA-256
`67c5550464b3997d90078cb4047c26241f1489f22582357b8647f6845024181c`.
Its exported file size is 16,274,348 bytes. The binary is kept outside Git; its
hash, size and ignored local storage path are recorded in the provenance
manifest. Model/optimizer state and resumability are preserved in that exported
checkpoint. This run was not resumed, so its success does not itself establish
a hosted interrupted-run resume result.

## Budget observations and remaining scope

Both exported start/end balances and the hourly-rate observation were **null**,
although `isolated_runtime_confirmed` was `true`. They remain unchanged in the
archive. A later user message reported **198.63 units remaining**, following the
previously reported 198.67. The **0.04-unit account decrease covers that interval**;
there are no recorded run-boundary balances that attribute it as isolated
notebook 04 consumption. The separate
[budget follow-up](budget/20261024-notebook04-followup.json) preserves this
distinction. The latest reported balance leaves **148.63 units above the
50-unit reserve**.

This completes the bounded hosted training pilot and its checkpoint/metric
review. The feature audit still has `training_ready: false` for broader historical
feature approval: raw mutable/text attributes remain excluded, and historical
foreign-key changes, deletions, merges and snapshot survivorship are unresolved.
The representative final workload, larger training budget, full predictive
evaluation, trained inference timing and isolated per-run compute cost remain
pending.
