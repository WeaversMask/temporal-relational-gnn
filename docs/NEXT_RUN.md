# Next run: independently repeat notebook 08

The first natural-GC measurement passed **240 tests on T4** and all **336 query
batches** in run `20261024T215147Z-bd9574`. Blocks reduced online latency versus
HGAM by **8.38% at batch 32** (primary, within-run interval **7.75–8.90%**),
**8.12% at 128** and **8.21% at 512**. The original weights, predictions and
sampled inputs matched. Read the
[verified findings](../reports/GC_AWARE_INFERENCE_STATUS.md).

## Repeat the same frozen measurement

Use the same [notebook 08](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/08_gc_aware_inference.ipynb)
in a **fresh T4 runtime**, with **`RESUME = False`** and no reused measurement
sidecars. No new notebook, model training or settings change is needed.

1. Disconnect other managed runtimes, enable the existing `GITHUB_TOKEN` secret,
   and enter current compute-unit balance/rate readings.
2. Upload the **same original notebook 05 `training.pt`** used for notebooks
   06–08. Keep the frozen configuration and run all cells.
3. Let source checks, full saved-validation replay and all three batch sizes
   complete. Record the ending balance, export the complete ZIP and disconnect.
4. Verify that new export before describing the version-2 result as independently
   reproduced. Keep each session's estimates and intervals separate.

See [the complete run instructions](GC_AWARE_INFERENCE.md).

## What this repeat resolves

The observed benefit remains positive when natural cleanup pauses are retained.
The first revised run still had three full collections inside online passes,
including two received by blocks. A fourth full collection occurred outside the
case-observer windows according to program counters; its separate duration is
unknown, while its cost remains in program wall time. Removing the avoidable
hash cycle does not eliminate all cleanup or explain every smaller slowdown.

The old size-512 HGAM spike position took 28.221 ms in this run. No size-512
pass exceeded 100 ms, and its reduction was 8.21%. That is consistent with
changed collection cadence; it does not retrospectively identify the untraced
old pauses. Keep the old notebook 06 results and notebook 07 diagnostic separate.

A second fresh version-2 session will show whether the current benefit repeats
under the same measurement protocol. Retain every observation and keep cleanup
cost inside its actual timing boundary. Program wall includes audits and stage
replays and must not be called serving throughput. No larger benefit is promised.

## Resources and scientific scope

The last verified balance was **194.63 units**, leaving **144.63 above reserve**;
use the current reading. The first notebook 08 interval consumed an isolated
**0.44 units** at an observed **1.07 units/hour**. Earlier account gaps remain
separate. Avoid idle sessions and keep all exported evidence.

The model still uses one checkpoint, schema, validation date and constant/time
features under unresolved exported-snapshot assumptions. Its predictive metrics
remain below the activity baseline's. The repeat performs no training or test
split evaluation. Decide on any additional workload only after this independent
repeat is verified.
