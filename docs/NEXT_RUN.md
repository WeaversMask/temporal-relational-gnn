# Next steps after independent reproduction

The version-2 primary result is independently reproduced: blocks reduced online
latency versus HGAM by **8.38% and 8.59%** in two fresh T4 sessions, using the
same checkpoint, sampled inputs and benchmark contract on different physical
GPUs. Both sessions passed **240 hosted tests** and all **336 query batches**.
Read the [consolidated findings](../reports/GC_AWARE_INFERENCE_REPRODUCTION.md).

## The bounded experiment is complete

No third unchanged GPU run or new notebook is needed for this result. The
consolidated report, both original exports, independent numerical audits and
budget records are preserved. Keep the downloaded checkpoints/ZIPs and disconnect
any remaining runtime once its export is saved.

The defensible finding is roughly **8.4–8.6% primary online latency reduction**
in this fixed workload. The two within-run intervals remain separate; no pooled
cross-session interval is claimed. GraphSAGE, HGAM and block execution remain
established methods, with this project adapting and evaluating their execution.

If the investigation is extended, choose a separate, predefined question such
as generality on a second schema/task or behavior in a deeper model. That work
requires its own protocol and evidence; it is not prepared or measured here.
The current findings alone do not establish broad deployment or publication
readiness. Repeating the same configuration to seek a larger percentage is not
part of the plan.

## What remains uncertain

The primary result agrees closely between sessions, and size 128 shows
8.12%/8.42% reductions. Size 512 is noisier: 8.21%/7.65%, with the repeat's
within-run interval widening to **2.07–11.49%**. Six additional size-512 passes
exceed 100 ms in that repeat with little or no GC overlap. Their causes are
unresolved; every observation remains in the estimator and bootstrap.

Full collections still occur. Each session has three inside observed online
passes and a fourth outside case-observer windows according to program counters;
its separate duration is unknown, while program wall retains the cost. Automatic
cleanup stays enabled, and no cleanup cost or slow trial is subtracted. Program
wall includes audits and stage replays and is not serving throughput.

The same original notebook 05 model uses one training seed, schema, validation
date and constant/time features under unresolved exported-snapshot assumptions.
Its predictive metrics remain below the activity baseline's. The test split
remains untouched. Preserve these boundaries in any write-up or résumé claim.

## Resources and deliberate reproduction

The latest verified balance is **194.15 units**, leaving **144.15 above the
50-unit reserve**. The repeat consumed an isolated **0.39 units** at an observed
**1.07 units/hour**. The intervening 0.09-unit account gap remains separate.
Use current readings for any future run and avoid idle GPU sessions.

For deliberate reproduction, retain the same
[notebook 08](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/08_gc_aware_inference.ipynb),
a fresh T4 runtime, `RESUME = False`, no reused sidecars and the **same original
notebook 05 `training.pt`**. See the
[full reproduction instructions](GC_AWARE_INFERENCE.md). No new training is
needed to reproduce this bounded result.
