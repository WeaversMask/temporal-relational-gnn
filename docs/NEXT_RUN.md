# Next step: paired inference with the trained checkpoint

Notebook 05 passed in hosted run `20261024T073010Z-1f9800`: **81 T4 tests,
2,000 reference updates and 64,000 distinct training queries processed once**.
Read the [training assessment](../reports/TRAINING_RUN_STATUS.md). The
[verbatim exports and hashes](../reports/colab/20261024T073010Z-1f9800/PROVENANCE.json)
preserve the measurements; the checkpoint binary stays outside Git.

On the same 8,192-query validation subset with 229 positives, GNN ROC-AUC/AP
were 0.83194/0.15439 and activity-baseline scores were 0.85721/0.27248.
The GNN improved over initialization and prevalence, while the activity
baseline scored better on all reported metrics. This checkpoint supports a
trained execution comparison; its scores do not establish it as the preferred
predictor. The official test split remains untouched.

## Frozen weights and next measurement

Use the completed checkpoint without retraining or changing its architecture,
features or sampling fanouts. Its SHA-256 is
`92d02f7d2007c0ab560629e9c4e0d48886a1d149930e68e494a861b85659e1d6`.
The model remains 64 channels, two layers, fanouts `[16, 8]`, FP32, constants
and relative record time. Keep the original training contract as provenance;
record inference batching separately.

The proposed measurement uses the following bounded workload. Implementation
and hosted execution are pending; there is no prepared notebook 06 yet.

1. Reconstruct the pinned graph, literal constant statistics and selected
   cohorts from the checkpoint contract. Verify source inputs, policy,
   runtime and model compatibility, then strictly load the saved weights.
2. Replay all 8,192 saved validation queries at batch size 32 and validation
   seed 1,000,042. Compare labels, all 256 sampler fingerprints and logits
   with the saved report before timing.
3. Measure all 8,192 label-independently selected validation occurrences
   at batch size 32: 256 query batches, 20 paired sampling repetitions and
   five warmups. Randomize execution-mode order and preserve paired draws.
4. Run a secondary comparison on the same 8,192 occurrences at batch sizes
   128 and 512 (64 and 16 query batches), keeping weights, fanouts and features
   fixed. Pair samples
   within each batch size. Grouping changes sampler draws across sizes, so
   this is a batch-scaling experiment rather than identical sampled graphs
   across sizes.
5. Report batch median/p95 latency, queries per second, paired reductions
   against HGAM/reference with uncertainty grouped by query batch, and raw
   neighborhood counts. Record allocated/reserved memory, total-device
   telemetry where available, and host RSS. Keep synchronized stage replays
   separate from complete online pipeline timing.

The current `pipeline_profile` CLI initializes random weights. A trained-profile
entry point must reconstruct this checkpoint before its output can be called
trained inference performance. Training resume is separate from changing
inference batch size.

## Resources and scope

The complete training call took **214.262 seconds** after setup. Recorded
training allocator peaks were **53.742 MiB allocated / 58 MiB reserved**.
GPU memory capacity is ample for this configuration. The next experiment can
measure larger inference batches and repeated paired trials; their speed and
memory costs remain unmeasured. Memory occupancy does not establish GPU compute
utilization, which was not sampled in this run.

The exported balance changed from 198.63 to **198.59**, with a displayed rate
of 1.07 units/hour. The isolation flag was false, so the 0.04-unit account
interval is not attributed as isolated notebook cost. The latest balance
leaves **148.59 above the 50-unit reserve**. No rerun is needed to fill that
attribution gap; record available observations in the next session.

The graph remains an exported snapshot with per-query record-time filtering.
Historical foreign-key edits, deletions, merges and survivorship remain
unresolved; no additional raw mutable or text features are enabled. The
[plan](PLAN.md) keeps these limitations separate from the execution experiment.
