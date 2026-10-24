# Next step: independent reproduction and final report

Notebook 06 completed run `20261024T082321Z-4a43ed`: all three batch sizes,
336 query batches, 6,720 paired trials and 171 T4 tests passed. The trained
weights stayed unchanged, and predictions matched within numerical tolerance.
Read the [verified inference report](../reports/TRAINED_INFERENCE_STATUS.md).

| Batch size | Blocks latency reduction vs HGAM | Within-run 95% interval |
| --- | ---: | ---: |
| **32, primary** | **8.91%** | 8.47–9.36% |
| 128 | 8.02% | 7.07–8.98% |
| 512 | 10.46% | 7.00–15.05% |

The primary result falls below the predefined 10% practical target. The 512
aggregate is sensitive to one 248.378 ms HGAM trial: excluding that entire
paired observation as a diagnostic gives 8.35%. The original all-data result
is retained. These results describe one session and a graph-loaded online
boundary; the intervals do not establish between-session reproducibility.

## Reproduce the fixed experiment

Use [notebook 06](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/06_trained_inference.ipynb)
in a clean independent **T4 session**, following
[the run instructions](TRAINED_INFERENCE.md). Keep the existing
[`configs/trained_inference.json`](../configs/trained_inference.json), checkpoint,
8,192-query cohort, batch sizes 32/128/512, 20 repeats, five warmups, seed 42
and 2,000 bootstrap draws unchanged.

Start with `RESUME = False` and a fresh run folder. Retain the old archive
separately; measure new batches rather than restoring its completed sidecars.
Upload notebook 05's `training.pt` when prompted. The source-validation replay
must pass before timing. Record current balance and hourly rate in their named
fields, export after success or failure, then disconnect the runtime.

Compare the independent session's primary reduction, distributions, sampled
neighborhoods, hardware/runtime metadata and outliers with this run. Preserve
both complete datasets. Keep the predefined primary comparison at batch 32;
report batch scaling and stage diagnostics as secondary findings. After that,
write the final report with within-session and between-session variation
clearly distinguished. No new notebook or architecture change is required
for this reproduction.

## Resources and scientific scope

In the first measurement, blocks throughput increased from 1,328 to 19,280
queries/s between batch sizes 32 and 512. Device-memory observations were
161–171 MiB. One-second GPU activity medians were 21%, 22% and 29% during
collection windows including warmups, audits and stage replays. Those readings
cannot identify an individual mode's utilization or a hardware bottleneck.

The user confirmed corrected budget readings of **198.50 at start, 198.00 at
end and 1.07 units/hour**. With the exported isolation flag true, the balance
decrease was 0.50 units; 148 units remain above the 50-unit reserve. The
[separate correction](../reports/budget/20261024-notebook06-observations.json)
preserves the original transposed/missing export fields.

The trained GNN's source validation quality remains below the activity
baseline's. Features are still constants plus relative time, and historical
foreign-key edits, deletions, merges and survivorship remain unresolved in
the exported snapshot. The execution result retains those limitations.
