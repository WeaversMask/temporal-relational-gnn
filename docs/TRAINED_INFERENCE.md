# Paired inference with the trained model

For the current investigation of the recurring size-512 pause, use the separate
[notebook 07 diagnostic](SPIKE_DIAGNOSTIC.md). This page retains the original
notebook 06 performance protocol for deliberate reproduction.

[Notebook 06](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/06_trained_inference.ipynb)
completed its bounded trained-model comparison and independent reproduction:
`20261024T082321Z-4a43ed` and `20261024T102607Z-6593a4`. Each different-T4
session passed 171 tests and completed all 336 query batches with 20 paired
repetitions each. Both used the same frozen protocol and notebook 05 checkpoint,
with no training or reused timing sidecars. The linked report records the
interpretation; this notebook remains available for deliberate reproduction.

| Query batch | First-run reduction vs HGAM | Reproduction reduction | Reproduction within-run 95% interval |
| --- | ---: | ---: | ---: |
| **32, primary** | **8.91%** | **8.77%** | 8.34–9.18% |
| 128 | 8.02% | 8.64% | 7.86–9.42% |
| 512 | 10.46% | 11.82% | 8.97–16.01% |

The first primary interval was 8.47–9.36%. Both primary estimates and intervals
are below the predefined 10% practical target. The secondary batch-512 results
retain an extreme HGAM trial that recurred at the same batch/repeat position
across different GPUs; its cause remains unresolved. Predictions passed the
existing numerical tolerances and weights stayed unchanged. See the
[current two-run report](../reports/TRAINED_INFERENCE_REPRODUCTION.md) for raw
evidence, sensitivity diagnostics and scope.

The primary question is whether blocks reduce online latency beyond HGAM at
batch size 32 while preserving predictions. Batch sizes 128 and 512 provide a
secondary throughput and memory comparison on the same validation cohort.
The source GNN's predictive scores remain below the activity baseline's;
execution improvements do not establish predictive superiority.

## Reproduce in a clean Colab session

1. Save a copy of notebook 06, select **T4 GPU**, disconnect other managed
   runtimes, and enable the existing read-only `GITHUB_TOKEN` secret.
2. Enter current budget observations in the first control cell when available:
   `COMPUTE_UNITS_REMAINING` is the **remaining balance** (197.46 at the end
   of the reproduction); `COMPUTE_UNITS_PER_HOUR` is the **hourly consumption
   rate** (1.07 in both runs). Use new readings for this session. Leave
   `RESUME = False` and do not restore earlier measurement sidecars for an
   independent reproduction. The 50-unit reserve still applies.
3. Run the cells in order. When prompted, upload **`checkpoints/training.pt`
   from notebook 05's exported ZIP**. Its expected SHA-256 is
   `92d02f7d2007c0ab560629e9c4e0d48886a1d149930e68e494a861b85659e1d6`.
   The parent report is already in Git. The notebook verifies both hashes
   before loading or downloading the database.
4. Allow reconstruction and the complete saved-validation replay to finish.
   A mismatch stops the experiment before timing. Keep the matching source
   package versions; this checkpoint requires CUDA for its source replay.
5. After measurement, inspect status and summaries for all three batch sizes.
   Run the ending-balance and ZIP-export cells after success or failure.
   Download the ZIP, then disconnect and delete the runtime.

This experiment repeats inference over many batches and separately replays
stages. Its session can take longer than the training run; no duration or
speedup is promised. It prints completed query batches as measurements finish.
Keep the same checkpoint and frozen settings when reproducing. Another notebook
or a larger model is unnecessary for this comparison.

## Fixed protocol

[`configs/trained_inference.json`](../configs/trained_inference.json) freezes
the parent report/checkpoint hashes and the following settings.

| Setting | Protocol |
| --- | --- |
| Weights, encoders and prediction head | Saved notebook 05 model |
| Features / channels / fanouts | Constants + relative time / 64 / `[16, 8]` |
| Precision | FP32 |
| Validation queries | All 8,192 source-selected occurrences, original order |
| Replay | Original batch size 32 and seed 1,000,042 |
| Primary measured batch size | 32: 256 query batches |
| Secondary batch sizes | 128: 64 batches; 512: 16 batches |
| Paired trials per query batch | 20 |
| Warmup passes per mode per query batch | 5 |
| Execution modes | Reference, HGAM, blocks |
| Logit/loss tolerances | `atol=1e-5`, `rtol=1e-4` |
| Interval estimation | 2,000 paired query-batch bootstrap draws |
| Test labels | Untouched |

Reconstruction checks the source checkpoint, architecture, literal constant
statistics, original code hashes, dependency versions, pinned inputs, feature
policy and selected train/validation rows. It refits no baseline. Replay checks
all 256 saved sampler fingerprints, labels, neighborhoods, logits, probabilities
and metrics before collecting measurements.

Each trial resets the sampler RNG for every execution mode and its stage
replay, checks full temporal-occurrence fingerprints, and randomizes mode order.
No global-ID deduplication is introduced. Different query grouping can change
sampler draws across batch sizes; equivalence is paired within each size.
Weights and buffers are checked before each completed batch is saved.

## Reading the measurements

The complete online boundary is serial sampling, gathering, transfer, feature
and time encoders, message passing, block construction when used, and readout.
Download, graph/model setup, validation replay, correctness checks, report
writes and DataLoader scheduling are excluded. Separate synchronized stage
replays locate costs; their medians are not summed into online latency.

For each batch size, summaries report pooled batch median/p95 latency,
queries/second from actual query counts divided by summed online time, and
paired blocks-versus-HGAM/reference reductions. The reduction uses total online
time, rather than a ratio of medians. Confidence intervals resample whole query
batches, retaining all repeats and mode pairs. They describe variation within
this cohort, rather than independent devices, sessions or training seeds.
Both runs measure graph-loaded online inference under one fixed workload.
Their intervals remain per-run; no pooled between-session confidence interval
is claimed. Two sessions do not establish portability across models, schemas
or dates. A secondary batch size does not replace the primary comparison or
its predefined practical target.

PyTorch allocated/reserved memory peaks are separate from optional one-second
`nvidia-smi` device-wide samples of memory, GPU activity and power. Sampled
maxima are not continuous peaks; other processes contribute to device-wide
counters. The observer's overhead is present during measurement. Batch UTC
collection windows and raw telemetry permit later inspection without treating
memory occupancy as GPU utilization. Unavailable telemetry is recorded honestly
and does not invalidate otherwise valid paired timings.

In the first session, blocks throughput rose from 1,328 to 19,280 queries/second
across batch sizes, a 14.52-fold batching observation under different grouping
and sampler draws. Device memory samples were 161–171 MiB. These observations
leave memory headroom but do not establish the source of a hardware bottleneck.
In the reproduction, every query-batch median favored blocks over both
baselines; message-passing replay improvement was roughly 17%, compared with
roughly 9% across the broader online boundary. The second budget export
recorded an isolated 0.44-unit interval (197.90 to 197.46), leaving 147.46
above reserve. Historical snapshot assumptions, constant/time features and the
stronger activity baseline remain part of the interpretation.

## Persistence and recovery

`trained-profile.json` contains reconstruction/replay audits, protocol, progress,
artifact hashes and summaries. Every complete query batch has an atomic raw
`batches/batch-<size>-<index>.json` sidecar with trials, sampler fingerprints,
equivalence errors, occurrence audits, timings, memory and session provenance.
The ZIP also contains telemetry, logs, budget observations, frozen configuration
and the input checkpoint; it excludes the database.

To continue in the same VM, set `RESUME = True` and rerun the control, restore
and measurement cells. The complete source replay runs again. Finished sidecars
are reused only after checksum, cohort, runtime, hardware, source/model and
protocol verification. An interrupted query batch is measured again. A previous
summary is archived before replacement; fresh execution cannot overwrite it.

For a fresh VM, keep the original profiling commit in `PROJECT_REF`, restore
the input checkpoint, and set `RESUME_ARTIFACTS_DIR` to the `batches/` directory
extracted from your own earlier ZIP. Keep that earlier archive, including its
telemetry and budget interval. Compatible CPU/GPU models, driver, Python and
package versions are required to reuse timings. A new session records its own
budget interval and origin; mixed-session results remain explicitly identified.

The exported-snapshot assumptions about historical foreign keys, deletions,
merges and survivorship remain unchanged. This profiler does not add historical
features or remove those limitations.
