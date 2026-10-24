# Paired inference with the trained model

[Notebook 06](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/06_trained_inference.ipynb)
prepares the trained-model speed comparison. Hosted timings are pending.
It uses the saved notebook 05 checkpoint and performs no training.

The primary question is whether blocks reduce online latency beyond HGAM at
batch size 32 while preserving predictions. Batch sizes 128 and 512 provide a
secondary throughput and memory comparison on the same validation cohort.
The source GNN's predictive scores remain below the activity baseline's;
execution improvements do not establish predictive superiority.

## Run in Colab

1. Save a copy of notebook 06, select **T4 GPU**, disconnect other managed
   runtimes, and enable the existing read-only `GITHUB_TOKEN` secret.
2. Enter current budget observations in the first control cell when available.
   Leave `RESUME = False` for the first profiling attempt. The 50-unit reserve
   still applies.
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

PyTorch allocated/reserved memory peaks are separate from optional one-second
`nvidia-smi` device-wide samples of memory, GPU activity and power. Sampled
maxima are not continuous peaks; other processes contribute to device-wide
counters. The observer's overhead is present during measurement. Batch UTC
collection windows and raw telemetry permit later inspection without treating
memory occupancy as GPU utilization. Unavailable telemetry is recorded honestly
and does not invalidate otherwise valid paired timings.

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
