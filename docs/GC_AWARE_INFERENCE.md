# Revised trained inference measurement with natural cleanup

[Notebook 08](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/08_gc_aware_inference.ipynb)
prepares a new measurement protocol after the completed
[collection diagnostic](../reports/SPIKE_DIAGNOSTIC_STATUS.md). No hosted result
has been measured for this protocol yet.

The revision removes the avoidable recursive closure in the sampler fingerprint
check. It keeps automatic garbage collection enabled and retains every actual
online duration, including cleanup pauses. Minimal callbacks observe collection
and its overlap with exact online timers. There is no forced collection between
passes and no host-stage wrapping from notebook 07. Observer overhead remains
present and is not subtracted.

Removing one cycle does not guarantee elimination of long pauses: notebook 07
showed that other collectible objects remain. The purpose is to measure the same
fixed model and workload with the audit correction and explicit cleanup accounting.

Preparation passed 230 local tests, with 10 CUDA/compiled-sampler checks skipped;
lint and formatting checks also passed. Notebook 08 runs the full suite on T4,
including a new CUDA check of this executor. Local checks do not establish GPU
correctness or a performance result.

## Run in a fresh Colab T4 session

1. Open notebook 08, save a copy, select **T4 GPU**, disconnect other managed
   runtimes and enable the existing read-only `GITHUB_TOKEN` secret.
2. Enter the current `COMPUTE_UNITS_REMAINING` balance and displayed
   `COMPUTE_UNITS_PER_HOUR` rate in the first control cell. The last verified
   balance was 195.36; use a new reading. Keep `RESUME = False` for this first
   independent measurement. The 50-unit reserve still applies.
3. Run cells in order. At the upload prompt choose the **same original
   notebook 05 `training.pt` used for notebooks 06 and 07**. Its expected hash is
   `92d02f7d2007c0ab560629e9c4e0d48886a1d149930e68e494a861b85659e1d6`.
4. Allow source checks and complete saved-validation replay to pass before
   measurement. Keep the frozen settings. A mismatch stops the run.
5. After completion or failure, inspect the report, record the actual ending
   balance and download the complete ZIP. Disconnect/delete the runtime once
   the export is saved. Provide the exported folder for verification.

The full cohort, setup, replay and repetitions can take roughly as long as
notebook 06. Runtime duration varies; progress is printed after each completed
query batch. This run measures all three batch sizes in one session and performs
no new training.

## Fixed scientific comparison

The [new configuration](../configs/gc_aware_inference.json) preserves the original
checkpoint, source training report and complete selected validation cohort.

| Setting | Protocol |
| --- | --- |
| Weights and model | Saved notebook 05 heterogeneous GraphSAGE, unchanged |
| Inputs / channels / fanouts | Constants + relative time / 64 / `[16, 8]` |
| Validation queries | Same 8,192 occurrences in original selected-row order |
| Primary query batch | 32, with 256 batches |
| Secondary query batches | 128 and 512, with 64 and 16 batches |
| Warmups / repeats / seed | 5 per mode per batch / 20 paired / 42 |
| Precision / tolerances | FP32 / `atol=1e-5`, `rtol=1e-4` |
| Modes | Reference, existing HGAM, bipartite blocks |
| Main estimand | `100 * (1 - summed blocks time / summed HGAM time)` |
| Within-session intervals | 2,000 paired whole-query-batch bootstrap draws |
| Cleanup policy | Natural automatic collection; no forced collections |
| Hash check | Cycle-free visitor with byte-identical occurrence serialization |

Before timing, reconstruction verifies the immutable training implementation,
input bytes, features, dependencies and saved predictions. Both saved-validation
replay and measured cases use the cycle-free fingerprint visitor, while requiring
every original sampled fingerprint to match. The new measurement
contract separately hashes the new executor, telemetry and audit implementation,
and records active collection thresholds. The historical training/profiling
source is preserved rather than silently changing the checkpoint contract.

Every measured case must match the archived notebook 06 query indices, seeds,
mode orders, sampled fingerprints/counts and visibility audits. Source case
records are verified without retaining the complete historical trial dataset
in the timed process heap. Weights/buffers are checked after every case.
The online operations and synchronization match the reference executor; only
exact timer metadata is retained after the timing window.

## Read the accounting without double counting

The online boundary remains resident-graph sampling, gathering, transfer,
encoders, GNN execution including block construction, and readout. Setup, graph
construction, validation replay, correctness auditing, reporting and DataLoader
scheduling remain outside it. Natural cleanup inside an online timer stays in
its latency and throughput denominator. Every observation remains in the
summed-time estimator; neither slow trials nor GC durations are subtracted.

Each batch saves the ordinary paired profile plus collection start/stop events,
generations, exact timer windows, state snapshots and recomputed overlaps.
Per-size summaries report observed collection counts/durations, overlap by
execution mode and collection outside online timers. These durations explain
parts of already measured time; adding them again would double count.

Experiment collection wall time is reported separately after setup and saved
validation replay. It includes warmups, audits, online trials, stage replays,
reporting and observation overhead. It is not serving latency or a denominator
for production throughput. Per-case GC events cover the case-observation window;
any collection during reporting/gaps still contributes to program wall time,
with boundary GC statistics recorded separately. No unobserved duration is
invented for those gaps.

Separate synchronized stage medians remain separate measurements and must not
be summed into online latency. Sampled GPU activity and device memory remain
distinct from PyTorch allocator peaks. Callback allocations can alter collection
cadence, so these are whole-online measurements with lightweight GC telemetry,
not claims of zero-overhead observation.

## Evidence, interruptions and interpretation

Reports, hashes, raw completed batch files and the checkpoint are exported after
success or failure. Use only notebook 08 sidecars for an intentional resume;
old notebook 06 or diagnostic artifacts are rejected. A fresh run is the default.
If resumed cases span sessions, that aggregate is explicitly marked and is not
an independent single-session reproduction. Within-size confidence intervals
are suppressed when that size combines measurement sessions.

Keep the new result separate from the earlier 8.77–8.91% observations. The
first corrected run needs review, followed by a clean independent repeat before
its percentage can be described as reproduced. Neither a larger benefit nor the
absence of all future pauses is promised.

Scope remains one checkpoint/training seed, schema, validation date and restricted
constant/time feature policy. The activity baseline remains the stronger source
predictor; historical snapshot assumptions and the untouched test split remain
unchanged. GraphSAGE, HGAM and block execution are established methods.
