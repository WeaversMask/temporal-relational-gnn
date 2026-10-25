# Diagnose the recurring batch-512 pause

[Notebook 07](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/07_spike_diagnostic.ipynb)
completed on T4 in run `20261024T121852Z-4747d5`, at clean source `da47db9`.
**200 tests passed in the T4 runtime**, and all **332 cases** completed: 326
original-history cases, three forced-collection controls and three cycle-free
hash controls. Weights, contracts and archived sampled-input fingerprints
matched. The original benchmark source and measurements remain unchanged.
These are instrumented diagnostic results, not replacement performance results.
See the [measured findings](../reports/SPIKE_DIAGNOSTIC_STATUS.md).

The revised [notebook 08 protocol](GC_AWARE_INFERENCE.md) has now run with
cycle-free auditing and natural cleanup retained in online timings. Its
[two independent measurements](../reports/GC_AWARE_INFERENCE_REPRODUCTION.md)
showed 8.38%/8.59% primary savings, retaining natural cleanup and all non-GC
tails. The instructions below remain available for deliberate diagnostic
reproduction.

## What we have established

Both notebook 06 sessions recorded their only online observation above 100 ms
at size 512, zero-based batch 4, repeat 13, sampler seed 2,000,135, mode order
reference → HGAM → blocks. HGAM took 248.378 and 252.555 ms; its ordinary median
was about 28 ms. The two runs used identical sampled inputs and different T4
UUIDs. Separate synchronized stage replays did not record a comparable pause.

The original `sampler_fingerprint` contains a recursive nested function whose
closure refers to itself. An isolated CPU fixture produced 100 identical hashes
but left 100 such functions alive with automatic cyclic collection disabled.
An explicit full collection removed 400 unreachable objects and all 100 weak
references became dead. An equivalent module-level visitor produced the same
hashes without those cycles. See the [raw CPU probe](../reports/analysis/512-fingerprint-gc-investigation.json).

This establishes a hashing-helper lifecycle issue on the recorded CPython
3.12 runtime. It does **not** establish that it caused either original T4 pause,
whose runtime was Python 3.13. Hashing occurs outside the online timer, but its
unreachable cycles can survive until a later collection. Notebook 07 now aligns
collection start/stop times with new observed slow passes; it cannot add missing
events to the historical runs.

Python documents [collection callbacks](https://docs.python.org/3.13/library/gc.html#gc.callbacks)
for measuring collection frequency and duration, and distinguishes automatic
cyclic collection from reference-count reclamation. The diagnostic uses those
callbacks without changing the old benchmark source.

## Measured diagnostic outcome

Four automatic-condition online passes took 251.483–266.993 ms and contained
generation-2 collections lasting 227.667–239.758 ms, across different modes
and sizes. Those collections explain most extra duration in the new traced
passes. The original batch-512 batch-4 repeat-13 HGAM position was normal at
28.770 ms with no GC overlap. This strongly supports a collection explanation
consistent with the historical spikes; their untraced causes remain unproved.
Two early 108.55/133.98 ms passes had no GC, so not all pauses have this source.

The original-hash automatic target window and the short cycle-free automatic
control both had zero generation-2 collections. This is not evidence that
changing the helper removed the spike; other cycles remain. Forced collection
performed 405 full collections outside original executor timers, totaling
88.656 seconds. That moves cleanup rather than removes its cost. Neither
control supplies a corrected online speedup.

The old 8.77–8.91% primary results remain old-protocol observations with every
trial retained. A future GC-aware benchmark must be a new protocol and account
for amortized cleanup instead of describing excluded work as an end-to-end gain.
The diagnostic budget was 195.94 to 195.36 units: 0.58 isolated consumption at
an observed 1.07 units/hour, leaving 145.36 above reserve. Earlier account gaps
are separate.

## Reproduce the completed diagnostic

1. Open notebook 07 in a fresh **T4 GPU** runtime. Disconnect other managed
   runtimes and leave `USE_GITHUB_TOKEN = False` for public cloning.
2. Enter the current `COMPUTE_UNITS_REMAINING` balance and displayed
   `COMPUTE_UNITS_PER_HOUR` rate. The latest verified balance is 194.15 units;
   use the new reading. Keep the 50-unit reserve.
3. Run cells in order and upload the same notebook 05 **`training.pt`** when
   prompted. The notebook checks its hash and the original reports before use.
4. Let checkpoint reconstruction, full saved-validation replay and diagnostic
   phases finish. Inspect completion, then record the ending balance and
   download the entire ZIP after success or failure.
5. Disconnect/delete the runtime once the ZIP is saved. Provide the exported
   folder for review, including `diagnostic-cases/` and all logs.

Do not use notebook 06 for this diagnostic. Its raw results stay separate.
No retraining, new dataset or restored measurement sidecars are needed.

## Conditions and recorded evidence

The frozen [diagnostic configuration](../configs/spike_diagnostic.json) points
to the verified second notebook 06 export, original checkpoint and protocol.
The CLI enforces the source, runtime, feature, cohort and sampling contracts,
and records hardware differences. Every diagnostic case must match the old
query indices, all sampler fingerprints/counts, visibility audits and mode order.
Model state is checked after every completed case.

| Condition | Workload | Purpose |
| --- | --- | --- |
| Original hashing, automatic collection | Complete sizes 32/128, then size 512 through batch 5 | Observe collection while replaying the preceding workload order |
| Original hashing, collection between passes | Size-512 batches 3, 4 and 5 | Move full cyclic collections outside each original executor timer |
| Cycle-free hashing, automatic collection | Size-512 batches 3, 4 and 5 | Check an equivalent hashing implementation without the recursive closure |

There are 326 original-history cases and six short control cases. The history
replay can take roughly as long as notebook 06; tracing and explicit full
collections add variable overhead. The notebook retains progress and logs rather
than relying on a promised duration. A deliberate shorter CLI run can omit
`--replay-prefix`, but then it does not recreate the preceding workload order.

A separate module temporarily wraps the unchanged original executor. It records
its exact monotonic start/end timer reads, collection start/stop events and
host spans around sampling, gathering, encoding, message passing, readout and
existing synchronization. It adds no per-stage CUDA synchronization. Host spans
include asynchronous dispatch and existing waits; they are not GPU kernel times.
All wrapped functions, callbacks, collection thresholds and enabled state are
restored on success, error or interruption. Between-pass collections are outside
the original executor's timer, and their cost remains in the raw trace.

Each completed case saves its ordinary paired profile and a separate trace with
SHA-256 references in the progress report. Trace summaries list full-collection
durations, online passes above 100 ms, and GC overlap with the exact timer
window. Interrupted or failed current-case traces are retained when available.

## How to interpret the result

Collection callbacks align long full collections with the new slow passes and
account for most added duration. Extra observation allocations changed when
collection occurred, so different modes and positions were affected. Host spans
are not CUDA kernel timings, and two early non-GC pauses remain separate.

The short controls follow the automatic phase and have different heap histories.
Their lack of a full collection cannot establish that the cycle-free helper
removed the historical spike. Collection between passes explicitly moves cost
outside the old executor timer; report that cleanup cost rather than treating it
as eliminated. The trace does not retroactively prove an unrecorded event in
notebook 06. Review the future timing boundary and amortized cleanup before
another performance experiment.

Keep instrumented times separate from the existing 8.77–8.91% primary result.
Any later timing-harness correction and clean benchmark must be recorded as a
new protocol; do not silently delete the old spikes or relabel observations.

## Validation and preserved evidence

Local validation passed 191 tests with nine environment-dependent skips, plus
Ruff lint and format checks. The 29 new diagnostic tests cover unchanged
fingerprints and predictions, precise timer/GC overlap, collection outside the
timed executor, restoration after interruption, frozen thread/seed settings,
source protection and partial-trace persistence. Notebook code cells compile
and the referenced source-report hashes match. These preparation checks were CPU
evidence. The completed hosted run separately
passed sampler/GPU preflight and all 200 tests in its T4 runtime. Neither test
count alone proves the historical pause cause. The archive preserves all root
reports and logs unchanged; the 664 profile/trace files are byte-identical entries
in `diagnostic-cases.tar.gz`, with per-entry SHA-256 hashes and sizes in
[PROVENANCE.json](../reports/colab/20261024T121852Z-4747d5/PROVENANCE.json).
