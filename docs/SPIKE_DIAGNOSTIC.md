# Diagnose the recurring batch-512 pause

[Notebook 07](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/07_spike_diagnostic.ipynb)
tests a concrete Python garbage-collection lead for the HGAM pause. The original
trained benchmark results and checkpoint-bound source remain unchanged. This is
an instrumented diagnostic, not a replacement performance measurement.

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
unreachable cycles can survive until a later collection. We need collection
start/stop times aligned to an actual reproduced pause.

Python documents [collection callbacks](https://docs.python.org/3.13/library/gc.html#gc.callbacks)
for measuring collection frequency and duration, and distinguishes automatic
cyclic collection from reference-count reclamation. The diagnostic uses those
callbacks without changing the old benchmark source.

## Run the prepared diagnostic

1. Open notebook 07 in a fresh **T4 GPU** runtime. Disconnect other managed
   runtimes and enable the existing read-only `GITHUB_TOKEN` secret.
2. Enter the current `COMPUTE_UNITS_REMAINING` balance and displayed
   `COMPUTE_UNITS_PER_HOUR` rate. The last verified balance was 197.46 units;
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

A long collection that occurs inside a reproduced slow online pass and accounts
for its extra duration would support collection as the immediate source of that
new pause. If the cycle-free control no longer accumulates the proven helper
cycles, that supports the proposed lifecycle correction. Inspect all modes and
positions; extra observation allocations can move collection away from the old
batch/repeat or into another mode.

Absence of a pause in the short controls alone does not prove its cause: they
follow the automatic condition and have different heap histories. Nor can a new
trace retroactively prove an unrecorded event in the original sessions. If the
pause does not reproduce or GC does not account for it, retain an unresolved
conclusion and use the recorded host spans to choose the next targeted trace.

Keep instrumented times separate from the existing 8.77–8.91% primary result.
Any later timing-harness correction and clean benchmark must be recorded as a
new protocol; do not silently delete the old spikes or relabel observations.

## Preparation checks

Local validation passed 191 tests with nine environment-dependent skips, plus
Ruff lint and format checks. The 29 new diagnostic tests cover unchanged
fingerprints and predictions, precise timer/GC overlap, collection outside the
timed executor, restoration after interruption, frozen thread/seed settings,
source protection and partial-trace persistence. Notebook code cells compile
and the referenced source-report hashes match. These CPU checks do not establish
CUDA correctness or the cause of the original GPU pause; notebook 07 performs
the separate sampler/GPU preflight and current suite before measurement.
