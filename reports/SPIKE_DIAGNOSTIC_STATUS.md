# Timing-spike diagnostic: verified 2026-10-24

Notebook 07 directly observed **full Python garbage collection taking
227.667–239.758 ms inside four slow online inference passes**. The complete
passes took 251.483–266.993 ms, and the collections occurred during host sampling
or feature gathering, across reference, HGAM and blocks. Expensive collection
can therefore be charged to whichever mode is running when collection starts.
This is a measured timing concern, rather than evidence of a batch-512 memory
limit or a uniquely expensive HGAM model operation.

The old batch-512/HGAM position ran normally in this diagnostic. The new evidence
strongly supports collection cadence as an explanation for the two earlier
quarter-second observations, but those historical passes have no collection
trace, so their exact cause cannot be established retrospectively. The
fingerprint helper's proven reference cycle is one contributor; the controls do
not establish it as the sole cause or show that removing it eliminates long
collections generally.

## Completion and preserved evidence

Run `20261024T121852Z-4747d5` passed at clean source
`da47db9d008d63046c5c35559765b199d455c174`. All **200 tests passed in the T4
runtime**, preflight and saved-validation replay passed, and **332/332 cases**
completed: 326 cases in original workload order, followed by three target cases
under each of two controls. Checkpoint and model digests remained unchanged;
archived query indices, sampling fingerprints/counts, audits and execution
orders matched. The source benchmark contract had no differences.
The runtime remained Python 3.13.16, PyTorch 2.11.0+cu130 and CUDA 13.0 on T4,
with one CPU thread. Maximum cross-mode logit/loss differences were
`9.5367431640625e-7` and `5.960464477539063e-8`, within the original tolerances.

The [provenance manifest](colab/20261024T121852Z-4747d5/PROVENANCE.json) records
680 original files totaling 221,112,536 bytes. Sixteen root files remain
unchanged; 664 case profile/trace files are preserved byte-for-byte after
extraction from [diagnostic-cases.tar.gz](colab/20261024T121852Z-4747d5/diagnostic-cases.tar.gz),
which is 14,133,897 bytes. Its member paths start with `diagnostic-cases/`.
Every decompressed member has a recorded SHA-256 and size. The model binary is
preserved outside Git and retains checkpoint hash
`92d02f7d2007c0ab560629e9c4e0d48886a1d149930e68e494a861b85659e1d6`.

The profiler, training source and original notebook 06 results remain unchanged.
New traces include diagnostic instrumentation and are not replacement benchmark
latencies. The exported `causal_status` remains its conservative automatic
placeholder; this report provides the interpretation after independent review.
See the [derived audit](analysis/20261024T121852Z-4747d5-audit.json).

## What the exact timestamps show

The observer recorded original executor timer reads and collection callbacks on
the same monotonic clock. The following full collections fall inside the exact
online timing windows. Batch and repeat indices below are zero-based.

| Batch size / batch / repeat | Mode | Online ms | Full GC inside timer ms | Host operation |
| --- | --- | ---: | ---: | --- |
| 32 / 60 / 1 | Blocks | 262.631 | 239.758 | Sampling |
| 32 / 139 / 0 | Blocks | 251.483 | 227.667 | Feature gathering |
| 32 / 218 / 11 | HGAM | 266.993 | 238.978 | Sampling |
| 128 / 41 / 9 | Reference | 259.183 | 231.588 | Sampling |

Collection accounts for approximately 89–91% of each complete slow pass. Transfer
and model-execution host spans in these passes were ordinary. These spans
measure CPU dispatch and existing waits around asynchronous CUDA operations,
not individual GPU kernel duration. We do not subtract GC time to construct
replacement inference measurements.

The automatic original-helper phase recorded 7,176 collections: 6,577 generation
0, 595 generation 1 and four generation 2. All four observed generation-2
collections coincided with the four online passes above 200 ms.

Two additional early batch-32 passes took 108.550 and 133.975 ms with **zero**
collection overlap. Their cause is unresolved. Collection explains the four
observed quarter-second pauses, not every slow observation in this run.

At the old size-512/batch-4/repeat-13 position, HGAM took **28.770 ms** with
original automatic collection, **33.216 ms** with between-pass collection and
**28.040 ms** with cycle-free hashing. None overlapped GC. Extra observation
allocations and changed heap history can move collection to another position;
the absence of the old spike alone would not have established its cause.

## What the controls establish

The following comparison uses the same size-512 target batches 3, 4 and 5 under
each condition; the automatic prefix's earlier batches are excluded from this
table. Every condition preserved the archived inputs and numerical tolerances.

| Target condition | Minor / full collections | Total recorded GC time | GC overlapping online timers | Largest online pass |
| --- | ---: | ---: | ---: | ---: |
| Original helper, automatic GC | 66 / 0 | 65.805 ms | 40.707 ms | 51.509 ms |
| Original helper, full GC between passes | 0 / 405 | **88.656 seconds** | 0 ms | 46.155 ms |
| Cycle-free helper, automatic GC | 66 / 0 | 60.250 ms | 37.537 ms | 45.168 ms |

The forced collections took a median 217.707 ms and maximum 270.444 ms each,
all outside the executor timer. This demonstrates relocating a substantial cost;
it is **not a speedup**. It would be misleading to omit those 88.656 seconds
and claim that complete execution became faster.

The original-helper target cases already had no full collection. The cycle-free
control followed 405 explicit collections and covered only three cases, so its
lack of a long pause does not prove the helper change fixes the historical
spike. It still collected 250,012 objects. The earlier CPU probe proves removal
of that specific helper cycle; these traces do not identify the ownership of
all other collectible objects. Phase order and different heap history remain
material limits on comparisons between controls.

## Resources and budget

Target-case maximum PyTorch allocations were identical across the three
conditions: 24.15 MiB reference, 22.68 MiB HGAM and 19.42 MiB blocks, with
38 MiB reserved. Device-wide target samples stayed at 169 MiB. GPU activity
medians were 27%, 5% and 27% over the original, forced and cycle-free target
collection windows. The low forced-control activity includes time spent in
CPU collection outside inference timers; it is not an individual mode's GPU
utilization or proof of a GPU execution defect.

Host lifetime peak RSS was 2.637 GiB. The diagnostic call took 1,689.953 seconds
(28 minutes 10 seconds), including reconstruction, replay and controls. The
isolated budget export recorded **195.94→195.36 units**, a **0.58-unit decrease**,
at **1.07 units/hour**. The balance leaves **145.36 units above reserve**. The
gap from the earlier run's balance to this run's starting reading is separate
and remains unattributed. See the [budget record](budget/20261024-notebook07-observations.json).

## Consequences and next step

The original 8.77–8.91% primary results remain measured observations under their
recorded protocol. This diagnostic does not supply a corrected percentage,
and no historical trial is deleted or rewritten. The batch-512 aggregate was
particularly sensitive to its long HGAM observation; its 10.46–11.82% figures
should not be treated as a stable execution-only advantage.

The immediate finding is that the measurement process can assign expensive CPU
cleanup to different execution modes. A defensible next step is to remove the
avoidable fingerprint cycle and predeclare a fresh measurement protocol that
makes collection policy and cost visible. Retain a natural-GC online result;
if a managed-GC comparison excludes cleanup from its timer, report that boundary
and the cleanup/total cost separately. Preserve weights, sampling, primary
batch size, tolerances and all raw observations.

A revised comparison and independent repeat are needed before describing a
stable efficiency result for the helper correction. The separate
[notebook 08](../docs/GC_AWARE_INFERENCE.md) retains natural cleanup and lightweight
GC observation under a new protocol. Its
[first measured result](GC_AWARE_INFERENCE_STATUS.md) is now verified; independent
reproduction remains pending. The full collection mechanism is directly observed, while the exact historical event and
the two non-GC early pauses remain appropriately qualified.
