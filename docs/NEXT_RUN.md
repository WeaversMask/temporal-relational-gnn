# Current finding: Python collection can dominate a timed pass

Notebook 07 completed all **332 cases**, with **200 tests passed on T4**.
Four online passes took 251–267 ms, with 228–240 ms of full Python garbage
collection inside their exact timers. They occurred in different modes and
batch sizes during sampling or gathering. The original batch-512/HGAM position
was ordinary (28.770 ms) in this instrumented run.

Read the [verified diagnostic report](../reports/SPIKE_DIAGNOSTIC_STATUS.md).
It supports collection cadence as the likely explanation of the earlier spikes,
while preserving the limit that those historical passes had no GC trace. Two
early slow passes in this run had no GC overlap and remain unexplained.

## What to change before another performance run

The original recursive fingerprint helper has a CPU-proven avoidable reference
cycle. An equivalent cycle-free implementation passed the actual T4 input and
prediction checks, but its short control does not establish that all long
collections disappear. It still collected many other objects.

The next engineering step is to remove that avoidable cycle under an explicitly
versioned measurement protocol, keeping the checkpoint, sampled occurrences,
primary batch size and tolerances fixed. The checkpoint binds original source
hashes, so do not silently edit the old training/profiling source or bypass its
checks. A new profiling implementation can preserve the legacy reconstruction
contract and record the changed audit implementation separately.

Predeclare how collection is treated. Natural automatic-GC online results should
retain its actual costs. Any controlled comparison that moves cleanup outside
the timer must label that boundary and report cleanup and total time as well.
The diagnostic's forced control spent **88.656 seconds** collecting outside its
inference timer; that is relocated work, not free acceleration.

After that change is concrete and CPU-verified, prepare a fresh uninstrumented
comparison. Do not reuse the instrumented notebook 07 latencies as corrected
speedups, subtract GC from the old trials, or replace the original primary
comparison with the tail-sensitive batch-512 aggregate. The original primary
8.77–8.91% remains a measured observation of the old protocol, pending this
additional measurement-method check.

## Resources and preserved evidence

The latest run recorded **195.94 to 195.36 units**, with isolation confirmed,
for **0.58 units consumed** at an observed 1.07 units/hour. The ending balance
leaves **145.36 above reserve**. Earlier account intervals remain separate.
All original diagnostic case bytes are retained in a compressed archive, with
a [complete provenance manifest](../reports/colab/20261024T121852Z-4747d5/PROVENANCE.json).
The trained checkpoint remains outside Git.

No additional GPU run is ready or required simply to recheck these traces.
Keep notebook 07 for deliberate diagnostic reproduction; its
[instructions](SPIKE_DIAGNOSTIC.md) document the exact conditions and limits.
