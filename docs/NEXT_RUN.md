# Next run: notebook 08, natural-GC trained inference

The revised [notebook 08](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/08_gc_aware_inference.ipynb)
is prepared to measure the same trained model with a cycle-free fingerprint
check and explicit cleanup accounting. Its hosted result is still pending.
Read the [run instructions](GC_AWARE_INFERENCE.md).

Use a fresh **T4** runtime, leave `RESUME = False`, enter current budget readings
and upload the **same original notebook 05 `training.pt`** used for notebooks
06 and 07. Run all cells. It measures all 8,192 queries at sizes **32, 128 and
512**, with the same five warmups, 20 paired repeats, seeds and tolerances.
After success or failure, record the ending balance, export the complete ZIP
and disconnect/delete the runtime.

## What changed and what the run must establish

Notebook 07 directly observed 228–240 ms full Python collections inside four
slow online passes across different execution modes and sizes. The old
batch-512/HGAM position was ordinary in that instrumented run. The evidence
supports collection cadence as an explanation of the earlier spikes without
proving those unrecorded historical events retrospectively. See the
[diagnostic findings](../reports/SPIKE_DIAGNOSTIC_STATUS.md).

The correction removes the CPU-proven avoidable hash cycle. Automatic collection
stays enabled; its pauses remain in each actual timing. Minimal callbacks record
cleanup, and the executor records exact boundaries without host-stage wrappers.
No collection is forced outside the timer and no observations are dropped.
GC durations overlap online/program time and must not be added or subtracted
again. Program wall time, including benchmark audits and stage replays, is
reported separately rather than called serving throughput.

The original checkpoint-bound implementation and earlier reports remain intact.
A new measurement contract binds the revised code and GC settings. Source
inputs, fingerprints, sampling order and weights must match the original archived
cohort. After this first revised run, verify the result before independently
repeating it; the old protocol's approximately 9% benefit is not a measured
result of this new protocol.

## Resources and scientific scope

The last verified balance was **195.36 units**, leaving **145.36 above reserve**;
use the current reading. The prior diagnostic consumed an isolated 0.58 units.
Runtime duration can be similar to notebook 06 and varies with setup and natural
cleanup. Avoid idle sessions and keep all exported evidence.

Removing one helper cycle does not establish elimination of all collections or
slowdowns. The model still uses one checkpoint, schema, validation date and
constant/time features under unresolved exported-snapshot assumptions. Its
source predictive metrics remain below the activity baseline's. This remains
an execution comparison, with no new training or test-split evaluation.
