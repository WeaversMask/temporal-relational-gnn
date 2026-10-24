# Next step: diagnose the recurring batch-512 pause

The trained inference comparison and independent repeat are complete. The
primary latency reduction versus HGAM was **8.91% and 8.77%**, with unchanged
weights and numerically equivalent predictions. The 10% planning target is
recorded for transparency; the repeated approximately 9% improvement remains
a positive result for this workload. See the
[two-session report](../reports/TRAINED_INFERENCE_REPRODUCTION.md).

The user has selected a focused follow-up: explain the one HGAM observation that
recurred at approximately 250 ms in both sessions at size 512, batch 4, repeat
13. A CPU probe found self-referential closures left by the sampler fingerprint
helper. Collection during a later timed pass is a concrete hypothesis; the
original GPU pause's cause is still unconfirmed.

## Run notebook 07

Use the new [batch-512 diagnostic notebook](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/07_spike_diagnostic.ipynb)
in a fresh **T4** runtime with the same notebook 05 `training.pt`. Run all cells,
record current start/end budget readings, export the complete ZIP and disconnect.
The [diagnostic instructions](SPIKE_DIAGNOSTIC.md) explain the three conditions
and the evidence needed before assigning a cause.

The first condition replays the original preceding workload order while recording
collection events. It can take roughly as long as notebook 06; two additional
short controls then examine batches 3–5. Extra observation can change where
collection occurs, so we inspect every slow pass rather than requiring the old
location to recur. All original benchmark code, weights, sampling and exports
remain preserved. Diagnostic times are not replacement speedup estimates.

The last verified balance was **197.46 units**, leaving **147.46 above reserve**.
The preceding repeat recorded an isolated 0.44-unit decrease. Enter a new balance
and rate rather than treating those observations as current measurements.

After the export, determine whether a recorded collection accounts for the pause.
If confirmed, evaluate a cycle-free hashing correction under a separately
recorded timing protocol. If not, retain the unresolved cause and use the traces
to narrow the next test. Do not replace the primary comparison with a secondary
result or retune the model to pursue the round 10% target.
