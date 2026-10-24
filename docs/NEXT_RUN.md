# Current result: independent reproduction completed

The fresh notebook 06 run `20261024T102607Z-6593a4` completed all three sizes,
336 query batches and 6,720 paired trials, with 171 tests passed on T4. It used
the same checkpoint and protocol as the first session, on a different device,
with no resumed measurement sidecars. Read the
[two-session investigation report](../reports/TRAINED_INFERENCE_REPRODUCTION.md).

| Batch size | First latency reduction vs HGAM | Independent repeat |
| --- | ---: | ---: |
| **32, primary** | **8.91%** | **8.77%** |
| 128 | 8.02% | 8.64% |
| 512 | 10.46% | 11.82% |

The primary improvement repeated and remains below the 10% practical target.
Predictions matched within unchanged tolerances and weights stayed fixed.
The larger batch's HGAM timing spike recurred at the same batch/repeat position
on a different GPU. All data remain included, with sensitivity diagnostics and
its unresolved cause disclosed in the report.

## No additional unchanged run is needed for this comparison

The planned independent repeat is complete, and the current evidence is ready
for explanation and reporting. Keep both original exports and the trained
checkpoint. If a Colab runtime is still connected, download its ZIP and
then disconnect/delete it.

The report distinguishes the approximately 9% complete online saving from the
approximately 17% message-passing-stage saving, records the unachieved primary
target, and preserves the activity baseline's predictive advantage and the
historical snapshot limitations. It makes no pooled cross-device confidence
claim from two sessions.

Explaining the recurring size-512 spike would need a targeted trace experiment
with a separately recorded protocol. Do not replace the primary batch-32 result
with a secondary result or retune the workload to pursue the threshold. A richer
model, additional training seeds, another schema or historical reconstruction
would extend the study beyond this frozen execution comparison.

## Budget and future reproduction

This repeat recorded **197.90 at start, 197.46 at end and 1.07 units/hour**,
with isolation confirmed. Its 0.44-unit balance decrease leaves **147.46 units
above reserve**. The [budget record](../reports/budget/20261024-notebook06-reproduction-observations.json)
is separate from the first session's corrected 0.50-unit interval.

For a deliberate future reproduction, use
[notebook 06](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/06_trained_inference.ipynb)
in a fresh T4 runtime with `RESUME = False`, the same notebook 05 `training.pt`
and unchanged [frozen protocol](../configs/trained_inference.json). Follow the
[run instructions](TRAINED_INFERENCE.md) and save new measurements separately.
