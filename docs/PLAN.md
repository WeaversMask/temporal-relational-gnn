# Plan: 14-day core, up to seven additional days

## First 48-72 hours: feasibility before investment

- [x] Pin inspected upstream source commits and public dataset revision.
- [x] Implement synthetic reference, HGAM, and destination-restricted execution.
- [x] Verify seed outputs, gradients, and an optimizer step on CPU fixtures.
- [x] Create a private remote and a resumable Colab starting notebook.
- [x] Run CUDA correctness checks on Colab T4 (18 tests passed).
- [ ] Record actual Colab unit/hour consumption (exported budget fields are blank).
- [x] Check the installed temporal sampler on a tiny real PyG graph.
- [x] Fetch the pinned task/schema manifests.
- [ ] Complete the temporal feature availability audit on real data.
- [ ] Download/project the required real data; build its historical graph.
- [ ] Run one real batch through the unchanged baseline and both alternative modes.
- [ ] Measure sampled graph size, host/VRAM use, and pipeline stage proportions.

Verified exports and hashes are under `reports/colab/`. The next notebook is
`notebooks/02_real_batch.ipynb`; its initial real-data model remains untrained and
uses constant node features plus relative time. This is not the final predictor.

Decide using representative evidence whether to continue block optimization.
The 10% end-to-end latency reduction is a practical hypothesis, not a promised
outcome. If sampling/encoding dominates, report that limit early. Do not expand
the configuration matrix to hide a weak primary result.

## Days 3-4: freeze the real experiment

Select safe historical features; exclude text, mutable snapshots, and task-label
augmentation initially. Confirm eligible users exist at each query time. Fit
preprocessing on training-period data; use official temporal splits. Keep keys
for connectivity and preserve all relationships.

Start from the provisional configuration in `configs/experiment.json`. Measure
100 steps before choosing the training budget. Train one credible checkpoint
and compare predictive quality with a historical-activity tabular baseline.
Freeze features, graph semantics, model, primary workload, and tolerances.

## Days 5-9: integrate and establish real-model equivalence

Reuse the actual encoder, time encoder, prediction head, weights, and identical
sampled batches. Compare full-model logits, loss, retained intermediates, and
small-batch gradients. Extend tests only for concrete real-data cases not already
covered. Investigate numerical differences rather than loosening tolerances.

## Days 10-14: measure, interpret, reproduce

Compare reference/HGAM/blocks on paired workloads in the same device session.
Report both message-passing-only and complete inference time, including sampling,
gathering, transfer, encoders, and block construction. Record hardware, thread
counts, exact dependencies, warmup, raw repetitions, and neighborhood sizes.
Use official predictive metrics plus average precision where appropriate.

Persist each completed configuration's results and save resumable training state
to Drive. Finish a clean reproduction, a concise report, and an explanation of
successes, regressions, and limitations. Never label synthetic timings as real.

## Optional days 15-21

Choose one extension after the core is complete: a deeper-model workload or a
small second schema. Extra training seeds are secondary to a reliable paired
execution comparison. Separate portability across schemas from learned transfer.

## Agentic development

Agents can help implement bounded changes, debug, create adversarial cases, and
organize experiments. The researcher must personally explain the dependency
argument, temporal occurrences, empty relations, and timing boundaries. Coding
speed does not remove training time or the need for experimental judgment.
