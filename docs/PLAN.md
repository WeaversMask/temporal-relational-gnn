# Plan: 14-day core, up to seven additional days

## First 48-72 hours: feasibility before investment

- [x] Pin inspected upstream source commits and public dataset revision.
- [x] Implement synthetic reference, HGAM, and destination-restricted execution.
- [x] Verify seed outputs, gradients, and an optimizer step on CPU fixtures.
- [x] Create a private remote and a resumable Colab starting notebook.
- [x] Run CUDA correctness checks on Colab T4 (81 tests passed in notebook 05).
- [x] Record the actual Colab unit/hour rate (1.07 observed in notebook 03).
- [x] Record a later balance (198.67); account decrease was 0.11 units with overlapping runtimes reported.
- [x] Record the notebook 04 follow-up balance (198.63); the later 0.04-unit interval decrease is not an isolated run cost.
- [x] Record notebook 05 balances (198.63 to 198.59) and 1.07 units/hour; isolation was not confirmed.
- [ ] Measure isolated per-run compute-unit consumption without overlapping runtimes.
- [x] Check the installed temporal sampler on a tiny real PyG graph.
- [x] Fetch the pinned task/schema manifests.
- [ ] Complete the temporal feature availability audit on real data.
- [x] Run the conservative feature-policy schema/footer audit; additional raw attributes remain excluded.
- [x] Download/project the required real data; build its historical graph.
- [x] Run one real batch through the unchanged baseline and both alternative modes.
- [x] Profile sampled graph sizes, host RSS, CUDA allocator peaks and separate pipeline stages for the current untrained resident-graph boundary.
- [ ] Validate representative trained-workload costs and total device VRAM; stage medians alone do not establish pipeline proportions.

Verified exports and hashes are under `reports/colab/`. Notebooks 02 and 03
passed on the hosted database with three batches of eight queries; see
[`reports/PIPELINE_PROFILE_STATUS.md`](../reports/PIPELINE_PROFILE_STATUS.md).
Notebook 03 measured the online boundary from sampling through logits after
graph/model setup. Blocks reduced its median latency by 9.04–11.68% versus
HGAM. Host RSS and CUDA allocator peaks are measured; total device VRAM is not.
Separate synchronized stage replays must not be summed into a pipeline total
or assumed to give exact proportions.

Notebook 03 used an untrained model, one query timestamp and no repeated global
IDs across its 30 trials. Notebook 04 has now completed 100 supervised updates
over 3,200 training queries, 39 timestamps and 3,177 unique users. Two training
updates contained repeated global post occurrences. The trained checkpoint
passed exact inference logit/loss parity on one validation batch; see
[`reports/TRAINING_PILOT_STATUS.md`](../reports/TRAINING_PILOT_STATUS.md).
The metadata audit remains `training_ready: false` for broader historical feature
approval. Historical relationship evidence and the representative final workload
remain open; the pilot retained constants/time and the documented snapshot contract.

Decide using representative evidence whether to continue block optimization.
The 10% end-to-end latency reduction is a practical hypothesis, not a promised
outcome. If sampling/encoding dominates, report that limit early. Do not expand
the configuration matrix to hide a weak primary result.

## Days 3-4: freeze the real experiment

Select safe historical features; exclude text, mutable snapshots, and task-label
augmentation initially. Confirm eligible users exist at each query time. Fit
preprocessing on training-period data; use official temporal splits. Keep keys
for connectivity and preserve all relationships.

The first 100-step pilot from `configs/training_pilot.json` passed before choosing
a larger training budget. Its fixed settings were 32 queries per batch, 3,200
training queries, 512 held-out validation queries, 64 channels, fanouts `[16, 8]`,
learning rate 0.001 and seed 42. It trained only the reference schedule, saved
resumable state, compared the train-fitted activity baseline, and verified
inference parity across all three schedules with the trained checkpoint. The
test split remained untouched. Its 512-query subset had only 17 positives; the
GNN reached ROC-AUC 0.85419 / AP 0.13373 and the activity baseline 0.85585 /
0.14655. These are preliminary subset results, not the final trained experiment
or a readiness claim for richer historical features.

- [x] Complete the hosted 100-step pilot and preserve losses, timings and checkpoint.
- [ ] Record isolated start/end unit balances for the next run; the pilot's boundary balances are unavailable.
- [x] Review held-out subset metrics, baseline quality and trained-checkpoint parity.
- [x] Prepare the next bounded training budget: 64,000 queries, 2,000 updates and 8,192 validation queries; see `configs/training_run.json`.
- [x] Complete the larger hosted training run and review its baseline comparison.
- [x] Preserve and select the notebook 05 checkpoint for the trained execution comparison.
- [x] Implement the paired trained inference workload and frozen notebook 06 protocol.
- [ ] Run the trained profiler on T4 and review paired latency, throughput and memory.

The pilot informed the next training configuration in
[`configs/training_run.json`](../configs/training_run.json); its
[notebook 05 instructions](TRAINING_RUN.md) retain the conditional snapshot
contract. Its 2,000 updates consumed all 64,000 training queries once and took
214.262 seconds. On 8,192 validation queries with 229 positives, the trained
GNN reached ROC-AUC 0.83194 / AP 0.15439, below the activity baseline's
0.85721 / 0.27248. These use different cohorts from the pilot. The weights are
preserved for an execution comparison, with the quality gap reported honestly.
The inference protocol in `configs/trained_inference.json` and
[notebook 06 instructions](TRAINED_INFERENCE.md) is implemented. Hosted
measurement remains pending. Its batch scaling keeps the trained model unchanged.

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

Persist each completed configuration's results and save exported resumable
training checkpoints outside the VM. Finish a clean reproduction, a concise
report, and an explanation of successes, regressions and limitations. Never label
synthetic timings as real.

## Optional days 15-21

Choose one extension after the core is complete: a deeper-model workload or a
small second schema. Extra training seeds are secondary to a reliable paired
execution comparison. Separate portability across schemas from learned transfer.

## Agentic development

Agents can help implement bounded changes, debug, create adversarial cases, and
organize experiments. The researcher must personally explain the dependency
argument, temporal occurrences, empty relations, and timing boundaries. Coding
speed does not remove training time or the need for experimental judgment.
