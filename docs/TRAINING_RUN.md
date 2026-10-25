# Larger training run: 64,000 queries

[Notebook 05](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/05_training_run.ipynb)
completed the bounded reference-training run after the successful
[100-update pilot](../reports/TRAINING_PILOT_STATUS.md). All 2,000 updates and
81 tests passed on the T4 in run `20261024T073010Z-1f9800`.
See the [reviewed results](../reports/TRAINING_RUN_STATUS.md) and
[export provenance](../reports/colab/20261024T073010Z-1f9800/PROVENANCE.json).

Training-call wall time was 214.262 seconds, with a 53.742 MiB peak PyTorch
allocation. On 8,192 validation queries with 229 positives, the GNN reached
ROC-AUC 0.83194 / AP 0.15439; the activity baseline reached 0.85721 / 0.27248.
One trained validation batch passed exact three-mode logit/loss parity.
Trained inference speed remains unmeasured.

The run starts from fresh weights. It keeps the pilot's architecture, constant
features, relative record time, optimizer and sampling settings. Both the GNN
and historical-activity logistic baseline fit the same selected training rows.
The new training and validation cohorts are selected independently of labels;
they need not contain the earlier pilot cohorts. Compare models on the new
cohort, rather than treating changes from the pilot's scores as measured gains.

## Fixed configuration

The notebook reads [`configs/training_run.json`](../configs/training_run.json)
and exports its frozen copy with the reports.

| Setting | Larger run |
| --- | --- |
| Official training queries | 64,000 |
| Reference optimizer updates | 2,000 |
| Queries per update | 32 |
| Training coverage | One pass through a fixed permutation |
| Official validation queries | 8,192 |
| Channels / layers | 64 / 2 |
| Fanouts per relation | `[16, 8]` |
| Optimizer / learning rate | Adam / 0.001 |
| Seed | 42 |
| Progress report / checkpoint interval | 100 updates |
| Test labels | Untouched |

Every update still records its loss, timing, sampled occurrence fingerprint,
neighborhood counts, temporal visibility audit and finite-gradient check.
Periodic JSON writes reduce serialization overhead while retaining the full
history in the final report. Checkpoints independently preserve model,
optimizer, query order, random state and the input contract. A hard VM loss
can require replaying updates after the last saved checkpoint.

## Reproducing the completed run

1. Open notebook 05 and save a copy. Disconnect other managed runtimes, select
   **T4 GPU**, and leave `USE_GITHUB_TOKEN = False` for public cloning.
2. In the first control cell, enter the current remaining units and displayed
   unit/hour rate when available. Set the isolation flag after checking other
   runtimes. Missing observations stay `None`; the previous 198.63 balance is
   a dated observation. Keep `RESUME = False` for this new run.
3. Run the cells in order. The notebook checks dependencies and correctness,
   reuses matching cached train/validation data, fits the baseline, and trains
   the reference model. It downloads missing pinned inputs when needed.
4. Inspect completion, validation metrics and trained-mode parity. After success
   or failure, run the ending-balance and ZIP-export cells. Keep the ZIP and
   confirm that it contains `checkpoints/training.pt` when training saved one.
5. Disconnect and delete the runtime after downloading the archive. Preserve
   the planned 50-unit reserve.

The completed run measured 185.104 seconds of timed training work and
214.262 seconds of training-call wall time. Setup, baseline fitting and
validation were outside that call; graph construction alone took 56.046
seconds. Reproduction runtimes may differ. The notebook does not measure
trained inference speedup.

## Recovery and evidence

Run artifacts are stored in a temporary folder separate from the database.
The ZIP includes the frozen configuration, setup/preflight/test logs, raw
`training-run.json`, budget observations and available checkpoints.
Resuming in the same folder preserves the preceding report as a separate
artifact, including its initial validation measurements.

To continue this run in the same VM, set `RESUME = True` and rerun the control
and restore cells before training. To recover in a fresh VM, use your own
`training.pt` from its exported ZIP and set `PROJECT_REF` to that run's recorded
source commit. Keep matching dependency versions. Resume checks source code,
input hashes, selected cohorts, model, optimizer and sampling settings before
continuing toward **2,000 total updates**. The 100-update pilot checkpoint uses
a different input contract and cannot be used to start this run.

The graph remains the pinned exported snapshot with per-query record-time
visibility. Earlier foreign-key changes, deletions, merges and survivorship
have not been reconstructed. The metadata audit's `training_ready: false`
still documents these broader limitations; this run adds no raw mutable or
text attributes. Its larger validation subset remains a subset, and final
predictive evaluation and paired trained inference measurements remain open.
