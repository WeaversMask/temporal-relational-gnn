# First training pilot: 100 reference updates

[Notebook 04](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/04_training_pilot.ipynb)
completed the first supervised training pilot on `rel-stack/user-badge` in run
`20261024T051434Z-948a5d`: all 100 updates and 69 tests passed on a Tesla T4.
See [the reviewed results](../reports/TRAINING_PILOT_STATUS.md) and
[verbatim exports with provenance](../reports/colab/20261024T051434Z-948a5d/PROVENANCE.json).
The checkpoint is preserved outside Git with its hash recorded in provenance.

The pilot uses constant node features plus relative record time and the reference
training schedule. It retains the published model's training behavior, then
compares reference, HGAM and block inference on one validation batch with the
same updated checkpoint and sampled occurrences. No trained inference speedup
is measured in this pilot.

On the fixed validation subset of 512 queries and 17 positives, the trained GNN
reached ROC-AUC 0.85419, average precision 0.13373 and BCE 0.13122. The
train-fitted activity baseline reached 0.85585, 0.14655 and 0.12523. The small
numerical difference does not establish a reliable ranking between models.
The final one-batch inference check reported zero logit and loss errors across
reference, HGAM and blocks. Full predictive quality and trained speedup remain
unmeasured.

## Frozen pilot configuration

The settings are recorded in [`configs/training_pilot.json`](../configs/training_pilot.json)
and copied into the exported run folder. They freeze this first pilot, not the
final experiment or the proposed larger training budget.

| Setting | First pilot |
| --- | --- |
| Total optimizer updates | 100 |
| Batch size | 32 |
| Selected official training queries | 3,200 |
| Fixed official validation subset | 512 queries |
| Channels / layers | 64 / 2 |
| Fanouts per relation | `[16, 8]` |
| Learning rate | 0.001 |
| Seed | 42 |
| Checkpoint interval | Every 10 completed updates |
| Training execution | Reference |
| Model inputs | Constant features and relative time |

Training and baseline fitting use official training rows only. The validation
subset is held out from optimizer updates and baseline fitting. The test split
is not downloaded, selected, fitted or evaluated. Validation metrics describe
this pilot subset, not the final full-validation score.

The activity baseline uses historical activity counts at each query time and
logistic regression fitted on training queries only. Before/after model metrics
and the baseline use the same selected validation cohort. They are preliminary
comparisons under the same hosted-snapshot assumptions.

Record-time filtering and per-query temporal sampling preserve visibility under
the chosen graph contract. The exported snapshot does not reconstruct historical
relationship changes, deletions, merges or survivorship. The feature metadata
audit still has `training_ready: false`: this bounded constant/time pilot does
not approve richer historical raw attributes or resolve those limitations.

## Reproducing the completed pilot

1. Save a copy of notebook 04. In Colab's managed sessions, disconnect other
   runtimes first. Use one active runtime, then select T4 and connect. For CPU
   development, select a CPU runtime and set `DEVICE = "cpu"`.
2. Enable this notebook's access to the existing read-only `GITHUB_TOKEN`. Enter
   the observed unit/hour rate and starting balance. Use `None` only when the
   resource panel does not provide an observation. Set
   `ISOLATED_RUNTIME_CONFIRMED = True` only after checking other runtimes are
   disconnected. This marks whether the balance change covers only this session.
3. Run the cells in order. Storage and logging start before clone and dependency
   installation. Sampler preflight and correctness tests must pass before the
   training command runs.
4. Keep the frozen configuration. The cache check requires the pinned receipt,
   manifests, database tables, and both `train.parquet` and `val.parquet`.
   Notebook 03's receipt alone is insufficient because it downloaded validation
   labels only. Missing inputs trigger a download; matching local files are reused.
5. Review the live step losses, report status, subset metrics and trained-mode
   logit checks. Do not enlarge the workload or start a longer training run from
   these preliminary outputs automatically.
6. Enter the observed ending balance in the optional balance cell. Download the
   final ZIP, confirm it contains `checkpoints/pilot.pt` when a checkpoint was
   produced, and keep it outside the VM. Disconnect and delete the runtime.

The 50-unit reserve is enforced before setup when the supplied starting balance
is at or below it. Continue checking the resource panel during the session. Null
observations remain null. This completed export recorded null start/end balances
and a null rate despite its isolation flag being true. A later user follow-up
reported 198.63 units remaining after the preceding 198.67 observation: the
0.04-unit interval decrease cannot establish this pilot's isolated cost. The
latest reported balance leaves 148.63 units above the reserve. See the
[separate budget follow-up](../reports/budget/20261024-notebook04-followup.json).
An observed account decrease is
attributed to this run only when the interval was isolated and no balance
increase occurred. Deltas use two decimal places, matching Colab's displayed
balance precision; overlapping runtimes cannot provide an isolated run cost.

## Evidence and resuming

Run artifacts live under `/content/temporary-trgnn-training/runs/<run-id>/`.
`training-pilot.json` records status, split/query provenance, feature assumptions,
raw step losses and timing history, before/after validation predictions and
metrics, the train-fitted activity baseline, and trained-checkpoint inference
parity. Binary cross-entropy, ROC AUC and average precision are subset metrics.
A `passed` status needs completion of the requested updates and the validation
and parity checks; inspect the reported completed step count alongside it.

The checkpoint preserves the model, optimizer, random state, completed step,
configuration and selected query provenance. Deterministic step seeds preserve
the intended sampling schedule when resuming. Its file hash and provenance
identify the checkpoint used for evaluation. Ordinary failures retain partial
history and the available checkpoint; a hard VM loss can erase unexported files.

To continue an interrupted pilot in the same VM, set `RESUME = True` and rerun
the first control cell, the restore cell and the training cell. The original run
ID and starting budget observation are preserved; an existing checkpoint is
reused. Avoid rerunning setup when it has already passed and dependencies have
not changed. The training command validates the saved contract before continuing.

After a VM reset, extract `checkpoints/pilot.pt` from your own earlier ZIP, open
the notebook again, set `RESUME = True`, and run setup. Set `RESUME_CHECKPOINT`
to its restored path, or leave it `None` and select that file in the restore
cell's upload prompt. The file is copied into a new run folder before `--resume`.
Set `PROJECT_REF` to the source commit recorded in the earlier `checkout.log`.
Code hashes, dependency versions, the frozen configuration, seed, selected rows
and model contract must match; the command stops if that contract has changed.
Initial validation metrics remain in the original archive rather than being
remeasured from the restored checkpoint as though it were the initial model.
Resuming a checkpoint at step 40 completes the remaining 60 updates of the
100-step target; it does not request 100 additional updates.

A fresh resumed VM session has its own start/end balance interval. Keep both the original
and resumed archives to preserve the full history. The ZIP includes checkpoints,
configuration, reports and logs, excluding the database. Export after failures
as well as success, and stop GPU use once the export is saved.
