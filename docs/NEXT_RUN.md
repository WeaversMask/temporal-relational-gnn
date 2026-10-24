# Next run: 64,000 training queries

Notebook 04 passed in hosted run `20261024T051434Z-948a5d`: 69 T4 tests,
100 reference-training updates, a train-fitted activity baseline and exact
trained-checkpoint inference parity on one validation batch. Read
[the training pilot assessment](../reports/TRAINING_PILOT_STATUS.md).
Its thirteen report/configuration/log files and checkpoint metadata are archived
with [SHA-256 provenance](../reports/colab/20261024T051434Z-948a5d/PROVENANCE.json).
The checkpoint binary is kept outside Git.

The trained GNN reached ROC-AUC 0.85419 and AP 0.13373 on the fixed 512-query
validation subset. The historical-activity baseline reached 0.85585 and 0.14655.
There were only 17 positives. The learned models' small numerical difference
does not establish a reliable ranking, and neither is a finished predictor.
Trained inference speedup was not measured.

The completed pilot used the frozen
[`configs/training_pilot.json`](../configs/training_pilot.json), constant/time
GNN inputs and official training rows. It covered 39 training timestamps and
3,177 users across 3,200 query occurrences; two training updates contained
repeated global post occurrences. Test labels remained untouched. Broader
historical feature approval, representative final workload selection, a larger
training budget, full predictive evaluation and trained inference measurements
remain open. The feature audit still reports `training_ready: false` for richer
raw feature approval; snapshot edits, deletions, merges and survivorship have
not been reconstructed.

The next prepared experiment is
[notebook 05: larger training run](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/05_training_run.ipynb).
Its frozen configuration uses 64,000 selected training rows and 2,000 updates
at batch size 32, with an 8,192-query validation subset. It keeps the same
architecture, optimizer and features, and fits the GNN and activity baseline
on the same 64,000 rows. It starts from fresh weights and processes each
selected training row once. Hosted execution remains pending.

Read the [run instructions](TRAINING_RUN.md). Progress reports and checkpoints
save every 100 updates, while the final report retains every recorded step.
Recovery uses this run's own checkpoint and input contract. The new fixed
cohorts need not contain the earlier pilot cohorts. Once its results are
reviewed, freeze the checkpoint and representative workload for paired trained
inference measurements.

The latest user-reported balance is 198.63 units, leaving 148.63 above the
50-unit reserve. The 0.04-unit decrease since the preceding 198.67 observation
describes an account interval, not isolated notebook 04 consumption. Exported
start/end balances and the hourly rate were null even though the isolation flag
was true. See the [budget follow-up](../reports/budget/20261024-notebook04-followup.json).

Preserve the exported checkpoint and original archives. The
[pilot documentation](TRAINING_PILOT.md) records the completed configuration,
evidence boundaries and resume procedure if an interrupted pilot needs recovery.
Notebooks 02 and 03 remain earlier diagnostics; their untrained measurements
do not establish trained inference performance. The outstanding research work
is recorded in [the plan](PLAN.md).
