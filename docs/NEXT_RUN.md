# Next run: the first 100-step training pilot

Notebooks 02 and 03 passed their hosted-data diagnostics. Notebook 03 run
`20261023T225501Z-efdf72` includes 36 passing T4 tests, a passed feature-policy
metadata audit and three paired resident-graph online pipeline batches. Read
[the measured results and limitations](../reports/PIPELINE_PROFILE_STATUS.md);
its export and hashes are archived under
[`reports/colab/20261023T225501Z-efdf72/`](../reports/colab/20261023T225501Z-efdf72/).

The next prepared run is [notebook 04: training pilot](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/04_training_pilot.ipynb).
It runs 100 reference-training updates with constant features plus relative time,
compares a train-fitted historical-activity baseline, evaluates a fixed
512-query official validation subset, and checks inference parity across
reference, HGAM and blocks using the resulting checkpoint. The test split stays
untouched. Its hosted-data GPU run and predictive results remain pending.

Read [the pilot instructions](TRAINING_PILOT.md) before running. The first pilot
configuration is frozen in
[`configs/training_pilot.json`](../configs/training_pilot.json); it does not freeze
the final experiment or approve additional historical raw features. Relationship
changes, deletions, merges and snapshot survivorship remain explicit assumptions.
The metadata audit still reports `training_ready: false`.

Use one active Colab runtime and disconnect other managed sessions before
starting. Enter the current resource-panel unit/hour rate and starting balance;
record the ending balance after the run. Leave unavailable values null. The
50-unit reserve applies throughout. The earlier 1.07-unit/hour rate and
198.67-unit balance are prior observations, not presets for the new session.
Notebook 03's earlier 0.11-unit account decrease involved overlapping runtimes
and does not establish its individual cost.

Notebook 04 reuses matching downloaded data, but also needs official training
labels. A matching notebook 03 receipt without `train.parquet` triggers a
download. Do not start another full download just to reproduce completed
notebooks 02 or 03. They remain available for deliberate reproduction when needed.

Keep the final ZIP: it contains the resumable checkpoint, configuration, raw
step history, reports and logs, excluding the database. Export it after success
or failure, then disconnect and delete the runtime. On a later session, restore
your own earlier `pilot.pt` and explicitly enable resume with the same frozen
configuration. The target remains 100 total updates. See
[resuming the pilot](TRAINING_PILOT.md#evidence-and-resuming).

A completed pilot will inform a later decision about historical features,
representative workload and training budget. It cannot establish final predictive
quality or trained-model inference acceleration on its own.
