# Efficient Inference for Temporal Relational GNNs

A focused investigation of destination-restricted block execution for temporal,
heterogeneous GraphSAGE. Compare the same weights and sampled occurrences under
three schedules: full reference execution, PyG HGAM, and bipartite blocks.

**Current stage:** notebook 07 completed the timing-pause diagnostic on T4:
**200 tests passed** and all **332 cases** completed with the original weights,
contracts and sampled-input fingerprints. Four new slow online passes
(251.483–266.993 ms) contained full Python garbage collections lasting
227.667–239.758 ms. The old batch-512 position was normal in this traced run.
This strongly supports garbage collection as an explanation consistent with
the historical spikes; the original untraced events cannot be proved retroactively.
Two other early slow passes had no collection, so collection does not explain
every pause.

The original notebook 06 primary reductions of **8.91% and 8.77%** remain
observations under the old protocol, both below the 10% target. Diagnostic
controls do not supply corrected speedup estimates: both short automatic
controls had no full collections, and forced collection moved **88.656 seconds**
of cleanup outside executor timers. The new
[notebook 08](docs/GC_AWARE_INFERENCE.md) is prepared with cycle-free auditing,
natural automatic cleanup retained in online timings, and separate accounting.
Its hosted result is pending. See the
[current diagnostic findings](reports/SPIKE_DIAGNOSTIC_STATUS.md),
[original two-run report](reports/TRAINED_INFERENCE_REPRODUCTION.md),
[training results](reports/TRAINING_RUN_STATUS.md),
[verified Colab status](docs/COLAB_STATUS.md), and
[current next steps](docs/NEXT_RUN.md). Raw exports are in `reports/colab/`.
The source activity baseline remains the stronger predictor; historical snapshot
assumptions, constant/time features and the single-checkpoint/schema/date scope
remain explicit. Existing notebooks remain available for deliberate reproduction.

## Start in Colab

[Open the feasibility notebook](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/01_feasibility.ipynb).
The repository is private: use your authorized GitHub account to open the
notebook. If Colab cannot open it directly, download the notebook from GitHub
and upload it to Colab. The notebook explains its separate runtime clone step.

1. Add a read-only, repository-scoped GitHub token to **Colab Secrets** as
   `GITHUB_TOKEN`, and allow this notebook to access it. Clone authentication is
   temporary; the token is not embedded in a clone URL or written to the repo.
2. Start with a CPU runtime. Run setup, the correctness suite, environment
   capture, a tiny synthetic benchmark, and the public manifest inspection.
3. Persist runs to Google Drive using the notebook's `RUN_DIR`.
4. Select a GPU runtime for the CUDA stage, rerun setup after a runtime reset,
   enable `RUN_GPU`, and run the GPU checks. Install `pyg-lib` with the supplied
   wheel helper before requesting the actual temporal-sampler smoke check.
5. Record the consumption rate displayed by Colab in `colab-budget.json` for
   each run. These are observations, not hard-coded GPU prices.

After those checks pass, open
[notebook 02: actual relational batches](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/02_real_batch.ipynb).
It uses temporary storage by default, streams its logs, and includes a ZIP export
cell. This avoids the Drive authorization problem encountered during initial setup.
The real database download is substantially larger than the manifest-only check.

Notebooks 02 and 03 have now passed; their exported evidence is archived in this
repository. [Notebook 03: feature audit and pipeline profile](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/03_audit_profile.ipynb)
is available for deliberate reproduction. The completed
[notebook 04: first training pilot](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/04_training_pilot.ipynb)
is documented in the [frozen pilot instructions](docs/TRAINING_PILOT.md) and
[current status notes](docs/NEXT_RUN.md). It kept constant/time features, used
official training rows and a 512-query validation subset, left the test split
untouched, and exported a resumable checkpoint. Its subset metrics are preliminary,
and the pilot did not benchmark trained inference speedup.

[Notebook 05](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/05_training_run.ipynb)
completed 2,000 updates over 64,000 selected training queries and an 8,192-query
validation subset. Both models fitted the same training cohort; the GNN started
from fresh weights. The exported checkpoint and full history are preserved.
See [the results](reports/TRAINING_RUN_STATUS.md) and
[reproduction instructions](docs/TRAINING_RUN.md).

[Notebook 06](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/06_trained_inference.ipynb)
uses that saved checkpoint for inference only. Upload notebook 05's
`checkpoints/training.pt` when prompted. The benchmark verifies source/data
compatibility and all saved validation predictions before timing. It persists
raw paired batch results, resumable progress, summaries and GPU telemetry;
both independent runs are archived. For deliberate reproduction, use a fresh
session with `RESUME = False`; see the [measurement protocol](docs/TRAINED_INFERENCE.md).

[Notebook 07](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/07_spike_diagnostic.ipynb)
completed the instrumented follow-up with the same checkpoint and saved-input
contracts. Its callbacks and host spans perturb allocation and timing, so its
measurements diagnose pauses rather than replace notebook 06 speed results.
See the [completed diagnostic and reproduction instructions](docs/SPIKE_DIAGNOSTIC.md).

Code stays in Git; large data stays out of Git. Run active data processing on the
VM's local disk. Save checkpoint/report ZIP exports outside the VM; notebooks 04
through 07 include their checkpoints in the download without mounting Drive. Colab VM
contents are disposable. Use one active runtime, record start/end balances, and
keep the 50-unit reserve. GPU access, RAM, and session lifetime are variable, so
use short, resumable runs. Disconnect GPUs once the export is saved.

## Local development

Python 3.11+; Python 3.12 is the development reference.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/cpu-py312.txt
python -m pip install -e '.[dev,relbench]' --no-build-isolation
python -m pytest -q
python -m trgnn.preflight --output results/preflight.json
python -m trgnn.benchmark --device cpu --output results/synthetic.json
```

CPU graph-core tests need no `pyg-lib` and no database download. The optional
`relbench` extra enables direct parity against the published reference model.
CUDA is checked separately, with an explicitly skipped test on CPU.
For an actual sampler check:

```bash
python scripts/install_sampler.py
python -m trgnn.preflight --check-sampler --output results/sampler.json
```

The binary wheel helper is primarily for Linux/Colab. A compatible wheel may
not exist for every local platform. This must not block CPU graph-core tests.

## Research question

Does restricting destination outputs at each layer improve latency or memory
beyond PyG's existing neighborhood trimming, while preserving supervised outputs?

The intended real task is `rel-stack/user-badge`: predict a future badge award
using the historical relational graph. See [the plan](docs/PLAN.md) and
[the execution contract](docs/EXECUTION.md). Source and dataset revisions are in
`configs/upstream.json`. The measured constant/time workload is frozen in
`configs/trained_inference.json` under its documented exported-snapshot
assumptions; broader historical availability remains unresolved.

The standalone layers mirror the inspected RelBench GraphSAGE configuration;
`trgnn.execution.run_layers` also accepts the existing model's `convs` and `norms`
for real-model integration. Notebook 02 uses the actual RelBench encoders,
GraphSAGE, and readout on hosted batches. The separate sampler smoke check
remains a tiny PyG test.

## What is measured now

The synthetic harness creates directed, disjoint BFS batches with empty
relationships, isolated seeds, same-table edges, and repeated global IDs.
Features are already encoded. It verifies seed embeddings and times the
message-passing core, including construction of block views.

It **excludes** sampling, feature gathering, CPU/GPU transfer, encoders, and the
prediction head. Synthetic timing must never be described as task-level,
end-to-end, or trained-model performance. The benchmark saves raw interleaved
measurements, output errors, device details, package versions, and Git revision.

Notebook 02 separately measures untrained forward passes on actual relational
batches, including the feature/time encoders and readout. Blocks reduced median
forward latency by 9.17–12.83% versus HGAM in its three tiny batches of 60–130
sampled nodes. Extra CUDA allocation savings were 18.5–88 KiB. These measurements
exclude sampling, gathering, and transfer and do not establish trained-model
or complete inference acceleration. See the
[real-batch report](reports/REAL_BATCH_STATUS.md) for raw evidence and limits.

Notebook 03 separately measured serial sampling, gathering, transfer, encoders,
message passing and readout from an already built resident graph. Blocks reduced
median online pipeline latency by 9.04–11.68% versus HGAM across three tiny
untrained batches. Setup was excluded; graph construction took 54.509 seconds.
The observed Colab rate was 1.07 units/hour, but per-run cost is unmeasured.
See the [pipeline profile report](reports/PIPELINE_PROFILE_STATUS.md) for paired
raw evidence, memory definitions, timing noise, and remaining limitations.

Notebook 04 completed 100 supervised updates over 3,200 training queries and
39 prediction timestamps. On the fixed 512-query validation subset, the trained
GNN reached ROC-AUC 0.85419 / AP 0.13373; the train-fitted activity baseline
reached 0.85585 / 0.14655. One trained validation batch matched all three
execution modes exactly. No trained inference speedup or isolated run cost was
measured. See the [pilot report](reports/TRAINING_PILOT_STATUS.md).

Notebook 05 completed a one-pass reference-training run and a larger fixed
validation comparison. The activity baseline had a substantial average-precision
advantage over the trained GNN. Training-call wall time was 3 minutes 34 seconds;
allocated/reserved training peaks were 53.742 / 58 MiB. These are allocator
measurements, rather than total device use or compute utilization. See the
[training run report](reports/TRAINING_RUN_STATUS.md). The ending balance was
198.59 units, leaving 148.59 above the reserve; notebook 05 isolated run cost
was unmeasured.

Notebook 06 measured the fixed trained model over all 8,192 validation queries
at batch sizes 32, 128 and 512, then reproduced the same protocol in a fresh
session on a different T4. The primary online reductions versus HGAM were
8.91% and 8.77%, both below the 10% target. The consistent benefit is about
17% in message passing and 9% in the graph-loaded online pipeline. In the
second run, every query-batch median favored blocks over both baselines.
Retain all trials, including the recurring secondary batch-512 extreme; its
cause is unresolved. See the [two-run report](reports/TRAINED_INFERENCE_REPRODUCTION.md)
for sensitivity diagnostics and scope.

The first run's blocks throughput rose from 1,328 to 19,280 queries/second
across sizes, a 14.52-fold batching observation with different grouping and
sampling. Its device-wide memory samples were 161–171 MiB, distinct from
PyTorch allocator peaks. Reproduction budget exports recorded 197.90 to
197.46 units at 1.07 units/hour with isolation confirmed: 0.44 units consumed,
leaving 147.46 above reserve. The first run's separately corrected 0.50-unit
interval is preserved in its historical budget record.

Notebook 07 recorded four generation-2 collections inside new slow online passes
across different modes and batch sizes, establishing collection as the immediate
source of most added duration in those new events. It did not trace the old
spikes, and no full collection occurred in either short automatic target-window
control. The cycle-free helper therefore has no demonstrated spike-removal
result. Forced cleanup was relocated outside original timers, rather than made
free. Its 664 case files are preserved in a verified compressed archive; root
reports and logs remain unchanged. See the
[diagnostic report](reports/SPIKE_DIAGNOSTIC_STATUS.md). The isolated budget
interval was **195.94 to 195.36 units**, or **0.58 consumed** at an observed
1.07 units/hour, leaving **145.36 above reserve**. Earlier account decreases
remain separate.

## Attribution and scope

- [RelBench](https://github.com/stanford-star/relbench): problem setting,
  architecture configuration, and intended real-data pipeline.
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric): GraphSAGE,
  heterogeneous convolution, node normalization, and HGAM implementation.
- [PyG HGAM](https://pytorch-geometric.readthedocs.io/en/latest/advanced/hgam.html)
  and [DGL message-flow blocks](https://www.dgl.ai/dgl_docs/guide/minibatch-node.html):
  established execution principles. Block execution is not claimed as new.

The contribution being investigated is adaptation, correctness coverage, and
controlled evaluation in a temporal relational pipeline. One schema does not
establish cross-schema transfer or a foundation model. Upstream licenses apply
to their respective software and datasets. This private project has no public
license grant at this stage.
