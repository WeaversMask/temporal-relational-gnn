# Efficient Inference for Temporal Relational GNNs

A focused investigation of destination-restricted block execution for temporal,
heterogeneous GraphSAGE. Compare the same weights and sampled occurrences under
three schedules: full reference execution, PyG HGAM, and bipartite blocks.

**Current stage:** notebook 03 passed on a Colab T4 with 36 passing tests,
a passed feature-policy metadata audit, and paired resident-graph online
pipeline measurements. The model remains untrained with constant features plus
relative time. Historical feature/relationship availability, a representative
frozen workload, training, and final trained evaluation remain pending. See the
[pipeline profile results](reports/PIPELINE_PROFILE_STATUS.md),
[real-batch results](reports/REAL_BATCH_STATUS.md),
[verified Colab status](docs/COLAB_STATUS.md), and
[next-run instructions](docs/NEXT_RUN.md). Raw exports are in `reports/colab/`.

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
is available for deliberate reproduction. Review the
[completed run and remaining decisions](docs/NEXT_RUN.md) before choosing
historical features, a representative workload, or a training budget.

Code stays in Git; large data stays out of Git. Run active data processing on the
VM's local disk. Keep durable checkpoints, caches, configurations, and raw logs
in Drive. Colab VM contents are disposable. GPU access, RAM, and session lifetime
are variable, so use short, resumable runs. Do not leave allocated GPUs idle
while coding or waiting for agents.

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
`configs/upstream.json`; experiment settings remain provisional until the
historical availability review and workload selection are complete. Notebook 03's
metadata audit and small untrained profile do not freeze that workload.

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
