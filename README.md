# Efficient Inference for Temporal Relational GNNs

A focused investigation of destination-restricted block execution for temporal,
heterogeneous GraphSAGE. Compare the same weights and sampled occurrences under
three schedules: full reference execution, PyG HGAM, and bipartite blocks.

**Current stage:** runnable synthetic correctness and profiling prototype. Real
RelBench training, real-batch parity, CUDA validation, and end-to-end results are
not yet complete. No acceleration claim has been established.

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
`configs/upstream.json`; experiment settings remain provisional until a real
Colab batch establishes feasibility.

The standalone layers mirror the inspected RelBench GraphSAGE configuration;
`trgnn.execution.run_layers` also accepts the existing model's `convs` and `norms`
for later integration. The actual sampler smoke check is a tiny PyG test, not a
RelBench task audit.

## What is measured now

The synthetic harness creates directed, disjoint BFS batches with empty
relationships, isolated seeds, same-table edges, and repeated global IDs.
Features are already encoded. It verifies seed embeddings and times the
message-passing core, including construction of block views.

It **excludes** sampling, feature gathering, CPU/GPU transfer, encoders, and the
prediction head. Synthetic timing must never be described as task-level,
end-to-end, or trained-model performance. The benchmark saves raw interleaved
measurements, output errors, device details, package versions, and Git revision.

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
