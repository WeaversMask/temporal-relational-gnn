# Bounded hosted training run: verified 2026-10-24

Notebook 05 passed in run `20261024T073010Z-1f9800`, completing **2,000
reference optimizer updates over 64,000 selected training query occurrences**.
All **81 tests passed on the Tesla T4**, and sampler preflight passed. The
thirteen exported report, configuration and log files are preserved verbatim in
[`colab/20261024T073010Z-1f9800/`](colab/20261024T073010Z-1f9800/), with a
[SHA-256 provenance manifest](colab/20261024T073010Z-1f9800/PROVENANCE.json).
The clean source revision was `c9879d3bcb5cd76814bd9bc088a9ea4481aa0a40`.

## Training and evaluation scope

This fresh model used 64 channels, two layers, fanouts `[16, 8]`, batches of 32,
Adam with learning rate 0.001, and seed 42. The training cohort had 3,065
positives, 54,030 unique users and 40 prediction timestamps from `2010-10-14`
through `2020-07-02`. All selected rows were consumed once. Every update
recorded finite gradients and passed sampled time visibility, disjoint ownership
and BFS prefix checks. Eighteen training batches contained repeated global IDs;
local occurrences remained distinct. The raw per-type counts, query indices,
input hashes, sampler fingerprints and loss/timing history are retained.

GNN features remained constants and relative record time, with literal constant
normalizer statistics independent of validation. The activity baseline fitted
its scaler and logistic regression on the same training cohort. Neither model
used task labels as node features or raw mutable/text attributes. The official
test split remained untouched. The exported snapshot still does not reconstruct
historical foreign-key edits, deletions, merges or survivorship; the result is
conditional on that historical graph contract. Broader feature approval remains
unresolved.

All reported predictors below use the same **8,192 official validation queries**,
with **229 positives (2.795%)**, at `2020-10-01`. This is a fixed subset of
validation, rather than full task evaluation.

| Predictor | ROC-AUC | Average precision | Binary cross-entropy |
| --- | ---: | ---: | ---: |
| Initial random-weight GNN | 0.50040 | 0.02585 | 1.03132 |
| GNN after 2,000 updates | 0.83194 | 0.15439 | 0.11392 |
| Train-fitted historical-activity logistic baseline | 0.85721 | 0.27248 | 0.09880 |
| Constant training-prevalence baseline | 0.50000 | 0.02795 | 0.13265 |

The trained GNN improved over initialization and the constant baseline. The
activity baseline scored better on all three validation metrics, with a sizable
average-precision gap. These are point estimates from one seed and one bounded
schedule; statistical significance, convergence and superiority of either model
were not established. Notebook 04 used different training and validation
cohorts, so its metrics do not provide a controlled before/after quality trend
for this run.

## Timing, memory and trained execution checks

The training call took **214.262 seconds (3.57 minutes)** after setup. Its median
recorded step time was 89.766 ms, and the sum of timed work across 2,000 updates
was 185.104 seconds. Those regions cover sampling/gathering and
transfer/forward/loss/backward/finite-gradient checks/optimizer work. Query
scheduling, visibility/fingerprint audits, checkpoint writes and reporting are
excluded from `step_ms`; the complete training-call wall time includes them.
Graph construction took 56.046 seconds separately. The exported notebook budget
observations span 7 minutes 46 seconds and include work outside training.

The largest recorded training-step **PyTorch allocator** peaks were **53.742 MiB
allocated and 58 MiB reserved**, on a T4 reporting 15,637,086,208 bytes of total
device memory (14.56 GiB). Process-lifetime host peak RSS after training was
2.481 GiB. Allocator figures exclude CUDA context and other device allocations,
so they are not a measurement of total GPU usage. The user's approximate
0.2 GB Colab panel observation is consistent with a small workload but is not
the same measurement. The run establishes considerable memory headroom for
this configuration; it does not establish GPU utilization or predict the
speedup from larger batches.

The [PyTorch memory documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management)
distinguishes tensor allocations from allocator reservations.
[NVIDIA's utilization definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)
measures kernel activity over time separately from memory capacity.

After training, **one sampled validation batch** used the same trained weights
and sampled occurrences across reference, HGAM and blocks. All absolute logit
and loss differences were zero, passing the existing tolerances. This run
measured neither trained inference latency nor trained inference speedup.
Changing batch size or fanouts for a performance study still requires paired
measurements and equivalence checks on that workload.

## Checkpoint, runtime and budget evidence

The final checkpoint records step 2,000, is 20,438,764 bytes, and has SHA-256
`92d02f7d2007c0ab560629e9c4e0d48886a1d149930e68e494a861b85659e1d6`.
The binary is excluded from Git; provenance records its hash, size and ignored
local storage path. An independent CPU load with `weights_only=True` verified
that saved step count, full history, query order and input contract match the
report, with finite model and optimizer tensors. The contract SHA-256 is
`7f141d0b0b5359b569533724d7155c263d33ef36e2c8dcfd40ee389be3a7e1de`.
This attempt did not resume, so it does not itself establish
a hosted interrupted-run resume result. The runtime used Python 3.13.16,
PyTorch 2.11.0+cu130, PyG 2.8.1, `pyg_lib` 0.9.0+pt211cu130, RelBench 3.0.2,
torch-frame 0.3.0 and one CPU thread; full package versions are archived.

The exported balances were **198.63 at start and 198.59 at end**, with an
observed rate of 1.07 units/hour. Their **0.04-unit account decrease** is not an
isolated notebook cost because `isolated_runtime_confirmed` was false. The
latest balance leaves **148.59 units above the 50-unit reserve**. Original
budget files remain unchanged; the
[budget summary](budget/20261024-notebook05-observations.json) preserves the
attribution limit. No budget-only rerun is needed.

The bounded training run is complete. Remaining investigation work includes
paired inference measurements using the frozen trained checkpoint, workload
coverage beyond this small batch configuration, and any separately planned
predictive-quality evaluation.
