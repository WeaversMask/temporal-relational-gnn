# Colab feasibility: verified 2026-10-24

The dependency checks and the first hosted real-batch diagnostic have passed.
The temporal feature audit and pipeline profile are the next hosted run;
training, predictive quality, and end-to-end acceleration remain unmeasured.
See [the detailed real-batch report](../reports/REAL_BATCH_STATUS.md) and
[next-run instructions](NEXT_RUN.md).

## Initial dependency and synthetic checks: 2026-10-23

Two independent exported runs are preserved verbatim with SHA-256 manifests in
`reports/colab/`. The second export includes the subsequently completed sampler
check from the same runtime. Original project revision: `4199f13`.

- CPU-only runtime: 17 tests passed; one CUDA test skipped.
- T4 runtime: 18 tests passed, including CUDA seed-output equivalence.
- Actual PyG temporal disjoint sampler: passed the small two-query visibility test.
- Pinned database/task manifests: retrieved successfully.
- GPU stack: Python 3.13.16, PyTorch 2.11.0+cu130, PyG 2.8.1,
  `pyg_lib` 0.9.0+pt211cu130, PyTorch Frame 0.3.0, RelBench 3.0.2.
- Reported Colab package versions are observations, not an independently tested
  environment installation lockfile.
- Compute-unit consumption was not recorded; the budget values are null.
- Drive mounting failed; the user completed checks with temporary storage and
  exported the results manually.

On the larger synthetic T4 batch, the median core latency was 5.363 ms for the
reference, 4.775 ms for HGAM, and 4.383 ms for blocks. Blocks reduced median
latency by 8.2% and extra peak allocated memory by 93.3% versus HGAM in this one
configuration. These are untrained synthetic message-passing measurements,
excluding sampling, transfer, encoders, and readout. Extra allocation is not total
VRAM. They do not establish real-data or end-to-end improvement.

## Hosted relational batches: completed 2026-10-23 UTC

Notebook 02 passed in the user-exported run `20261023T223033Z-4cb94d`, recorded
at project revision `31d6b15910cb537d8f8024d11e97fa87d3fd8d34` with a clean
working tree. All seven original files are archived verbatim with SHA-256
hashes in [`reports/colab/20261023T223033Z-4cb94d/`](../reports/colab/20261023T223033Z-4cb94d/).
The correctness suite passed 26 tests on the T4, including CUDA equivalence,
and the compiled sampler preflight passed.

The pinned `rel-stack` database, schema/task manifests, and validation labels
downloaded successfully. The projected historical graph retained seven tables,
22 directed relationships, and 4,121,747 rows at the maximum selected query
timestamp, `2020-10-01`. Three batches of eight queries passed eligibility,
sampled time visibility, disjoint ownership, BFS prefix, logit, and loss checks.
The largest block logit error was `1.1920928955078125e-7`; HGAM matched exactly.

| Batch | Reference ms | HGAM ms | Blocks ms | Blocks latency reduction vs HGAM |
| --- | ---: | ---: | ---: | ---: |
| 1 | 21.268 | 20.138 | 18.292 | 9.17% |
| 2 | 21.684 | 21.050 | 18.349 | 12.83% |
| 3 | 20.037 | 19.207 | 17.276 | 10.05% |

These are ten-repetition forward medians from one untrained, constant-feature
model with relative-time encoding, including the actual RelBench encoders and
readout. The sampled batches contained only 60, 130, and 68 nodes. Extra peak
allocated CUDA memory fell by 19.37%, 38.26%, and 36.29% versus HGAM, which
corresponds to small absolute savings of 18.5, 88, and 45 KiB. These are not
total-VRAM savings. No repeated global IDs occurred in the measured batches.

Timing covers the forward pass on identical already sampled/transferred batches.
Neighbor sampling and gathering are recorded separately, and this is explicitly
not a complete inference throughput result. Transfer and peak host RAM were
unmeasured. Budget observations are still null. The next step is notebook 03's
temporal feature audit and pipeline profile; those real-data milestones remain
pending until their exported outputs are reviewed. Model training, useful
predictive quality, and the final evaluation remain later work.
