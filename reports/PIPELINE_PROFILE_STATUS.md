# Hosted pipeline profile: verified 2026-10-24

Notebook 03 passed in the user-exported run `20261023T225501Z-efdf72`.
All eleven source files are preserved verbatim in
[`colab/20261023T225501Z-efdf72/`](colab/20261023T225501Z-efdf72/), with a
[SHA-256 provenance manifest](colab/20261023T225501Z-efdf72/PROVENANCE.json).
The recorded project revision is `c3f5f6e5e64d13dc3fdfb57abdbfd704f0078ed7`,
with a clean working tree. Every reported pipeline and stage median was independently
recomputed from the raw repetitions.

The Tesla T4 runtime used Python 3.13.16, PyTorch 2.11.0+cu130, PyG 2.8.1,
`pyg_lib` 0.9.0+pt211cu130, PyTorch Frame 0.3.0, RelBench 3.0.2, pandas 2.2.3,
and PyArrow 23.0.1. CPU execution used one thread. All **36 tests passed**,
including CUDA and compiled-sampler checks, and sampler preflight passed.

## Correctness and historical inputs

Three batches of eight validation queries passed all paired comparisons, sampled
record-time visibility, disjoint edge ownership, and BFS prefix checks. Each of
the 30 trials records a full sampler fingerprint; the profiler checked equality
across execution modes and each stage replay outside the timing regions. HGAM
logits matched the reference exactly. The maximum absolute block logit and loss
errors were each `1.1920928955078125e-7`, within the existing tolerances.

The embedded feature-policy metadata audit passed with no errors. All seven
table schemas were classified, raw feature columns remained empty, and inputs
remained constants plus relative record time. No preprocessing was fitted and
`training_ready` remains `false`. This does not resolve historical foreign-key
changes, merges, deleted records, survivorship, or candidate attribute values;
the complete historical availability milestone remains open.

All 24 query timestamps were `2020-10-01`. Every per-table repeated-global-ID
count was zero across all 30 trials. This run therefore adds no hosted evidence
for varied query times or repeated temporal occurrences. Sampled node counts
ranged from 59–63, 124–139, and 68–73 in the respective batches; edge counts
ranged from 114–114, 218–233, and 106–112. These are small neighborhoods.
The validation row cohort matches notebook 02, but sampled neighborhoods differ.
Cross-notebook timings are therefore not paired comparisons of pipeline stages.
The manifest hashes and selected validation row IDs match notebook 02, but
sampled neighborhoods differ between runs and between trials. Subtracting
notebook 02 forward medians from these pipeline medians would not isolate
sampling, gathering or transfer costs; the runs are not paired measurements.

## Preliminary online pipeline measurements

The measured boundary starts with serial sampling from an already built CPU
graph, includes feature gathering, device transfer, feature/time encoders,
message passing and readout, and ends at logits. The model is resident on the
GPU. Download, projection, graph/CSC construction, model initialization,
correctness checks, DataLoader scheduling and output serialization are excluded.
The raw report's `end_to_end: true` applies to this resident-graph online
boundary; it does not establish a final trained application speedup.

Each entry is the median of ten interleaved repetitions after two warmup passes
per mode. Samples change between trials and are identical across paired modes.

| Batch | Reference ms | HGAM ms | Blocks ms | Blocks reduction vs HGAM | Blocks reduction vs reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 28.098 | 27.662 | 24.430 | 11.68% | 13.05% |
| 2 | 26.904 | 27.712 | 24.661 | 11.01% | 8.34% |
| 3 | 27.230 | 25.233 | 22.953 | 9.04% | 15.71% |

Reductions use the unrounded medians and `100 * (1 - blocks / comparison)`.
These preliminary results support profiling feasibility for this untrained
configuration. They do not establish the 10% target on a representative frozen
workload: there is one runtime, one random-weight model, one query timestamp,
and only three tiny batches.

Separate synchronized replays measured each stage. Across the three batches,
HGAM median encoding took 9.174–10.206 ms and message passing 14.013–14.745 ms;
blocks took 8.866–9.185 ms and 11.402–11.541 ms respectively. Sampling,
gathering and transfer were measured rather than omitted. The stage replays
have a different synchronization boundary from whole-pipeline passes. Their
medians must not be added to manufacture a pipeline total or stage proportions.

Raw measurements show noise: batch 2 includes pipeline observations of
48.483 ms for reference and 43.916 ms for HGAM. Its separate reference stage
replay at sampler seed 60 recorded a 192.752 ms feature-gathering outlier.
That replay is separate from the measured online pipeline total and does not
change the reported pipeline medians.

## Setup, memory, and compute budget

Setup was measured separately: graph construction took **54.509 seconds**,
projection 1.121 seconds, loader construction 1.529 seconds, and model
initialization 0.291 seconds. Download/verification took 5.315 seconds in this
runtime; the export does not establish fresh-download performance or cache
reuse. Setup and its amortization are outside the latency table.

Peak host RSS reached **2.0415 GiB**, a process-lifetime high-water mark that
includes imports and setup. CUDA allocated peaks ranged from
**14.564–14.755 MiB**. These are this process's PyTorch allocator measurements,
not total device VRAM or driver/context memory. Reserved memory includes the
warm allocator cache from earlier modes and replays and cannot be attributed
to one mode in isolation.

Median extra CUDA allocations above the resident allocation were:

| Batch | Reference KiB | HGAM KiB | Blocks KiB | Saved vs HGAM KiB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 128.00 | 112.00 | 92.50 | 19.50 |
| 2 | 263.75 | 229.00 | 140.25 | 88.75 |
| 3 | 159.25 | 141.50 | 92.75 | 48.75 |

These savings are small in absolute terms, about 0.13%, 0.59%, and 0.33% of
the corresponding allocated peaks. They do not imply similar total-VRAM savings.

The recorded Colab rate was **1.07 units/hour**, with **198.78 units remaining**
at the initial observation. The planned budget remains 200 units with a 50-unit
reserve. No ending balance or measured per-run consumption was recorded, so
the actual cost of this run cannot be calculated from the export.

Historical feature/relationship evidence, representative workload selection,
training budget and checkpoint, predictive quality, and the final trained
evaluation remain pending. Next work is to settle historical input assumptions,
freeze a representative workload, and prepare a bounded training pilot with
checkpointing.
