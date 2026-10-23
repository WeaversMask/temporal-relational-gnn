# First hosted real-batch diagnostic: verified 2026-10-24

Notebook 02 completed successfully in the user-exported run
`20261023T223033Z-4cb94d`. All seven exported files are preserved verbatim in
[`colab/20261023T223033Z-4cb94d/`](colab/20261023T223033Z-4cb94d/), with a
[SHA-256 provenance manifest](colab/20261023T223033Z-4cb94d/PROVENANCE.json).
The recorded project revision is
`31d6b15910cb537d8f8024d11e97fa87d3fd8d34`; its working tree was clean.

The runtime was a Tesla T4 with Python 3.13.16, PyTorch 2.11.0+cu130, PyG 2.8.1,
`pyg_lib` 0.9.0+pt211cu130, PyTorch Frame 0.3.0, and RelBench 3.0.2. CPU execution
used one thread. The correctness suite passed **26 tests**, including the CUDA
test, and the compiled temporal sampler preflight passed.

## What passed

The pinned `rel-stack/user-badge` database downloaded and its historical graph
was built with seven tables and 22 directed relationship types. Projection
retained 4,121,747 rows at the maximum selected query timestamp,
`2020-10-01 00:00:00`. Invalid references removed after this cutoff included
35 post-owner references and 74 vote-post references.

Three batches of eight validation queries passed target eligibility, sampled
record-time visibility, disjoint edge ownership, and directed BFS prefix checks.
The same random-weight RelBench feature/time encoders, GraphSAGE layers, and
linear readout were used across reference, HGAM, and blocks. HGAM logits matched
the reference exactly; the largest absolute block logit difference was
`1.1920928955078125e-7`. The loss comparisons passed the existing tolerances.

## Measured forward passes

Each entry is the median of ten interleaved repetitions on one already sampled
and transferred batch. Forward time includes feature encoding, relative-time
encoding, message passing, and readout. It excludes sampling, feature gathering,
transfer, graph building, and download.

| Batch | Sampled nodes | Sampled edges | Reference ms | HGAM ms | Blocks ms | Blocks latency reduction vs HGAM | Blocks latency reduction vs reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 60 | 114 | 21.268 | 20.138 | 18.292 | 9.17% | 13.99% |
| 2 | 130 | 225 | 21.684 | 21.050 | 18.349 | 12.83% | 15.38% |
| 3 | 68 | 107 | 20.037 | 19.207 | 17.276 | 10.05% | 13.78% |

Latency reduction is `100 * (1 - blocks_median / comparison_median)`, computed
from the unrounded measurements in the JSON. Sampling plus feature gathering
was recorded once per batch: 2.719, 3.785, and 3.872 ms. Transfer was unmeasured,
so adding these values to forward medians would not establish complete
inference latency. The raw forward repetitions include outliers up to 30.8 ms.

The CUDA memory measurement is additional peak allocated memory during one
forward pass above the resident model/batch allocation:

| Batch | Reference KiB | HGAM KiB | Blocks KiB | Blocks allocation reduction vs HGAM | Saved vs HGAM KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 111.0 | 95.5 | 77.0 | 19.37% | 18.5 |
| 2 | 263.5 | 230.0 | 142.0 | 38.26% | 88.0 |
| 3 | 141.5 | 124.0 | 79.0 | 36.29% | 45.0 |

These are small absolute savings in extra allocations. They do not measure
total VRAM, allocator reservations, CUDA context memory, or peak host RAM.

## Evidence limits and next work

This establishes an actual hosted-data feasibility result with an **untrained**
model and constant node features plus relative time. It does not establish
predictive quality, trained-model equivalence, or end-to-end acceleration. The
24-query workload has tiny sampled neighborhoods, one configuration, one random
model, and one runtime. All measured per-table repeated-global-ID counts were
zero; real repeated temporal occurrences were therefore untested in these
three batches. The separate tiny sampler fixture and synthetic tests cover
repeated IDs. The export records the maximum query timestamp but omits the
individual timestamps, so timestamp diversity cannot be verified from it alone.
Record-time visibility alone does not certify the historical availability of
every retained relationship value from the hosted snapshots; the feature and
relationship availability policy still needs review.

Actual compute-unit consumption and remaining units were both left null. The
planned budget remains 200 units with a 50-unit reserve; no observed rate or
remaining balance can be inferred from this export.

The temporal feature availability audit and a pipeline profile remain pending
their hosted run. Follow [the next-run instructions](../docs/NEXT_RUN.md) and
notebook `03_audit_profile.ipynb` before choosing training features, workload,
or a training budget. Model training and the final predictive/end-to-end
evaluation remain later milestones.
