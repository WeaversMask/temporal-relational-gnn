# Colab feasibility: verified 2026-10-24

The dependency checks, hosted real-batch diagnostic, notebook 03 pipeline profile,
notebook 04 pilot and notebook 05's 2,000-update training run have passed. The
trained checkpoints passed exact inference parity on one batch. Subset metrics are measured;
finished predictive quality and trained inference acceleration remain unmeasured.
The metadata audit does not resolve broader historical feature availability.
See [the training pilot report](../reports/TRAINING_PILOT_STATUS.md) and
[remaining decisions](NEXT_RUN.md).

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
unmeasured in this notebook 02 export, and its budget observations were null.
The later notebook 03 run below adds the pipeline and memory measurements.

## Hosted feature-policy audit and pipeline profile: completed 2026-10-23 UTC

Notebook 03 passed in run `20261023T225501Z-efdf72`, at clean project revision
`c3f5f6e5e64d13dc3fdfb57abdbfd704f0078ed7`. All eleven exported source files
are preserved verbatim with [SHA-256 hashes](../reports/colab/20261023T225501Z-efdf72/PROVENANCE.json).
All 36 tests passed on the Tesla T4, including CUDA and compiled sampler checks.
The runtime also confirmed compatibility with pandas 2.2.3 and PyArrow 23.0.1.

All three paired batches passed the sampler fingerprint, visibility, ownership,
BFS prefix, logit and loss checks over ten trials each. The maximum block logit
error remained `1.1920928955078125e-7`; HGAM matched exactly.

| Batch | Reference ms | HGAM ms | Blocks ms | Blocks reduction vs HGAM |
| --- | ---: | ---: | ---: | ---: |
| 1 | 28.098 | 27.662 | 24.430 | 11.68% |
| 2 | 26.904 | 27.712 | 24.661 | 11.01% |
| 3 | 27.230 | 25.233 | 22.953 | 9.04% |

These medians cover serial sampling, gathering, transfer, encoding, message
passing and readout from an already built graph with a resident model. Setup,
DataLoader scheduling and serialization are excluded. Graph construction took
54.509 seconds. Peak process-lifetime host RSS was 2.0415 GiB; allocated CUDA
peaks were 14.564–14.755 MiB, which are not total VRAM. Stage costs come from
separate synchronized replays, not parts whose medians can be summed into the
measured pipeline total. Raw outliers and precise boundaries are documented in
[the detailed report](../reports/PIPELINE_PROFILE_STATUS.md).

The feature-policy metadata audit passed, approving only constants and relative
record time; `training_ready` remains `false`. All 24 queries share the timestamp
`2020-10-01`, and all repeated-global-ID counts were zero across the 30 trials.
These remain preliminary small-workload, untrained results. Historical feature
and relationship semantics, the representative frozen workload, training and
predictive evaluation remain pending.

The observed rate was 1.07 Colab units/hour and the initial remaining balance
was 198.78 units. The user later reported 198.67 remaining and a notebook 02
runtime left active in the background. The 0.11-unit account decrease covers
the observed interval, rather than an isolated notebook 03 cost. The user
believes the session rate remained 1.07 units/hour; a combined rate is
unmeasured. See the [budget follow-up](../reports/budget/20261024-notebook03-followup.json).
That earlier balance left 148.67 units above the planned 50-unit reserve.

## Hosted reference-training pilot: completed 2026-10-24 UTC

Notebook 04 passed in run `20261024T051434Z-948a5d`, at clean source revision
`78feba6ba26622b77e1530ae3ae93315a7857109`. All 69 tests passed on the Tesla
T4. Thirteen original report/configuration/log files are archived verbatim with
[SHA-256 provenance](../reports/colab/20261024T051434Z-948a5d/PROVENANCE.json).
The checkpoint binary stays outside Git; its hash and size are recorded there.

All 100 optimizer updates completed with finite gradients and passed temporal
occurrence audits. The 3,200-query training cohort covered 39 timestamps and
3,177 unique users; repeated global post IDs occurred in two training updates
(three extra occurrences total). The final one-batch comparison had zero logit
and loss differences across reference, HGAM and blocks. Trained inference
speedup was not measured.

| Predictor on the same 512-query validation subset | ROC-AUC | Average precision | BCE |
| --- | ---: | ---: | ---: |
| Initial GNN | 0.47807 | 0.03259 | 1.02604 |
| GNN after 100 updates | 0.85419 | 0.13373 | 0.13122 |
| Historical-activity baseline | 0.85585 | 0.14655 | 0.12523 |
| Constant training-prevalence baseline | 0.50000 | 0.03320 | 0.14812 |

Only 17 validation examples were positive. The activity baseline's slightly
better numerical metrics do not establish statistical superiority, and these
subset results are not finished predictive quality. Inputs and snapshot
assumptions stayed unchanged; broader historical feature approval, final
workload selection and full evaluation remain pending.

The exported start/end balance and hourly rate fields were null despite an
`isolated_runtime_confirmed: true` flag. A later user follow-up reported 198.63
units remaining. The decrease from the prior 198.67 observation is 0.04 units
over that interval and does not establish isolated notebook 04 cost. The
[separate follow-up](../reports/budget/20261024-notebook04-followup.json) preserves
those distinctions without rewriting the export. That earlier balance
left 148.63 units above the 50-unit reserve.

## Hosted larger training run: completed 2026-10-24 UTC

[Notebook 05](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/05_training_run.ipynb)
passed in run `20261024T073010Z-1f9800` at clean source commit
`c9879d3bcb5cd76814bd9bc088a9ea4481aa0a40`. All **81 tests passed on T4**.
All 64,000 selected queries were consumed once across 2,000 finite-gradient
updates with passed occurrence audits. Eighteen training batches had repeated
global IDs, which remained distinct occurrences. The checkpoint is preserved
outside Git, with [export hashes](../reports/colab/20261024T073010Z-1f9800/PROVENANCE.json).

| Same 8,192-query validation cohort, 229 positives | ROC-AUC | AP | BCE |
| --- | ---: | ---: | ---: |
| Trained GNN | 0.83194 | 0.15439 | 0.11392 |
| Historical-activity baseline | 0.85721 | 0.27248 | 0.09880 |

The activity baseline scored better on all three metrics. Different cohorts
prevent interpreting the earlier pilot's scores as a controlled quality trend.
One trained validation batch again passed exact three-mode logits/loss parity;
trained inference speed remains unmeasured.

Training-call wall time was 214.262 seconds, with 185.104 seconds of timed
work. Graph construction took 56.046 seconds separately. Training allocator
peaks were 53.742 MiB allocated and 58 MiB reserved; process-lifetime host
peak RSS reached 2.481 GiB. Memory headroom does not establish GPU compute
utilization, and allocator figures are not total device-memory use.

The export recorded 198.63 to 198.59 units remaining at 1.07 units/hour.
Isolation was not confirmed, so the 0.04-unit account interval remains
unattributed. The latest balance leaves 148.59 units above the reserve.
See the [full training report](../reports/TRAINING_RUN_STATUS.md),
[budget summary](../reports/budget/20261024-notebook05-observations.json), and
[next trained-inference requirements](NEXT_RUN.md).

## Prepared trained inference comparison

[Notebook 06](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/06_trained_inference.ipynb)
is prepared with the notebook 05 checkpoint, complete source-validation replay,
and paired reference/HGAM/block timing at query batch sizes 32, 128 and 512.
It saves raw measurement sidecars for verified resume and records GPU telemetry
when available. No hosted trained inference timings have been measured yet.
See the [fixed measurement protocol](TRAINED_INFERENCE.md).
