# Next Colab run: historical inputs and pipeline costs

Use `notebooks/03_audit_profile.ipynb` after notebook 02 passes its actual
relational-batch checks. This next run audits which fields are safe to use and
measures where inference time goes. It keeps the same untrained model with
constant node features plus relative time. It does not start training, enable
additional features, or expand the workload.

The new notebook and profiler require validation in the user's Colab GPU
runtime. Local tests and synthetic CUDA evidence do not establish that this new
hosted-data run passes. No timings for this next run have been measured yet.

## Run the notebook

1. Open [notebook 03 in Colab](https://colab.research.google.com/github/WeaversMask/temporal-relational-gnn/blob/main/notebooks/03_audit_profile.ipynb)
   after the new code is pushed. Save a copy to Drive; the runtime does not mount
   Drive. Enable this notebook's access to the existing read-only `GITHUB_TOKEN`.
2. Select a T4 GPU and connect. For a CPU diagnostic, set `DEVICE = "cpu"` in the
   first code cell. Leave the other workload settings at their defaults.
3. Read Colab's resource panel and enter the observed unit/hour rate and remaining
   units in the first code cell. Use `None` only when an observation is unavailable.
   The notebook records those values and stops before setup when a supplied
   remaining balance is 50 units or less. Keep checking the panel during the run;
   an unknown balance does not establish that spending is within budget.
4. Run the cells in order. Clone and package installation logs are captured from
   the start. The binary sampler, CUDA availability and correctness tests must
   pass before data processing.
5. The notebook reuses `/content/temporal-relational-gnn/data/rel-stack` if its
   download receipt matches the pinned dataset. Otherwise it requests a download.
   A fresh VM needs the full database again; projected columns do not reduce
   download size. No Hugging Face token is needed for the public data.
6. Inspect the status and per-mode pipeline/stage measurements. Run the final
   ZIP-download cell, keep the archive, then disconnect and delete the runtime.

Defaults remain 3 batches of 8 queries, fanouts `[4, 2]`, 64 channels, 2 layers
and 10 repetitions. Do not enlarge this run or begin training automatically.

## Read the report

`pipeline-profile.json` contains the historical feature audit, exact environment,
configuration, paired correctness checks, raw repetitions, stage measurements,
and overall inference timings. The embedded feature audit classifies fields and
keeps pending or unsafe fields excluded. A passed audit applies to the current
constant-feature and relative-time policy; it does not mean the project is ready
to train on additional historical features.

The inference boundary starts with sampling from an already built resident
historical graph, then includes feature gathering, device transfer, feature/time
encoding, GNN execution and the prediction head. It ends at logits. Download,
projection, graph construction, loader preparation and model initialization are
one-time setup and are excluded from inference totals. Their setup costs are
recorded separately.

Whole-pipeline timings come from separate uninstrumented passes. Additional
instrumented passes separate stage costs. Do not add medians of the stages and
present that sum as the measured pipeline total. Execution modes share weights
and paired sampled occurrences. These random-weight measurements can establish
correctness and locate bottlenecks, not predictive quality or performance of a
trained final predictor.

A successful run needs `status: passed`, a passed embedded `feature_audit`, and
all requested paired batches. Check raw repetitions and correctness errors
alongside the medians. Host process lifetime peak memory and total allocated or
reserved CUDA peaks have different boundaries from extra forward allocations in
notebook 02; compare each metric using its recorded definition.

## If a cell fails

Stop the experiment and run the final export cell. Storage and logging are
initialized before clone and installation, so their failures can be exported too.
Do not rerun the first initialization cell before exporting: it creates a new run
folder. Reports and logs are under
`/content/temporary-trgnn-profile/runs/<run-id>/`. The ZIP excludes the database.

Ordinary profiler exceptions save a failed report and any completed batches.
A VM deletion or hard process kill can interrupt reporting; preserve whatever
logs remain. Export before disconnecting. Review the first failed check before
retrying or changing settings. A partial report is evidence of a diagnostic
attempt, not a completed result.
