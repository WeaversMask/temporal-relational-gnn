# Colab dependency feasibility: verified 2026-10-23

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

## Next: an actual relational batch

Notebook 02 downloads only the pinned `rel-stack` database, the schema/task
manifests, and validation labels. It projects keys and record timestamps before
loading data frames, clips records to the largest selected query time, validates
the primary-key prefix, and removes dangling references after the cutoff.

The diagnostic retains all tables and relationships but uses constant features
plus relative-time encoding and random weights. It compares actual RelBench
GraphSAGE against HGAM/blocks, including the same feature/time encoders and linear
readout. It checks eligible targets, time visibility, disjoint edge ownership,
the sampled BFS prefix contract, logits, and loss.

Timing covers the forward pass on identical already sampled/transferred batches.
Neighbor sampling and gathering are recorded separately, and this is explicitly
not a complete inference throughput result. Model training, useful predictive
quality, broader feature choices, and the final evaluation are still pending.
