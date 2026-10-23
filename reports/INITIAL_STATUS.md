# Initial feasibility status

Recorded 2026-10-23. This is the first synthetic CPU milestone.

- Reference, HGAM, and block schedules are implemented using the same PyG layers.
- Seed-output equivalence, seed-only loss, gradients, and one optimizer step pass.
- Direct parity against the published RelBench 3.0.2 GraphSAGE model passes.
- The complete local suite has 17 passing tests and one skipped CUDA test.
- Tests cover empty relationships, isolated seeds, same-table edges, repeated
  temporal occurrences, zero fanout, and up to three message-passing layers.
- CUDA checks must run in Colab; local tests explicitly skip CUDA when unavailable.
- The pinned public database/task manifests were fetched successfully without
  downloading the complete database.
- The Colab notebook has been syntax checked, but has not been executed inside
  an authenticated Colab session.

`initial-synthetic-cpu.json` contains the raw timing repetitions and environment
from a tiny, untrained, synthetic CPU workload, generated before the initial Git
commit. These measurements are not end-to-end, GPU, or RelBench task results.
The next milestone is CUDA/sampler preflight and parity on an actual RelBench batch.
