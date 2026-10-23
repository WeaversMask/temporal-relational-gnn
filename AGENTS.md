# Project working instructions

- Goal: finish one rigorous temporal relational GNN inference investigation in
  14-21 days. Begin with the feasibility milestone in `docs/PLAN.md`.
- Preserve the reference model's weights, sampling, and feature semantics across
  execution modes. Empty relationships can contribute root transforms/biases.
- Treat local node IDs as per-query occurrences. Never deduplicate temporal
  batches using global IDs alone.
- Attribute HGAM, GraphSAGE, and established block execution. Do not claim them
  as inventions of this project.
- Distinguish synthetic/model-core, real-model, and end-to-end results in code,
  filenames, and reports. Never present unmeasured numbers as findings.
- Keep data, checkpoints, credentials, and runtime caches out of Git. Record
  dependency versions and raw measurements for actual experiments.
- Reserve 50 of the planned 200 Colab units. Measure the actual unit/hour rate.
  Use CPU development where possible and avoid idle GPU sessions.
- Run `python -m pytest -q`, `python -m ruff check .`, and
  `python -m ruff format --check .` after relevant code changes. CUDA results
  require separate validation on an NVIDIA GPU; a skipped GPU test is not proof.
- Use `131781531+WeaversMask@users.noreply.github.com` for both commit author and
  committer. Verify both identities before every push.
