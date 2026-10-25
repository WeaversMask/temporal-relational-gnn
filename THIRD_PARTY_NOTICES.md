# Third-party software and dataset notices

This project uses the software, methods, and dataset material identified below.
Their respective terms continue to apply. Original project contributions are
licensed under [Apache-2.0](LICENSE), with the project notice in [NOTICE](NOTICE).
This notice does not relicense third-party material or apply a dataset license
to all project code.

The inspected revisions are recorded in [configs/upstream.json](configs/upstream.json).
The license texts in [licenses/](licenses/) are unmodified upstream downloads.

## RelBench software and adapted reference harness

- Project: [RelBench](https://github.com/stanford-star/relbench), by the RelBench
  contributors.
- Inspected revision: `91c1f6420a03feb61c279aaeb194d7268435f278`; runtime package
  version: `3.0.2`.
- License at that revision: **Apache License, Version 2.0**. The complete text is
  included as [licenses/RelBench-APACHE-2.0.txt](licenses/RelBench-APACHE-2.0.txt).
- Reference implementation:
  [relbench/modeling/nn.py at the inspected revision](https://github.com/stanford-star/relbench/blob/91c1f6420a03feb61c279aaeb194d7268435f278/relbench/modeling/nn.py),
  particularly `HeteroGraphSAGE`.

The project's `RelationalGraphSAGE` harness and reference-layer execution in
[src/trgnn/execution.py](src/trgnn/execution.py) adapt the RelBench configuration
and execution structure. They are not a claim to authorship of GraphSAGE or
RelBench's reference architecture.

**Project modification notice:** the reference layout has been changed and
extended to provide a standalone harness, explicit reference/HGAM/block mode
selection, sampled-prefix validation, and destination-restricted bipartite
execution. The project also adds occurrence-aware correctness tests and
measurement tooling. These changes preserve the reference operators and their
weights across execution modes.

[src/trgnn/real_batch.py](src/trgnn/real_batch.py) integrates the imported RelBench
`HeteroEncoder`, `HeteroTemporalEncoder`, and `HeteroGraphSAGE` with the project's
readout and execution-mode adapter. Those imported implementations remain in
the installed upstream dependency. This notice does not suggest that all
independent project code is an Apache-licensed derivative merely because it
imports RelBench interfaces.

The inspected RelBench tree contains no upstream `NOTICE` file. Its complete
Apache license is retained here, and the reference provenance and project
modifications are identified above. These notices supplement the existing
source descriptions; historical source files used by experiment hashes remain
unchanged. The adjacent [execution.py.license](src/trgnn/execution.py.license)
travels with the reference harness and records its provenance and project
modifications without changing its measured executable bytes.

## PyTorch Geometric and established execution methods

- Project: [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric).
- Inspected revision: `ea2c5f62d54c8b414dd3ea7139814a7c1ef83182`; runtime package
  version: `2.8.1`.
- License: **MIT**; upstream notice: copyright (c) 2023 PyG Team
  `<team@pyg.org>`. The complete copyright, permission, and warranty text is
  included as [licenses/PyG-MIT.txt](licenses/PyG-MIT.txt).

The project imports PyG's GraphSAGE, heterogeneous convolution, normalization,
sampling, and `trim_to_layer` APIs. In particular, HGAM uses the installed
[upstream trimming implementation](https://github.com/pyg-team/pytorch_geometric/blob/ea2c5f62d54c8b414dd3ea7139814a7c1ef83182/torch_geometric/utils/_trim_to_layer.py);
that implementation is not vendored as project-owned source. The included MIT
text preserves the upstream notice and does not imply that this repository
redistributes the complete PyG package.

[HGAM](https://pytorch-geometric.readthedocs.io/en/latest/advanced/hgam.html),
GraphSAGE, and bipartite/message-flow block execution are established methods.
[DGL's block-execution documentation](https://www.dgl.ai/dgl_docs/guide/minibatch-node.html)
is an explanatory reference, not a vendored dependency or copied implementation.
This project's work concerns adaptation, correctness, and controlled measurement
within a temporal relational workload.

## RelBench dataset contributions and original Stack Exchange material

- Hosted dataset:
  [stanford-star/relbench-v1](https://huggingface.co/datasets/stanford-star/relbench-v1).
- Pinned dataset revision: `21d7284df2a9621c71f21bc4a1a30b7e8effb8dd`.
- Database/task: `rel-stack` / `user-badge`.
- Pinned dataset terms: [upstream LICENSE.md](https://huggingface.co/datasets/stanford-star/relbench-v1/blob/21d7284df2a9621c71f21bc4a1a30b7e8effb8dd/LICENSE.md),
  reproduced unchanged in
  [licenses/RelBench-DATASET-LICENSE.md](licenses/RelBench-DATASET-LICENSE.md).
- Original source identified in the pinned dataset statistics:
  [Stack Exchange data archives](https://archive.org/details/stackexchange).
  The pinned [RelBench provenance generator](https://github.com/stanford-star/relbench/blob/91c1f6420a03feb61c279aaeb194d7268435f278/provenance/stack.py)
  documents the forum-dump conversion and key reindexing.

The pinned terms distinguish the original databases from RelBench's
contributions. RelBench's conversion, schemas, task definitions, splits, and
statistics use **Community Data License Agreement — Permissive, Version 2.0**.
The complete agreement is included as
[licenses/CDLA-Permissive-2.0.txt](licenses/CDLA-Permissive-2.0.txt), from the
[canonical CDLA source](https://cdla.dev/permissive-2-0/).

The pinned `STATS/databases.parquet` entry describes `rel-stack` as
**CC BY-SA 4.0, with older posts under CC BY-SA 2.5/3.0**. These are the original
Stack Exchange contributors' applicable terms, not one blanket new license for
the entire dataset or repository. The official license texts are:

- [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/legalcode).
- [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode).
- [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en).

Stack Exchange's [licensing explanation](https://stackoverflow.com/help/licensing)
describes the contribution-date boundaries and revision-specific licensing.
Applicable attribution, source/license links, and ShareAlike conditions follow
protected original material when shared or adapted. This notice does not
declare that every factual identifier, numeric label, model, or independent
program is covered by the same copyright or database rights.

### Scope of records and results in this repository

These dataset notices apply to relevant copied material throughout
`reports/colab/` and `reports/analysis/`, including JSON members inside compressed
report archives. Reports contain more than aggregate timings: some preserve
selected official row indices, query/entity identifiers and timestamps, and
copied official validation labels. The absence of full database Parquet files
does not make those copied records independent of their source terms.

Project processing includes fixed cohort selection, temporal visibility
filtering, constant/relative-time feature projections, and occurrence-based
sampling. The reports distinguish copied official labels from project-computed
predictions, probabilities, metrics, timing observations, and fitted parameters.
The dataset revision, selection policy, and input hashes provide provenance for
the reported subsets; original Stack Exchange text is not represented as new
project authorship.

CDLA-Permissive-2.0 section 2 requires the agreement text to accompany shared
Data; that text is supplied above. Section 3 places no obligations on Results
under that agreement, including computational outcomes such as models and
insights. This distinction does not remove any applicable original-source
rights in copied material. Source-data notices do not impose ShareAlike on the
project's unrelated software or automatically relicense every analytical result.

## License-text integrity

Retrieved on **2026-10-25** from the official sources below. SHA-256 values refer
to the exact downloaded bytes, without newline or encoding normalization.

| Included file | SHA-256 |
| --- | --- |
| `licenses/RelBench-APACHE-2.0.txt` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `licenses/PyG-MIT.txt` | `89cfb6edc309735916a0c1189dccf761add4af6062ddb34b5a26f60d3efadfea` |
| `licenses/RelBench-DATASET-LICENSE.md` | `0d5ffb5bd5ac93c1e8e9109fb0844ec88219877f123c7a0126415aed69bce4ad` |
| `licenses/CDLA-Permissive-2.0.txt` | `4531a67d443284d93ffed0803df5b10634aff21c3d77e381f2d48af01d875868` |

Download sources:

- [Pinned RelBench license](https://raw.githubusercontent.com/stanford-star/relbench/91c1f6420a03feb61c279aaeb194d7268435f278/LICENSE).
- [Pinned PyG license](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/ea2c5f62d54c8b414dd3ea7139814a7c1ef83182/LICENSE).
- [Pinned dataset license](https://huggingface.co/datasets/stanford-star/relbench-v1/raw/21d7284df2a9621c71f21bc4a1a30b7e8effb8dd/LICENSE.md).
- [Canonical CDLA-Permissive-2.0 plain text](https://raw.githubusercontent.com/Community-Data-License-Agreements/Releases/main/CDLA-Permissive-2.0.txt).
