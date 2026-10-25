# Public release and preserved evidence

This release cleanup adds Apache-2.0 licensing for original project contributions,
upstream software/data notices, and anonymous public setup for all eight Colab
notebooks. Repository visibility is an owner-controlled GitHub setting; this
cleanup does not change it.

## Licensing and attribution

[LICENSE](../LICENSE) and [NOTICE](../NOTICE) cover original project contributions
under Apache-2.0. Imported dependencies, adapted upstream material and copied
dataset records retain their applicable terms. The complete provenance and
scope are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md); unmodified
license texts are in [licenses/](../licenses/).

The `rel-stack/user-badge` experiment uses the pinned RelBench conversion of
Stack Exchange material. Copied task/conversion material covered by
CDLA-Permissive-2.0 is accompanied by that agreement's text. Applicable original
Stack Exchange licenses remain separate. The project's Apache license does not
replace those data terms or declare all analytical results to be ShareAlike.

The associated notice [execution.py.license](../src/trgnn/execution.py.license)
identifies the RelBench reference harness and project modifications. It is
packaged with the source file, allowing its measured executable bytes and
historical checkpoint hashes to remain unchanged. Keep that companion notice
when redistributing the associated file separately.

## What remains in the public evidence

The experiment records remain byte-for-byte intact, including compressed
archive members and provenance manifests. They include:

- Selected official benchmark row indices, a small set of entity identifiers
  and query timestamps, and copied validation labels.
- Per-example predictions/probabilities, fitted activity-baseline parameters,
  aggregate predictive scores, and raw paired timing observations.
- Recorded compute-unit balances/rates and runtime windows, with incomplete or
  overlapping account intervals distinguished from isolated run costs.
- GPU UUIDs, experiment/session IDs, dependency versions and temporary Colab
  paths. These are technical provenance, not authentication credentials.
- Git commit authorship under Martin Rodriguez and the GitHub noreply address.

Full database Parquet files and trained GNN checkpoint binaries are excluded
from Git, as are runtime caches and credentials. The small fitted baseline
parameters in JSON reports are analytical outputs, not the GNN checkpoint.
The repository is not anonymized: public visibility also exposes its earlier
commits and retained metadata. No history rewrite or selective alteration of
original observations was performed for this cleanup.

## Colab setup and reproduction boundaries

Public setup defaults to `USE_GITHUB_TOKEN = False`; it does not read Colab
Secrets or require a token. For an intentionally private copy only, set the
flag to `True` and grant access to a read-only repository-scoped `GITHUB_TOKEN`.
The optional authentication helper is temporary, keeps the token out of clone
URLs, and removes authentication data from its environment after setup.
Before the owner makes this repository public, anonymous cloning will fail as
expected; the explicit private option remains available.

Changes to notebooks are limited to introductory instructions and clone/setup
cells. Measurement commands, source-revision pinning, model semantics, frozen
configurations, original measurement implementation and archived reports retain
their earlier contents. Historical source commits and notebook versions remain
available for exact provenance; edited setup cells are not represented as the
notebook bytes used by the older exports.

CPU correctness tests and synthetic benchmarks do not need the original
trained checkpoint or full database. Exact replay of the published trained
measurements requires the original notebook 05 `training.pt`, whose SHA-256 is
`92d02f7d2007c0ab560629e9c4e0d48886a1d149930e68e494a861b85659e1d6`.
That binary is not distributed in this repository. A newly trained model is a
separate experiment and cannot be substituted into the fixed report/checkpoint
contract. This release does not claim a new GPU measurement or broader result.
