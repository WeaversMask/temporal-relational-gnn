# Historical feature availability policy

The current approved inputs remain one constant feature per table and relative
record time. No raw non-key attribute is approved for training. The audit is
partial: it checks the feature policy and local file metadata, not historical
reconstruction or training readiness.

The hosted notebook 03 metadata audit passed in run
`20261023T225501Z-efdf72`, with no errors, no fitted preprocessing and no
approved raw attributes. Its embedded report is preserved in
[`pipeline-profile.json`](../reports/colab/20261023T225501Z-efdf72/pipeline-profile.json).
The same run passed sampled-batch visibility checks. `training_ready` remains
`false`; the complete historical feature/relationship milestone stays pending.
See [the run assessment](../reports/PIPELINE_PROFILE_STATUS.md).

For deliberate reproduction after downloading the pinned files, run from the
project folder:

```sh
python -m trgnn.feature_audit --data-dir data/rel-stack --output results/feature-audit.json
```

The command verifies the pinned-data receipt and task, enumerates every table's
Arrow schema, and inspects timestamp null counts in Parquet footers. It does not
load feature values, download data, fit preprocessing, or change the model. The
report records the input/policy manifest hashes, dependency environment, selected
inputs, excluded fields, pending candidates, errors, and remaining limitations.
A `passed` status only covers metadata checks for the current constant and
relative-time policy; `training_ready` remains `false`.

Primary and foreign keys remain connectivity only. Record time supplies the
visibility cutoff and relative-time encoder. The actual sampled-batch visibility
and disjoint-occurrence checks still run in the real-batch diagnostic. Unknown
fields are excluded by default. Missing required key/time columns and known
missing timestamps fail the audit. If footer timestamp null statistics are
unavailable, the report records that limitation and full value validation in the
data loader remains necessary.

The pinned hosted schemas were inspected through bounded Parquet footer reads on
2026-10-24. This inspection establishes names and types, not historical values.
The policy in `configs/feature_policy.json` classifies those fields as follows:

| Table | Excluded attributes | Candidate attributes, still excluded |
| --- | --- | --- |
| users | AccountId, DisplayName, Location, WebsiteUrl, AboutMe | None |
| badges | None beyond keys/time | Class, Name, TagBased |
| votes | None beyond keys/time | VoteTypeId |
| comments | ContentLicense, UserDisplayName, Text | None |
| posts | OwnerDisplayName, Title, Tags, ContentLicense, Body | PostTypeId |
| postLinks | None beyond keys/time | LinkTypeId |
| postHistory | UserDisplayName, ContentLicense, RevisionGUID, Text, Comment | PostHistoryTypeId |

The policy also explicitly excludes common mutable snapshot additions, including
user reputation and post scores/counts. Their inclusion in the defensive policy
does not claim that they exist in the current pinned files. Targets are always
excluded from node features and remain supervision only.

Event categories are candidates because an award, vote, link, or history event
may preserve its category at event time. Approving one requires evidence about
the export's update/deletion behavior and each field's meaning; a timestamp and
categorical type alone are insufficient. Post types may change through conversion,
and badge names may change, so neither is presumed immutable. No candidate is
used automatically.

Before a trained experiment, establish feature-specific historical availability,
resolve the export's handling of relationship changes, merges and deletions,
choose a useful safe feature set, and fit any preprocessing on the official
training period only. Record-time filtering cannot reconstruct deleted records
or earlier foreign-key values. This audit supports the existing untrained
diagnostic and leaves those larger temporal semantics questions explicit.
