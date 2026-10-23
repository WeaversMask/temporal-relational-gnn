"""Fail-closed feature policy audit using local Parquet schemas and footers only."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trgnn.artifacts import environment, write_json
from trgnn.real_data import load_specs, manifest_hashes, verify_inputs


def _classify_column(name: str, spec: dict, task: dict, policy: dict) -> dict:
    if name == spec["pkey"]:
        role, usage = "primary_key", "connectivity_only"
        reason = "Preserve the hosted row identity; never use the key as a learned feature."
    elif name in spec.get("fkeys", {}):
        role, usage = "foreign_key", "connectivity_only"
        reason = "Preserve the relationship; historical key changes remain an audit limitation."
    elif name == spec["time_col"]:
        role, usage = "record_time", "relative_time_only"
        reason = "Used for visibility and relative time, subject to sampled-batch checks."
    elif name.casefold() == task["target_col"].casefold():
        role, usage = "excluded_task_label", "excluded"
        reason = "The forecasting target is supervision and must not become a node feature."
    elif name in policy.get("pending_event_fields", {}):
        role, usage = "pending_event_semantics", "excluded"
        reason = policy["pending_event_fields"][name]
    elif name in policy.get("excluded_snapshot_fields", []):
        role, usage = "excluded_mutable_snapshot", "excluded"
        reason = "A current snapshot or identifier does not establish its historical value."
    elif name in policy.get("excluded_text_fields", []):
        role, usage = "excluded_text_or_identifier", "excluded"
        reason = "Text/identifiers need historical reconstruction and a separate feature decision."
    else:
        role, usage = "excluded_unknown", "excluded"
        reason = "No explicit historical availability justification; excluded by default."
    return {"role": role, "permitted_usage": usage, "raw_feature_approved": False, "reason": reason}


def audit_feature_availability(root: Path, pins: dict) -> dict:
    """Return a metadata audit without loading table values, fitting, or writing files.

    A passed status covers only the current constant/relative-time input policy.
    Unknown columns are excluded rather than silently accepted. It is not a
    certificate that richer features or historical graph relationships are safe.
    """
    verify_inputs(root, pins)
    database, task = load_specs(root)
    policy_path = Path(__file__).resolve().parents[2] / "configs/feature_policy.json"
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    if (
        database.get("name") != policy["database"]
        or pins["dataset"]["database"] != policy["database"]
        or task.get("name") != policy["task"]
        or pins["dataset"]["task"] != policy["task"]
    ):
        raise ValueError("The feature policy is specific to pinned rel-stack/user-badge")
    report = {
        "status": "passed",
        "scope": "historical_feature_policy_metadata_audit",
        "training_ready": False,
        "dataset": pins["dataset"],
        "policy_version": policy["policy_version"],
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "manifest_sha256": manifest_hashes(root),
        "approved_input_features": policy["approved_input_features"],
        "raw_feature_columns": [],
        "tables": {},
        "pending_candidates": {},
        "errors": [],
        "limitations": list(policy["limitations"]),
        "preprocessing_fitted": False,
        "sampled_batch_visibility": "not_checked_here",
    }
    if policy["raw_feature_columns"]:
        raise ValueError("This audit only approves constant and relative-time inputs")
    for name, spec in database["tables"].items():
        path = root / "db" / f"{name}.parquet"
        if not path.is_file():
            report["errors"].append(f"{name}: missing local Parquet file")
            continue
        with pq.ParquetFile(path) as parquet:
            schema, metadata = parquet.schema_arrow, parquet.metadata
        table_policy = policy["tables"].get(name, {})
        fields = {}
        for field in schema:
            fields[field.name] = {
                "arrow_type": str(field.type),
                **_classify_column(field.name, spec, task, table_policy),
            }
        required = [spec["pkey"], spec["time_col"], *spec.get("fkeys", {})]
        missing = sorted(set(required) - set(schema.names))
        if missing:
            report["errors"].append(f"{name}: missing connectivity/time columns {missing}")
        for key in [spec["pkey"], *spec.get("fkeys", {})]:
            if key in schema.names and not pa.types.is_integer(schema.field(key).type):
                report["errors"].append(f"{name}.{key}: keys must have an integer Arrow type")
        time_name = spec["time_col"]
        time_audit = {"null_count": None, "statistics_complete": False}
        if time_name in schema.names:
            if not pa.types.is_timestamp(schema.field(time_name).type):
                report["errors"].append(f"{name}: record time must have a timestamp Arrow type")
            column_index = next(
                i
                for i in range(metadata.num_columns)
                if metadata.schema.column(i).name == time_name
            )
            statistics = [
                metadata.row_group(i).column(column_index).statistics
                for i in range(metadata.num_row_groups)
            ]
            complete = all(stats is not None and stats.has_null_count for stats in statistics)
            time_audit["statistics_complete"] = complete
            if complete:
                null_count = sum(stats.null_count for stats in statistics)
                time_audit["null_count"] = null_count
                if null_count:
                    report["errors"].append(f"{name}: {null_count} missing record timestamps")
            else:
                report["limitations"].append(
                    f"{name}: footer null counts unavailable; validate times in the loader."
                )
        report["tables"][name] = {
            "hosted_rows": metadata.num_rows,
            "file_size_bytes": path.stat().st_size,
            "columns": fields,
            "record_time": time_audit,
            "projected_columns": list(dict.fromkeys(required)),
        }
        candidates = [
            key for key, value in fields.items() if value["role"] == "pending_event_semantics"
        ]
        if candidates:
            report["pending_candidates"][name] = candidates
    if report["errors"]:
        report["status"] = "failed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rel-stack"))
    parser.add_argument("--output", type=Path, default=Path("results/feature-audit.json"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    pins = json.loads((project_root / "configs/upstream.json").read_text())
    try:
        report = audit_feature_availability(args.data_dir, pins)
    except Exception as exc:
        report = {
            "status": "failed",
            "scope": "historical_feature_policy_metadata_audit",
            "training_ready": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    report["environment"] = environment()
    write_json(args.output, report)
    print(f"Saved {args.output}; metadata audit={report['status']}; training_ready=False")
    if report["status"] != "passed":
        raise SystemExit("\n".join(report["errors"]))


if __name__ == "__main__":
    main()
