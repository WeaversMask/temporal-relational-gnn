import json

import pandas as pd
import pytest
import yaml

from trgnn.feature_audit import audit_feature_availability


@pytest.fixture
def audit_inputs(tmp_path):
    root = tmp_path / "rel-stack"
    (root / "db").mkdir(parents=True)
    (root / "tasks/user-badge").mkdir(parents=True)
    dataset = {"database": "rel-stack", "task": "user-badge", "revision": "fixture"}
    (root / ".trgnn-source.json").write_text(json.dumps(dataset))
    database = {
        "name": "rel-stack",
        "tables": {
            "users": {"pkey": "Id", "time_col": "CreationDate", "fkeys": {}},
            "badges": {"pkey": "Id", "time_col": "Date", "fkeys": {"UserId": "users"}},
        },
    }
    task = {
        "name": "user-badge",
        "kind": "forecast",
        "task_type": "binary_classification",
        "timedelta": "91 days",
        "target_col": "WillGetBadge",
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(database))
    (root / "tasks/user-badge/manifest.yaml").write_text(yaml.safe_dump(task))
    pd.DataFrame(
        {
            "Id": [0],
            "CreationDate": pd.to_datetime(["2020-01-01"]),
            "Reputation": [999],
            "DisplayName": ["current name"],
            "UnexpectedAttribute": ["unknown historical availability"],
            "WillGetBadge": [1],
        }
    ).to_parquet(root / "db/users.parquet", index=False)
    pd.DataFrame(
        {"Id": [0], "Date": pd.to_datetime(["2020-02-01"]), "UserId": [0], "Class": [1]}
    ).to_parquet(root / "db/badges.parquet", index=False)
    return root, {"dataset": dataset}


def test_mutable_unknown_labels_and_candidates_fail_closed(audit_inputs, monkeypatch):
    root, pins = audit_inputs

    def disallow_value_reads(*args, **kwargs):
        raise AssertionError("Feature audit must read schemas/footers, not data frames")

    monkeypatch.setattr(pd, "read_parquet", disallow_value_reads)
    report = audit_feature_availability(root, pins)
    assert report["status"] == "passed"
    assert report["training_ready"] is False
    assert report["raw_feature_columns"] == []
    users = report["tables"]["users"]["columns"]
    assert users["Reputation"]["role"] == "excluded_mutable_snapshot"
    assert users["UnexpectedAttribute"]["role"] == "excluded_unknown"
    assert users["WillGetBadge"]["role"] == "excluded_task_label"
    assert all(not item["raw_feature_approved"] for item in users.values())
    assert users["Id"]["permitted_usage"] == "connectivity_only"
    assert users["CreationDate"]["permitted_usage"] == "relative_time_only"
    badges = report["tables"]["badges"]["columns"]
    assert badges["Class"]["permitted_usage"] == "excluded"
    assert report["pending_candidates"] == {"badges": ["Class"]}
    assert report["tables"]["badges"]["projected_columns"] == ["Id", "Date", "UserId"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("problem", ["missing_foreign_key", "missing_record_time", "null_time"])
def test_invalid_required_schema_or_time_is_a_failed_audit(audit_inputs, problem):
    root, pins = audit_inputs
    path = root / "db/badges.parquet"
    frame = pd.read_parquet(path)
    if problem == "missing_foreign_key":
        frame = frame.drop(columns="UserId")
    elif problem == "missing_record_time":
        frame = frame.drop(columns="Date")
    else:
        frame.loc[0, "Date"] = pd.NaT
    frame.to_parquet(path, index=False)
    report = audit_feature_availability(root, pins)
    assert report["status"] == "failed"
    assert report["errors"]
    assert report["training_ready"] is False


def test_feature_audit_requires_pinned_data_receipt(audit_inputs):
    root, pins = audit_inputs
    (root / ".trgnn-source.json").unlink()
    with pytest.raises(ValueError, match="pinned-data receipt"):
        audit_feature_availability(root, pins)
