import hashlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from relbench.base import Database, Table

from trgnn.real_data import download_inputs
from trgnn.training_data import select_training_queries, validate_query_entities


@pytest.fixture
def training_inputs(tmp_path):
    root = tmp_path / "rel-stack"
    task_dir = root / "tasks/user-badge"
    task_dir.mkdir(parents=True)
    database = {"name": "rel-stack", "val_timestamp": "2020-10-01", "test_timestamp": "2021-01-01"}
    task = {
        "kind": "forecast",
        "task_type": "binary_classification",
        "timedelta": "91 days",
        "entity_table": "users",
        "entity_col": "UserId",
        "time_col": "timestamp",
        "target_col": "WillGetBadge",
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(database))
    (task_dir / "manifest.yaml").write_text(yaml.safe_dump(task))
    pd.DataFrame(
        {
            "UserId": [0, 0, 1, 1, 2, 2],
            "timestamp": pd.to_datetime(
                ["2020-01-02", "2020-04-02", "2020-07-02", "2019-10-03", "2020-01-02", "2020-04-02"]
            ).as_unit("us"),
            "WillGetBadge": [0, 1, 0, 1, 0, 1],
        }
    ).to_parquet(task_dir / "train.parquet", index=False)
    pd.DataFrame(
        {
            "UserId": [0, 1, 2],
            "timestamp": pd.to_datetime(["2020-10-01"] * 3).as_unit("ms"),
            "WillGetBadge": [0, 1, 0],
        }
    ).to_parquet(task_dir / "val.parquet", index=False)
    return root, task


def test_official_cohorts_preserve_temporal_occurrences_and_source_hashes(training_inputs):
    root, _ = training_inputs
    train, val, audit = select_training_queries(root, 6, 3, 42)
    assert len(train) == 6 and train["UserId"].nunique() == 3
    assert train.groupby("UserId")["timestamp"].nunique().eq(2).all()
    assert str(train["timestamp"].dtype) == str(val["timestamp"].dtype) == "datetime64[ns]"
    assert (train["timestamp"] + pd.Timedelta(days=91) <= pd.Timestamp("2020-10-01")).all()
    assert audit["class_balancing"] is False
    for split in ["train", "val"]:
        path = root / "tasks/user-badge" / f"{split}.parquet"
        assert audit["splits"][split]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_selection_is_independent_of_labels(training_inputs):
    root, _ = training_inputs
    _, _, before = select_training_queries(root, 4, 2, 42)
    for split in ["train", "val"]:
        path = root / "tasks/user-badge" / f"{split}.parquet"
        frame = pd.read_parquet(path)
        frame["WillGetBadge"] = 1 - frame["WillGetBadge"]
        frame.to_parquet(path, index=False)
    _, _, after = select_training_queries(root, 4, 2, 42)
    for split in ["train", "val"]:
        assert (
            before["splits"][split]["selected_row_ids"]
            == after["splits"][split]["selected_row_ids"]
        )


def test_missing_official_train_is_rejected_without_sql_regeneration(training_inputs):
    root, _ = training_inputs
    (root / "tasks/user-badge/train.parquet").unlink()
    with pytest.raises(
        ValueError, match="Missing official train.parquet.*SQL regeneration is disabled"
    ):
        select_training_queries(root, 2, 2, 42)


@pytest.mark.parametrize(
    ("split", "date"), [("train", "2020-07-03"), ("val", "2020-09-30"), ("val", "2020-10-03")]
)
def test_rejects_prediction_or_label_horizon_overlap(training_inputs, split, date):
    root, _ = training_inputs
    path = root / "tasks/user-badge" / f"{split}.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "timestamp"] = pd.Timestamp(date)
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="official temporal split"):
        select_training_queries(root, 2, 2, 42)


@pytest.mark.parametrize(
    ("column", "value"), [("UserId", 0.5), ("UserId", -1), ("WillGetBadge", 2)]
)
def test_invalid_ids_and_labels_are_rejected(training_inputs, column, value):
    root, _ = training_inputs
    path = root / "tasks/user-badge/train.parquet"
    frame = pd.read_parquet(path)
    if column == "UserId":
        frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        select_training_queries(root, 2, 2, 42)


def test_query_entities_checked_at_each_prediction_time(training_inputs):
    _, task = training_inputs
    users = Table(
        pd.DataFrame({"Id": [0], "Created": pd.to_datetime(["2020-01-01"])}),
        {},
        pkey_col="Id",
        time_col="Created",
    )
    database = Database({"users": users})
    queries = pd.DataFrame(
        {"UserId": [0, 0], "timestamp": pd.to_datetime(["2020-02-01", "2020-01-01"])}
    )
    assert validate_query_entities(database, queries, task)["query_count"] == 2
    queries.loc[1, "timestamp"] = pd.Timestamp("2019-12-31")
    with pytest.raises(ValueError, match="did not exist"):
        validate_query_entities(database, queries, task)
    queries.loc[1, "UserId"] = 99
    with pytest.raises(ValueError, match="unavailable"):
        validate_query_entities(database, queries, task)


def test_downloader_requests_training_only_when_explicit(training_inputs, monkeypatch):
    root, _ = training_inputs
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs)),
    )
    pins = {"dataset": {"database": "rel-stack", "repository": "fixture", "revision": "pinned"}}
    download_inputs(root, pins)
    assert "rel-stack/tasks/user-badge/train.parquet" not in calls[-1]["allow_patterns"]
    download_inputs(root, pins, splits=("train", "val"))
    assert "rel-stack/tasks/user-badge/train.parquet" in calls[-1]["allow_patterns"]
    assert "rel-stack/tasks/user-badge/val.parquet" in calls[-1]["allow_patterns"]
    with pytest.raises(ValueError, match="Only explicitly requested train/val"):
        download_inputs(root, pins, splits=("test",))
