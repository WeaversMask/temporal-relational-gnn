"""Explicit official label loading and temporal cohorts for a bounded training pilot."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from trgnn.real_data import load_specs, normalized_time


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_ids(series: pd.Series) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise").astype("Int64")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Query entity IDs must be non-missing non-negative int64 values") from exc
    if values.isna().any() or (values < 0).any():
        raise ValueError("Query entity IDs must be non-missing non-negative int64 values")
    return values.astype("int64")


def _summary(frame: pd.DataFrame, task: dict) -> dict:
    times = frame[task["time_col"]]
    labels = frame[task["target_col"]]
    return {
        "rows": len(frame),
        "unique_entities": int(frame[task["entity_col"]].nunique()),
        "unique_timestamps": int(times.nunique()),
        "timestamp_min": str(times.min()),
        "timestamp_max": str(times.max()),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
    }


def select_training_queries(
    root: Path, train_count: int, val_count: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Require official files and sample label-independent occurrence rows.

    This never calls RelBench's fallback SQL label generator. The caller must
    verify the pinned source receipt, as with the other local projection APIs.
    """
    database, task = load_specs(root)
    val_time = pd.Timestamp(database["val_timestamp"])
    test_time = pd.Timestamp(database["test_timestamp"])
    horizon = pd.Timedelta(task["timedelta"])
    if val_time + horizon > test_time:
        raise ValueError("Forecast horizon overlaps the official validation/test boundary")
    if min(train_count, val_count) < 1:
        raise ValueError("Training and validation cohort counts must be positive")
    selected, audit = {}, {}
    generators = [np.random.default_rng(part) for part in np.random.SeedSequence(seed).spawn(2)]
    for split, count, generator in zip(
        ("train", "val"), (train_count, val_count), generators, strict=True
    ):
        path = root / "tasks/user-badge" / f"{split}.parquet"
        if not path.is_file():
            raise ValueError(
                f"Missing official {split}.parquet; download the pinned label file explicitly. "
                "SQL regeneration is disabled."
            )
        columns = [task["entity_col"], task["time_col"], task["target_col"]]
        frame = pd.read_parquet(path, columns=columns)
        if len(frame) < count:
            raise ValueError(
                f"Requested {count} {split} queries, but the official file has {len(frame)}"
            )
        frame[task["time_col"]] = normalized_time(frame[task["time_col"]])
        frame[task["entity_col"]] = _integer_ids(frame[task["entity_col"]])
        if not frame[task["target_col"]].isin([0, 1]).all():
            raise ValueError(f"{split}: official forecasting labels must be binary and non-missing")
        frame[task["target_col"]] = frame[task["target_col"]].astype("int64")
        times = frame[task["time_col"]]
        if split == "train":
            valid = times + horizon <= val_time
        else:
            valid = (times >= val_time) & (times < test_time) & (times + horizon <= test_time)
        if not valid.all():
            raise ValueError(f"{split}: query/label horizon violates the official temporal split")
        indices = generator.choice(len(frame), count, replace=False)
        selected[split] = frame.iloc[indices].copy().reset_index(drop=True)
        audit[split] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "selected_row_ids": indices.tolist(),
            "source": _summary(frame, task),
            "selected": _summary(selected[split], task),
        }
    return (
        selected["train"],
        selected["val"],
        {
            "status": "passed",
            "scope": "official_train_validation_query_cohorts",
            "selection": "uniform occurrence rows without replacement; independent of labels",
            "class_balancing": False,
            "seed": seed,
            "forecast_horizon_days": float(horizon / pd.Timedelta(days=1)),
            "val_timestamp": str(val_time),
            "test_timestamp": str(test_time),
            "splits": audit,
        },
    )


def validate_query_entities(database, queries: pd.DataFrame, task_spec: dict) -> dict:
    """Reject unknown users and users created after their own query occurrence."""
    entity = task_spec["entity_table"]
    table = database.table_dict[entity]
    ids = _integer_ids(queries[task_spec["entity_col"]]).to_numpy()
    if (ids >= len(table)).any():
        raise ValueError("A query entity is unavailable in the projected graph")
    if table.time_col is None:
        raise ValueError("Query entities require an explicit creation timestamp")
    created = normalized_time(table.df[table.time_col]).iloc[ids].reset_index(drop=True)
    times = normalized_time(queries[task_spec["time_col"]]).reset_index(drop=True)
    if not (created <= times).all():
        raise ValueError("A query entity did not exist at its prediction timestamp")
    return {"status": "passed", "query_count": len(queries), "entity_table": entity}
