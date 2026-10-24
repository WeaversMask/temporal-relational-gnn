"""Pinned, projected relational data for the first real-batch diagnostic.

Only keys and record timestamps are loaded. All feature values are constant;
this is an untrained structure/time experiment, not the final forecasting model.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def load_specs(root: Path) -> tuple[dict, dict]:
    database = yaml.safe_load((root / "manifest.yaml").read_text())
    task = yaml.safe_load((root / "tasks/user-badge/manifest.yaml").read_text())
    if task.get("kind") != "forecast" or task.get("task_type") != "binary_classification":
        raise ValueError("Expected the pinned binary forecasting task")
    if pd.Timedelta(task["timedelta"]) != pd.Timedelta(days=91):
        raise ValueError("Unexpected forecasting horizon; inspect the task definition")
    return database, task


def download_inputs(root: Path, pins: dict, splits: tuple[str, ...] = ("val",)) -> None:
    from huggingface_hub import snapshot_download

    dataset = pins["dataset"]
    if not splits or any(split not in {"train", "val"} for split in splits):
        raise ValueError("Only explicitly requested train/val labels may be downloaded")
    if root.name != dataset["database"]:
        raise ValueError(f"Data directory must end with {dataset['database']!r}")
    prefix = dataset["database"]
    print(f"Downloading pinned database and task files for splits {splits}.", flush=True)
    snapshot_download(
        repo_id=dataset["repository"],
        repo_type="dataset",
        revision=dataset["revision"],
        allow_patterns=[
            f"{prefix}/manifest.yaml",
            f"{prefix}/db/*.parquet",
            f"{prefix}/tasks/user-badge/manifest.yaml",
            *(f"{prefix}/tasks/user-badge/{split}.parquet" for split in dict.fromkeys(splits)),
        ],
        local_dir=str(root.parent),
    )
    (root / ".trgnn-source.json").write_text(json.dumps(dataset, indent=2) + "\n")


def verify_inputs(root: Path, pins: dict) -> None:
    receipt = root / ".trgnn-source.json"
    if not receipt.exists() or json.loads(receipt.read_text()) != pins["dataset"]:
        raise ValueError("Missing or mismatched pinned-data receipt; rerun with --download")


def normalized_time(series: pd.Series) -> pd.Series:
    """Force nanoseconds across pandas versions before RelBench's time conversion."""
    result = pd.to_datetime(series, utc=True).dt.tz_convert(None).astype("datetime64[ns]")
    if result.isna().any():
        raise ValueError(
            f"Missing timestamps in {series.name}; an explicit visibility policy is needed"
        )
    return result


def projected_database(root: Path, cutoff: pd.Timestamp):
    from relbench.base import Database, Table

    database_spec, _ = load_specs(root)
    tables, audit = {}, {}
    for name, spec in database_spec["tables"].items():
        columns = list(dict.fromkeys([spec["pkey"], spec["time_col"], *spec.get("fkeys", {})]))
        frame = pd.read_parquet(root / "db" / f"{name}.parquet", columns=columns)
        frame[spec["time_col"]] = normalized_time(frame[spec["time_col"]])
        original_rows = len(frame)
        frame = frame.loc[frame[spec["time_col"]] <= cutoff].copy()
        if not np.array_equal(frame[spec["pkey"]].to_numpy(), np.arange(len(frame))):
            raise ValueError(f"{name}: cutoff must preserve the hosted primary-key prefix")
        tables[name] = Table(
            frame,
            fkey_col_to_pkey_table=spec.get("fkeys", {}),
            pkey_col=spec["pkey"],
            time_col=spec["time_col"],
        )
        audit[name] = {
            "hosted_rows": original_rows,
            "visible_rows": len(frame),
            "loaded_columns": columns,
            "dangling_foreign_keys_removed": {},
        }
        print(f"Projected {name}: {len(frame):,} visible rows", flush=True)
    for name, table in tables.items():
        for key, target in table.fkey_col_to_pkey_table.items():
            values = table.df[key].astype("Int64")
            invalid = values.notna() & ((values < 0) | (values >= len(tables[target])))
            audit[name]["dangling_foreign_keys_removed"][key] = int(invalid.sum())
            table.df[key] = values.mask(invalid)
    return Database(tables), audit


def select_queries(root: Path, count: int, seed: int):
    _, spec = load_specs(root)
    columns = [spec["entity_col"], spec["time_col"], spec["target_col"]]
    frame = pd.read_parquet(root / "tasks/user-badge/val.parquet", columns=columns)
    frame[spec["time_col"]] = normalized_time(frame[spec["time_col"]])
    if count < 1 or len(frame) < count:
        raise ValueError("The validation cohort is smaller than the requested query count")
    # Selection is independent of labels. Preserve the original row IDs in the report.
    indices = np.random.default_rng(seed).choice(len(frame), count, replace=False)
    selected = frame.iloc[indices].copy().reset_index(drop=True)
    if not selected[spec["target_col"]].isin([0, 1]).all():
        raise ValueError("Expected binary validation labels")
    return selected, indices.tolist()


def audit_sampled_batch(batch, entity: str) -> dict:
    seed_times = batch[entity].seed_time
    repeated_occurrences = {}
    for name in batch.node_types:
        owners = batch[name].batch
        if not bool((batch[name].time <= seed_times[owners]).all()):
            raise AssertionError(f"Future record sampled in {name}")
        repeated_occurrences[name] = int(
            batch[name].n_id.numel() - batch[name].n_id.unique().numel()
        )
    for (source, _, target), edges in batch.edge_index_dict.items():
        if not torch.equal(batch[source].batch[edges[0]], batch[target].batch[edges[1]]):
            raise AssertionError("An edge crosses disjoint query occurrences")
    return {"status": "passed", "repeated_global_id_occurrences": repeated_occurrences}


def manifest_hashes(root: Path) -> dict:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [root / "manifest.yaml", root / "tasks/user-badge/manifest.yaml"]
    }
