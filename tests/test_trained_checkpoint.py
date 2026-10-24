"""Frozen inference rejects changed evidence before any performance measurement."""

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from relbench.modeling.graph import make_pkey_fkey_graph
from torch_geometric.typing import WITH_PYG_LIB

import trgnn.trained_checkpoint as frozen
from trgnn.artifacts import environment, write_json
from trgnn.real_batch import RealBatchModel
from trgnn.real_data import manifest_hashes, projected_database
from trgnn.training import (
    CHECKPOINT_VERSION,
    QUERY_SCHEME,
    SEED_SCHEME,
    _architecture,
    _input_signature,
    _optimizer_groups,
)
from trgnn.training_data import select_training_queries, validate_query_entities
from trgnn.training_pilot import file_sha256, fixed_constant_statistics, make_query_loader


def input_files(tmp_path):
    root = tmp_path / "rel-stack"
    (root / "db").mkdir(parents=True)
    task_dir = root / "tasks/user-badge"
    task_dir.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    pins = json.loads((project / "configs/upstream.json").read_text())
    (root / ".trgnn-source.json").write_text(json.dumps(pins["dataset"]))
    database = {
        "name": "rel-stack",
        "manifest_version": 1,
        "val_timestamp": "2020-10-01",
        "test_timestamp": "2021-01-01",
        "tables": {
            "users": {"pkey": "Id", "time_col": "Created", "fkeys": {}},
            "events": {"pkey": "Id", "time_col": "Created", "fkeys": {"UserId": "users"}},
        },
    }
    task = {
        "name": "user-badge",
        "manifest_version": 1,
        "kind": "forecast",
        "task_type": "binary_classification",
        "entity_table": "users",
        "entity_col": "UserId",
        "time_col": "timestamp",
        "target_col": "Target",
        "timedelta": "91 days",
        "sql": "SELECT t.timestamp,u.Id AS UserId,0 AS Target FROM timestamps t CROSS JOIN users u",
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(database))
    (task_dir / "manifest.yaml").write_text(yaml.safe_dump(task))
    pd.DataFrame({"Id": [0, 1], "Created": pd.to_datetime(["2020-01-01"] * 2)}).to_parquet(
        root / "db/users.parquet", index=False
    )
    pd.DataFrame(
        {
            "Id": [0, 1, 2],
            "UserId": [0, 1, 1],
            "Created": pd.to_datetime(["2020-02-01", "2020-02-02", "2020-08-01"]),
        }
    ).to_parquet(root / "db/events.parquet", index=False)
    for split, ids, dates, labels in (
        ("train", [0, 1, 0, 1], ["2020-04-01"] * 2 + ["2020-07-01"] * 2, [0, 1, 0, 1]),
        ("val", [0, 1], ["2020-10-01"] * 2, [0, 1]),
    ):
        pd.DataFrame(
            {"UserId": ids, "timestamp": pd.to_datetime(dates), "Target": labels}
        ).to_parquet(task_dir / f"{split}.parquet", index=False)
    return root, pins, task


@pytest.fixture
def frozen_source(tmp_path):
    """Small manually assembled evidence fixture; no claim of measured training."""
    torch.set_num_threads(1)
    root, pins, task_spec = input_files(tmp_path)
    project = Path(__file__).resolve().parents[1]
    train, val, split_audit = select_training_queries(root, 4, 2, 42)
    cutoff = val.timestamp.max()
    database, tables = projected_database(root, cutoff)
    eligibility = {
        "train": validate_query_entities(database, train, task_spec),
        "val": validate_query_entities(database, val, task_spec),
    }
    graph, _ = make_pkey_fkey_graph(database, {name: {} for name in database.table_dict})
    task = frozen.load_dataset(root).load_task("user-badge")
    loader = make_query_loader(graph, train, task, [1, 1], 2)
    model = RealBatchModel(graph, fixed_constant_statistics(graph), 8, 2)
    settings = {
        "device": "cpu",
        "batch_size": 2,
        "train_queries": 4,
        "val_queries": 2,
        "channels": 8,
        "fanouts": [1, 1],
        "learning_rate": 0.001,
        "seed": 42,
    }
    runtime = environment()
    contract = {
        "format_version": 1,
        "dataset": pins["dataset"],
        "settings": settings,
        "manifest_sha256": manifest_hashes(root),
        "database_sha256": {
            str(path.relative_to(root)): file_sha256(path)
            for path in (root / "db").glob("*.parquet")
        },
        "split_audit": split_audit,
        "feature_policy_sha256": file_sha256(project / "configs/feature_policy.json"),
        "code_sha256": {
            name: file_sha256(project / "src/trgnn" / name) for name in frozen.SOURCE_FILES
        },
        "runtime_versions": {name: runtime["packages"].get(name) for name in frozen.PACKAGE_NAMES},
        "graph_cutoff": str(cutoff),
        "training_execution_mode": "reference",
    }
    order = torch.randperm(4, generator=torch.Generator().manual_seed(42))
    history = [
        {
            "step": 1,
            "sampler_seed": 42,
            "query_indices": order[:2].tolist(),
            "query_count": 2,
            "gradient_finite": True,
            "loss": 0.5,
            "visibility_audit": {"status": "passed"},
        }
    ]
    checkpoint = {
        "format_version": CHECKPOINT_VERSION,
        "contract": contract,
        "completed_steps": 1,
        "query_order": order,
        "history": history,
        "model": model.state_dict(),
        "settings": {
            "seed": 42,
            "batch_size": 2,
            "query_count": 4,
            "entity": "users",
            "input_sha256": _input_signature(loader),
            "architecture": _architecture(model),
            "optimizer": "torch.optim.adam.Adam",
            "optimizer_groups": _optimizer_groups(torch.optim.Adam(model.parameters(), lr=0.001)),
            "seed_scheme": SEED_SCHEME,
            "query_scheme": QUERY_SCHEME,
        },
    }
    source = {
        "status": "passed",
        "trained": True,
        "environment": runtime,
        "config": {**settings, "steps": 1},
        "input_contract": contract,
        "input_contract_sha256": frozen._canonical_sha256(contract),
        "training": {"status": "passed", "completed_steps": 1, "history": history},
        "checkpoint": {"completed_steps": 1},
        "split_audit": split_audit,
        "graph_visibility_cutoff": str(cutoff),
        "table_audit": tables,
        "query_eligibility": eligibility,
        "validation_after": {
            "metrics": {"count": 2, "roc_auc": 1.0, "limitations": [], "average_precision": 1.0},
            "labels": [0, 1],
            "logits": [-1.0, 1.0],
            "predictions": torch.tensor([-1.0, 1.0], dtype=torch.float64).sigmoid().tolist(),
            "sampler_sha256": ["known-ordered-occurrence-fingerprint"],
            "sampled_counts": [{"nodes": {"users": [2, 0, 0]}}],
            "visibility_audits": [{"status": "passed"}],
        },
    }
    checkpoint_path, report_path = tmp_path / "frozen.pt", tmp_path / "source.json"

    def save():
        torch.save(checkpoint, checkpoint_path)
        source["checkpoint"]["sha256"] = file_sha256(checkpoint_path)
        source["input_contract_sha256"] = frozen._canonical_sha256(source["input_contract"])
        write_json(report_path, source)

    save()
    return root, checkpoint_path, report_path, checkpoint, source, save


def test_reconstructs_frozen_weights_without_training_or_reading_test(frozen_source, monkeypatch):
    root, path, source_path, checkpoint, source, _ = frozen_source
    before = path.read_bytes()
    reads, original = [], pd.read_parquet

    def read(path, *args, **kwargs):
        reads.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    loaded = frozen.load_trained_workload(path, source_path, root)
    assert not loaded.model.training
    for name, value in loaded.model.state_dict().items():
        assert torch.equal(value, checkpoint["model"][name])
    assert len(loaded.val_queries) == 2
    assert loaded.contract == source["input_contract"]
    assert loaded.checkpoint_metadata["completed_steps"] == 1
    assert loaded.audits["baseline_refitted"] is False
    assert loaded.audits["test_queries_read"] is False
    assert "test.parquet" not in reads
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["status", "trained"])
def test_rejects_incomplete_source_reports(frozen_source, field):
    root, path, source_path, _, source, save = frozen_source
    source[field] = "failed" if field == "status" else False
    save()
    with pytest.raises(ValueError, match="passed trained source report"):
        frozen.load_trained_workload(path, source_path, root)


def test_rejects_checkpoint_bytes_from_another_run(frozen_source):
    root, path, source_path, _, _, _ = frozen_source
    path.write_bytes(path.read_bytes() + b"changed checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        frozen.load_trained_workload(path, source_path, root)


@pytest.mark.parametrize(
    "change", ["code", "runtime", "query_order", "history", "nonfinite", "architecture"]
)
def test_rejects_changed_checkpoint_contract_or_model(frozen_source, change):
    root, path, source_path, checkpoint, _, save = frozen_source
    if change == "code":
        checkpoint["contract"]["code_sha256"]["execution.py"] = "changed"
        match = "source file execution.py"
    elif change == "runtime":
        checkpoint["contract"]["runtime_versions"]["torch"] = "changed"
        match = "runtime package torch"
    elif change == "query_order":
        checkpoint["query_order"] = checkpoint["query_order"].roll(1)
        match = "query order"
    elif change == "history":
        checkpoint["history"][0]["sampler_seed"] += 1
        match = "history sampler seed"
    elif change == "nonfinite":
        checkpoint["model"]["head.bias"][0] = float("nan")
        match = "Non-finite"
    else:
        checkpoint["settings"]["architecture"]["module_sha256"] = "different-model"
        match = "architecture"
    save()
    with pytest.raises(ValueError, match=match):
        frozen.load_trained_workload(path, source_path, root)


@pytest.mark.parametrize("change", ["database", "labels", "manifest"])
def test_rejects_changed_dataset_or_official_labels(frozen_source, change):
    root, path, source_path, _, _, _ = frozen_source
    if change == "manifest":
        target = root / "manifest.yaml"
        target.write_text(target.read_text() + "\n")
        match = "manifest bytes"
    else:
        target = root / (
            "db/events.parquet" if change == "database" else "tasks/user-badge/val.parquet"
        )
        frame = pd.read_parquet(target)
        if change == "database":
            frame.loc[0, "Created"] = pd.Timestamp("2020-02-03")
        else:
            frame["Target"] = 1 - frame["Target"]
        frame.to_parquet(target, index=False)
        match = "database bytes" if change == "database" else "official split bytes"
    with pytest.raises(ValueError, match=match):
        frozen.load_trained_workload(path, source_path, root)


@pytest.mark.parametrize("change", [None, "labels", "fingerprint", "logits", "metrics"])
def test_full_validation_replay_is_required_to_match_saved_evidence(
    frozen_source, monkeypatch, change
):
    root, path, source_path, _, source, _ = frozen_source
    workload = frozen.load_trained_workload(path, source_path, root)
    replayed = copy.deepcopy(source["validation_after"])
    if change == "labels":
        replayed["labels"].reverse()
    elif change == "fingerprint":
        replayed["sampler_sha256"] = ["another-sampled-occurrence-set"]
    elif change == "logits":
        replayed["logits"][0] += 0.1
    elif change == "metrics":
        replayed["metrics"]["roc_auc"] = 0.5
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        return replayed

    monkeypatch.setattr(frozen, "evaluate", evaluate)
    if change is None:
        result = frozen.replay_saved_validation(workload)
        assert result["status"] == "passed"
        assert result["validated_query_count"] == 2
        assert result["max_absolute_logit_error"] == 0
        assert result["new_training_updates"] == 0
    else:
        with pytest.raises(ValueError, match="validation"):
            frozen.replay_saved_validation(workload)
    assert calls[0]["batch_size"] == 2
    assert calls[0]["seed"] == 1_000_042
    assert not workload.model.training


@pytest.mark.skipif(not WITH_PYG_LIB, reason="Requires the compiled temporal sampler")
def test_actual_training_checkpoint_reconstructs_and_replays_all_queries(tmp_path, monkeypatch):
    from trgnn.training_pilot import main

    root, _, _ = input_files(tmp_path)
    checkpoint, source_path = tmp_path / "trained.pt", tmp_path / "trained.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training_pilot",
            "--data-dir",
            str(root),
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--train-queries",
            "4",
            "--val-queries",
            "2",
            "--channels",
            "8",
            "--fanouts",
            "1",
            "1",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(source_path),
        ],
    )
    main()
    workload = frozen.load_trained_workload(checkpoint, source_path, root)
    replay = frozen.replay_saved_validation(workload)
    assert replay["status"] == "passed"
    assert replay["validated_query_count"] == 2
    assert replay["max_absolute_logit_error"] == 0
