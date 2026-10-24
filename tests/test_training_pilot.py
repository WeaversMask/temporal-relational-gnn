import json
import sys

import pandas as pd
import pytest
import torch
import yaml
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.typing import WITH_PYG_LIB

from trgnn.training_pilot import fixed_constant_statistics, main


def test_pilot_uses_literal_statistics_and_rejects_other_features():
    graph = HeteroData()
    graph["users"].tf = (
        Dataset(pd.DataFrame({"__const__": [1.0, 1.0]}), {"__const__": stype.numerical})
        .materialize()
        .tensor_frame
    )
    from torch_frame.data.stats import StatType

    stats = fixed_constant_statistics(graph)
    assert stats["users"]["__const__"][StatType.MEAN] == 1
    assert stats["users"]["__const__"][StatType.STD] == 0
    graph["users"].tf.feat_dict[stype.numerical][0, 0] = 2
    with pytest.raises(ValueError, match="literal constant-one"):
        fixed_constant_statistics(graph)
    graph["users"].tf = (
        Dataset(pd.DataFrame({"Reputation": [1.0, 1.0]}), {"Reputation": stype.numerical})
        .materialize()
        .tensor_frame
    )
    with pytest.raises(ValueError, match="approved constant"):
        fixed_constant_statistics(graph)


@pytest.mark.skipif(not WITH_PYG_LIB, reason="Requires the compiled temporal sampler")
def test_complete_pilot_cli_and_resume_on_local_relational_data(tmp_path, monkeypatch):
    root = tmp_path / "rel-stack"
    (root / "db").mkdir(parents=True)
    (root / "tasks/user-badge").mkdir(parents=True)
    from pathlib import Path

    pins = json.loads((Path(__file__).resolve().parents[1] / "configs/upstream.json").read_text())
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
    (root / "tasks/user-badge/manifest.yaml").write_text(yaml.safe_dump(task))
    pd.DataFrame(
        {"Id": [0, 1], "Created": pd.to_datetime(["2020-01-01", "2020-01-02"])}
    ).to_parquet(root / "db/users.parquet", index=False)
    pd.DataFrame(
        {
            "Id": [0, 1, 2],
            "UserId": [0, 1, 1],
            "Created": pd.to_datetime(["2020-02-01", "2020-02-02", "2020-08-01"]),
        }
    ).to_parquet(root / "db/events.parquet", index=False)
    pd.DataFrame(
        {
            "UserId": [0, 1, 0, 1],
            "Target": [0, 1, 0, 1],
            "timestamp": pd.to_datetime(["2020-04-01"] * 2 + ["2020-07-01"] * 2),
        }
    ).to_parquet(root / "tasks/user-badge/train.parquet", index=False)
    pd.DataFrame(
        {
            "UserId": [0, 1],
            "Target": [0, 1],
            "timestamp": pd.to_datetime(["2020-10-01"] * 2),
        }
    ).to_parquet(root / "tasks/user-badge/val.parquet", index=False)
    checkpoint, output = tmp_path / "pilot.pt", tmp_path / "pilot.json"
    command = [
        "training_pilot",
        "--data-dir",
        str(root),
        "--steps",
        "1",
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
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", command)
    main()
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["trained"] is True
    assert report["training"]["completed_steps"] == 1
    assert report["validation_after"]["metrics"]["count"] == 2
    assert report["trained_mode_equivalence"]["status"] == "passed"
    assert checkpoint.is_file()
    command[command.index("--steps") + 1] = "2"
    monkeypatch.setattr(sys, "argv", [*command, "--resume"])
    main()
    resumed = json.loads(output.read_text())
    assert resumed["status"] == "passed"
    assert resumed["training"]["completed_steps"] == 2
    assert resumed["training"]["resumed"] is True
    assert resumed["checkpoint"]["resume_source_sha256"] == report["checkpoint"]["sha256"]
    assert resumed["validation_before"]["status"] == "not_measured_on_resume"
    assert torch.isfinite(torch.tensor(resumed["validation_after"]["logits"])).all()

    import trgnn.training_pilot as pilot_module

    original_train = pilot_module.train_pilot

    def fail_after_resumed_update(*args, **kwargs):
        record = kwargs["on_step"]

        def record_then_fail(row):
            record(row)
            raise RuntimeError("simulated interruption after resumed update")

        kwargs["on_step"] = record_then_fail
        return original_train(*args, **kwargs)

    monkeypatch.setattr(pilot_module, "train_pilot", fail_after_resumed_update)
    command[command.index("--steps") + 1] = "3"
    monkeypatch.setattr(sys, "argv", [*command, "--resume"])
    with pytest.raises(RuntimeError, match="simulated interruption"):
        main()
    failed = json.loads(output.read_text())
    assert failed["status"] == "failed"
    assert failed["trained"] is True
    assert failed["training"]["completed_steps"] == 3
    assert [row["step"] for row in failed["training"]["history"]] == [1, 2, 3]
