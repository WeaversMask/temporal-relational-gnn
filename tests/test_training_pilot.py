import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.typing import WITH_PYG_LIB

from trgnn.artifacts import write_json
from trgnn.training_pilot import (
    TrainingProgressReporter,
    archive_previous_report,
    file_sha256,
    fixed_constant_statistics,
    main,
    parse_args,
)


def test_periodic_reporting_keeps_every_update_and_tracks_independent_checkpoints(tmp_path, capsys):
    checkpoint, output = tmp_path / "model.pt", tmp_path / "report.json"
    report = {
        "trained": False,
        "training": {"completed_steps": 0, "history": []},
        "checkpoint": {"path": str(checkpoint)},
    }
    checkpoint.write_bytes(b"initial checkpoint")
    write_json(output, report)
    progress = TrainingProgressReporter(
        report,
        output=output,
        checkpoint=checkpoint,
        steps=7,
        checkpoint_every=2,
        report_every=3,
    )
    for step in range(1, 8):
        if step % 2 == 0 or step == 7:
            checkpoint.write_bytes(f"checkpoint at {step}".encode())
        progress.record_step({"step": step, "loss": 1.0 / step})
        assert report["training"]["completed_steps"] == step
        persisted = json.loads(output.read_text())
        expected = step if step == 7 else step // 3 * 3
        assert persisted["training"]["completed_steps"] == expected
        assert [row["step"] for row in persisted["training"]["history"]] == list(
            range(1, expected + 1)
        )
        saved_step = step if step == 7 else step // 2 * 2
        assert report["checkpoint"]["completed_steps"] == saved_step
        assert report["checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert report["trained"] is True
    assert capsys.readouterr().out.splitlines() == [
        "Step 3/7: loss=0.333333",
        "Step 6/7: loss=0.166667",
        "Step 7/7: loss=0.142857",
    ]


def test_restored_history_is_durable_before_any_new_reporting_interval(tmp_path):
    checkpoint, output = tmp_path / "model.pt", tmp_path / "report.json"
    checkpoint.write_bytes(b"restored checkpoint")
    report = {
        "trained": False,
        "training": {"completed_steps": 0, "history": []},
        "checkpoint": {},
    }
    progress = TrainingProgressReporter(
        report,
        output=output,
        checkpoint=checkpoint,
        steps=2000,
        checkpoint_every=100,
        report_every=100,
    )
    progress.record_restore(
        {"completed_steps": 2, "history": [{"step": 1}, {"step": 2}], "resumed": True}
    )
    persisted = json.loads(output.read_text())
    assert persisted["trained"] is True
    assert persisted["training"]["completed_steps"] == 2
    assert persisted["training"]["resumed"] is True
    assert persisted["checkpoint"]["sha256"] == file_sha256(checkpoint)
    progress.record_step({"step": 3, "loss": 0.5})
    assert report["training"]["completed_steps"] == 3
    assert json.loads(output.read_text())["training"]["completed_steps"] == 2


def test_resume_archives_previous_report_verbatim_without_reusing_archive_paths(tmp_path):
    output = tmp_path / "training.json"
    original = b'{"validation_before": {"metrics": {"count": 8192}}}\n'
    output.write_bytes(original)
    first = archive_previous_report(output)
    second = archive_previous_report(output)
    assert first["path"] != second["path"]
    for archived in (first, second):
        path = Path(archived["path"])
        assert path.parent == output.parent
        assert path.suffix == ".json"
        assert path.read_bytes() == original
        assert archived["sha256"] == file_sha256(output)
    output.write_text('{"status": "started"}')
    assert Path(first["path"]).read_bytes() == original


def test_report_interval_defaults_to_each_update_and_must_be_positive(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["training_pilot"])
    assert parse_args().report_every == 1
    monkeypatch.setattr(sys, "argv", ["training_pilot", "--report-every", "0"])
    with pytest.raises(SystemExit):
        parse_args()


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
@pytest.mark.parametrize("interruption", [RuntimeError, KeyboardInterrupt])
def test_complete_pilot_cli_and_resume_on_local_relational_data(
    tmp_path, monkeypatch, interruption
):
    root = tmp_path / "rel-stack"
    (root / "db").mkdir(parents=True)
    (root / "tasks/user-badge").mkdir(parents=True)
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
        "--report-every",
        "1",
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
    original_report_bytes = output.read_bytes()
    command[command.index("--steps") + 1] = "2"
    command[command.index("--report-every") + 1] = "100"
    monkeypatch.setattr(sys, "argv", [*command, "--resume"])
    main()
    resumed = json.loads(output.read_text())
    assert resumed["status"] == "passed"
    assert resumed["training"]["completed_steps"] == 2
    assert resumed["training"]["resumed"] is True
    assert resumed["checkpoint"]["resume_source_sha256"] == report["checkpoint"]["sha256"]
    assert resumed["validation_before"]["status"] == "not_measured_on_resume"
    archived = resumed["archived_previous_report"]
    assert Path(archived["path"]).read_bytes() == original_report_bytes
    assert archived["sha256"] == file_sha256(Path(archived["path"]))
    assert resumed["config"]["report_every"] == 100
    assert "report_every" not in resumed["input_contract"]["settings"]
    assert torch.isfinite(torch.tensor(resumed["validation_after"]["logits"])).all()

    import trgnn.training_pilot as pilot_module

    original_train = pilot_module.train_pilot

    def fail_after_resumed_update(*args, **kwargs):
        record = kwargs["on_step"]

        def record_then_fail(row):
            record(row)
            raise interruption("simulated interruption after resumed update")

        kwargs["on_step"] = record_then_fail
        return original_train(*args, **kwargs)

    monkeypatch.setattr(pilot_module, "train_pilot", fail_after_resumed_update)
    command[command.index("--steps") + 1] = "4"
    monkeypatch.setattr(sys, "argv", [*command, "--resume"])
    resumed_report_bytes = output.read_bytes()
    with pytest.raises(interruption, match="simulated interruption"):
        main()
    failed = json.loads(output.read_text())
    assert failed["status"] == "failed"
    assert failed["trained"] is True
    assert failed["training"]["completed_steps"] == 3
    assert [row["step"] for row in failed["training"]["history"]] == [1, 2, 3]
    assert failed["checkpoint"]["completed_steps"] == 2
    assert torch.load(checkpoint, map_location="cpu", weights_only=True)["completed_steps"] == 2
    assert Path(failed["archived_previous_report"]["path"]).read_bytes() == resumed_report_bytes
