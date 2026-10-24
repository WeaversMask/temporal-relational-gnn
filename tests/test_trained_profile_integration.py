"""Actual local training, frozen reconstruction, paired profiling and resume."""

import json
import sys
from pathlib import Path

import pytest
from test_trained_checkpoint import input_files
from torch_geometric.typing import WITH_PYG_LIB

import trgnn.trained_profile as profiler
from trgnn.training_pilot import main as train_main


@pytest.mark.skipif(not WITH_PYG_LIB, reason="Requires the compiled temporal sampler")
def test_actual_trained_profile_and_complete_resume_preserve_source_evidence(tmp_path, monkeypatch):
    root, _, _ = input_files(tmp_path)
    checkpoint, training_report = tmp_path / "trained.pt", tmp_path / "training.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training_pilot",
            "--data-dir",
            str(root),
            "--device",
            "cpu",
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
            str(training_report),
        ],
    )
    train_main()
    checkpoint_bytes, source_bytes = checkpoint.read_bytes(), training_report.read_bytes()
    output, artifacts = tmp_path / "inference.json", tmp_path / "paired-batches"
    command = [
        "trained_profile",
        "--checkpoint",
        str(checkpoint),
        "--training-report",
        str(training_report),
        "--data-dir",
        str(root),
        "--device",
        "cpu",
        "--batch-sizes",
        "1",
        "2",
        "--repeats",
        "3",
        "--warmup",
        "0",
        "--bootstrap-samples",
        "20",
        "--output",
        str(output),
        "--artifacts-dir",
        str(artifacts),
    ]
    monkeypatch.setattr(sys, "argv", command)
    profiler.main()
    result_bytes = output.read_bytes()
    result = json.loads(result_bytes)
    assert result["status"] == "passed"
    assert result["training_performed"] is False
    assert result["validation_replay"]["status"] == "passed"
    assert result["validation_replay"]["validated_query_count"] == 2
    assert result["validation_replay"]["max_absolute_logit_error"] == 0
    assert result["completed_batches"] == result["expected_batches"] == 3
    assert result["completed_batches_this_attempt"] == 3
    assert result["model_state_sha256_before"] == result["model_state_sha256_after"]
    assert set(result["summary_by_batch_size"]) == {"1", "2"}
    for size, summary in result["summary_by_batch_size"].items():
        assert summary["query_count"] == 2
        assert summary["measured_query_count"] == 6
        assert summary["query_batches"] == 2 // int(size)
        assert summary["repeats_per_batch"] == 3
        for mode in ("reference", "hgam", "blocks"):
            assert summary["modes"][mode]["throughput_queries_per_second"] > 0
    sidecar_bytes = {path: path.read_bytes() for path in artifacts.glob("batch-*.json")}
    assert len(sidecar_bytes) == 3
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert training_report.read_bytes() == source_bytes

    def unexpected_measurement(*args, **kwargs):
        raise AssertionError("A complete resume must reuse every sidecar")

    monkeypatch.setattr(profiler, "profile_batch", unexpected_measurement)
    monkeypatch.setattr(sys, "argv", [*command, "--resume"])
    profiler.main()
    resumed = json.loads(output.read_text())
    assert resumed["status"] == "passed"
    assert resumed["validation_replay"]["status"] == "passed"
    assert resumed["validation_replay"]["validated_query_count"] == 2
    assert resumed["completed_batches"] == 3
    assert resumed["completed_batches_this_attempt"] == 0
    assert all(item["reused"] for item in resumed["artifacts"])
    assert resumed["summary_by_batch_size"] == result["summary_by_batch_size"]
    assert resumed["source_session_ids"] == [result["session_id"]]
    assert resumed["session_id"] != result["session_id"]
    assert Path(resumed["archived_previous_report"]["path"]).read_bytes() == result_bytes
    assert {path: path.read_bytes() for path in artifacts.glob("batch-*.json")} == sidecar_bytes
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert training_report.read_bytes() == source_bytes
