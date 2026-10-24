"""Exercise profiling, resume and failure boundaries without a GPU or downloads."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import trgnn.trained_profile as profiler
from trgnn.artifacts import environment
from trgnn.execution import MODES
from trgnn.pipeline_profile import STAGES


def paired_profile(indices, repeats, seed):
    return {
        "trials": [
            {
                "sampler_seed": seed + repeat,
                "execution_order": list(MODES),
                "sampler_sha256": "a" * 64,
                "visibility_audit": {"status": "passed"},
                "pipeline_ms": {"reference": 3.0, "hgam": 2.0, "blocks": 1.0},
                "stage_ms": {mode: {stage: 0.5 for stage in STAGES} for mode in MODES},
                "cuda_memory": {mode: None for mode in MODES},
                "sampled_counts": {"nodes": {"users": [len(indices), 0, 0]}, "edges": {}},
                "seed_times_unix_seconds": [1601510400] * len(indices),
            }
            for repeat in range(repeats)
        ],
        "max_absolute_logit_error": {mode: 0.0 for mode in MODES},
        "max_absolute_loss_error": {mode: 0.0 for mode in MODES},
    }


@pytest.fixture
def profile_run(tmp_path, monkeypatch):
    checkpoint = tmp_path / "training.pt"
    checkpoint.write_bytes(b"source model checkpoint, never overwritten")
    source = tmp_path / "source-report.json"
    source.write_text('{"status":"passed"}')
    model = torch.nn.Linear(2, 1).eval()
    model.register_buffer("scalar_counter", torch.tensor(0))
    runtime = environment()
    runtime["gpu"] = None
    workload = SimpleNamespace(
        model=model,
        graph=object(),
        task=SimpleNamespace(entity_table="users"),
        val_queries=pd.DataFrame({"UserId": range(6)}),
        checkpoint_metadata={
            "sha256": profiler.file_sha256(checkpoint),
            "source_report_sha256": profiler.file_sha256(source),
            "input_contract_sha256": "b" * 64,
            "completed_steps": 2000,
        },
        contract={
            "runtime_versions": {"torch": torch.__version__},
            "settings": {"fanouts": [1, 1]},
        },
        setup_seconds={},
        environment=runtime,
        audits={"status": "passed", "source_files_match": True},
    )
    calls = {"loads": 0, "replays": 0, "profiles": []}

    def load(*args, **kwargs):
        calls["loads"] += 1
        return workload

    def replay(*args, **kwargs):
        calls["replays"] += 1
        return {"status": "passed", "validated_query_count": 6, "max_absolute_logit_error": 0.0}

    def profile(loader, indices, received_model, entity, *, repeats, warmup, seed, device):
        assert calls["replays"] >= 1
        assert received_model is model and not model.training
        assert torch.initial_seed() == seed
        calls["profiles"].append((list(indices), repeats, seed))
        return paired_profile(indices, repeats, seed)

    monkeypatch.setattr(profiler, "load_trained_workload", load)
    monkeypatch.setattr(profiler, "replay_saved_validation", replay)
    monkeypatch.setattr(profiler, "make_query_loader", lambda *args: object())
    monkeypatch.setattr(profiler, "profile_batch", profile)
    monkeypatch.setattr(profiler, "gpu_identity", lambda device: {"status": "unavailable"})
    output = tmp_path / "run/trained-profile.json"
    directory = tmp_path / "run/batches"
    arguments = [
        "trained_profile",
        "--checkpoint",
        str(checkpoint),
        "--training-report",
        str(source),
        "--batch-sizes",
        "2",
        "4",
        "--repeats",
        "3",
        "--warmup",
        "0",
        "--bootstrap-samples",
        "20",
        "--output",
        str(output),
        "--artifacts-dir",
        str(directory),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    return SimpleNamespace(
        arguments=arguments,
        output=output,
        directory=directory,
        workload=workload,
        calls=calls,
        profile=profile,
        checkpoint=checkpoint,
        source=source,
    )


def test_complete_cohort_profile_preserves_model_and_source_evidence(profile_run):
    run = profile_run
    original = profiler.model_sha256(run.workload.model)
    profiler.main()
    report = json.loads(run.output.read_text())
    assert report["status"] == "passed" and report["trained"] is True
    assert report["measurements_complete"] is True and report["training_performed"] is False
    assert report["completed_batches"] == report["expected_batches"] == 5
    assert run.calls["loads"] == run.calls["replays"] == 1
    assert report["model_state_sha256_before"] == report["model_state_sha256_after"] == original
    assert profiler.file_sha256(run.checkpoint) == run.workload.checkpoint_metadata["sha256"]
    assert (
        profiler.file_sha256(run.source) == run.workload.checkpoint_metadata["source_report_sha256"]
    )
    assert report["benchmark_contract"]["profiling_runtime"]["python"]
    for size in (2, 4):
        records = [
            json.loads(Path(r["path"]).read_text())
            for r in report["artifacts"]
            if r["batch_size"] == size
        ]
        assert [i for r in records for i in r["query_indices"]] == list(range(6))
        assert all(r["model_state_sha256"] == original for r in records)
        summary = report["summary_by_batch_size"][str(size)]
        assert summary["query_count"] == 6 and summary["measured_query_count"] == 18
        assert summary["comparisons"]["hgam"]["blocks_latency_reduction_percent"] == 50


@pytest.mark.parametrize("interruption", [RuntimeError, KeyboardInterrupt])
def test_partial_attempt_resume_replays_source_and_reuses_only_complete_batches(
    profile_run, monkeypatch, interruption
):
    run = profile_run
    count = 0

    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise interruption("stopped between query batches")
        return run.profile(*args, **kwargs)

    monkeypatch.setattr(profiler, "profile_batch", fail_second)
    with pytest.raises(interruption, match="stopped between"):
        profiler.main()
    first = json.loads(run.output.read_text())
    assert first["status"] == "failed" and first["completed_batches"] == 1
    assert first["measurements_complete"] is False
    saved_path = Path(first["artifacts"][0]["path"])
    saved = saved_path.read_bytes()
    original_report = run.output.read_bytes()
    monkeypatch.setattr(profiler, "profile_batch", run.profile)
    monkeypatch.setattr(sys, "argv", [*run.arguments, "--resume"])
    profiler.main()
    resumed = json.loads(run.output.read_text())
    assert resumed["status"] == "passed"
    assert resumed["completed_batches_this_attempt"] == 4
    assert resumed["artifacts"][0]["reused"] is True
    assert saved_path.read_bytes() == saved
    assert Path(resumed["archived_previous_report"]["path"]).read_bytes() == original_report
    assert run.calls["loads"] == run.calls["replays"] == 2
    assert len(resumed["source_session_ids"]) == 2


def test_resume_rejects_altered_sidecar_before_measuring_more(profile_run, monkeypatch):
    run = profile_run
    profiler.main()
    report = json.loads(run.output.read_text())
    path = Path(report["artifacts"][0]["path"])
    raw = json.loads(path.read_text())
    raw["query_indices"][0] = 99
    path.write_text(json.dumps(raw))
    before = len(run.calls["profiles"])
    monkeypatch.setattr(sys, "argv", [*run.arguments, "--resume"])
    with pytest.raises(ValueError, match="checksum"):
        profiler.main()
    assert len(run.calls["profiles"]) == before
    assert json.loads(run.output.read_text())["status"] == "failed"


@pytest.mark.parametrize("changed", ["repeats", "python"])
def test_resume_rejects_changed_measurement_protocol(profile_run, monkeypatch, changed):
    run = profile_run
    profiler.main()
    arguments = list(run.arguments)
    if changed == "repeats":
        arguments[arguments.index("--repeats") + 1] = "4"
    else:
        run.workload.environment["python"] = "different-interpreter"
    before = len(run.calls["profiles"])
    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    with pytest.raises(ValueError, match="protocol"):
        profiler.main()
    assert len(run.calls["profiles"]) == before


def test_model_mutation_never_commits_a_resumable_measurement(profile_run, monkeypatch):
    run = profile_run
    original = run.workload.model.weight.detach().clone()

    def mutate(*args, **kwargs):
        result = run.profile(*args, **kwargs)
        with torch.no_grad():
            run.workload.model.weight.add_(1)
        return result

    monkeypatch.setattr(profiler, "profile_batch", mutate)
    with pytest.raises(AssertionError, match="Frozen model"):
        profiler.main()
    assert not list(run.directory.glob("batch-*.json"))
    assert json.loads(run.output.read_text())["completed_batches"] == 0
    with torch.no_grad():
        run.workload.model.weight.copy_(original)
    monkeypatch.setattr(profiler, "profile_batch", run.profile)
    monkeypatch.setattr(sys, "argv", [*run.arguments, "--resume"])
    profiler.main()
    assert json.loads(run.output.read_text())["completed_batches_this_attempt"] == 5


def test_failed_saved_validation_produces_no_timing_artifact(profile_run, monkeypatch):
    run = profile_run

    def fail_replay(*args, **kwargs):
        raise ValueError("saved logits mismatch")

    monkeypatch.setattr(profiler, "replay_saved_validation", fail_replay)
    with pytest.raises(ValueError, match="saved logits"):
        profiler.main()
    assert run.calls["profiles"] == []
    assert not list(run.directory.glob("batch-*.json"))


def test_fresh_rerun_cannot_overwrite_prior_results(profile_run):
    profiler.main()
    before = profile_run.output.read_bytes()
    with pytest.raises(FileExistsError, match="--resume"):
        profiler.main()
    assert profile_run.output.read_bytes() == before


def test_model_digest_covers_scalar_buffers_and_weights():
    model = torch.nn.Linear(2, 1)
    model.register_buffer("counter", torch.tensor(0))
    original = profiler.model_sha256(model)
    model.counter.add_(1)
    assert profiler.model_sha256(model) != original


def test_sidecar_rejects_missing_paired_fields_even_with_recomputed_checksum(profile_run):
    profiler.main()
    report = json.loads(profile_run.output.read_text())
    raw = json.loads(Path(report["artifacts"][0]["path"]).read_text())
    raw["trials"][0]["cuda_memory"].pop("hgam")
    payload = {key: value for key, value in raw.items() if key != "artifact_payload_sha256"}
    raw["artifact_payload_sha256"] = profiler.canonical_sha256(payload)
    planned = {
        key: raw[key]
        for key in (
            "batch_size",
            "batch_index",
            "query_indices",
            "sampling_seed",
            "model_state_sha256",
        )
    }
    with pytest.raises(ValueError, match="profile field"):
        profiler.validate_artifact(raw, planned, report["benchmark_contract_sha256"], 3)
