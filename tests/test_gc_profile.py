"""Protect source evidence and exercise real CPU V2 collection and resume."""

import copy
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import test_pipeline_profile
import torch
from relbench.modeling.graph import AttachTargetTransform
from torch_geometric.loader import NodeLoader

from trgnn import gc_profile as profiler
from trgnn import pipeline_profile, training
from trgnn.artifacts import environment
from trgnn.gc_measurement import acyclic_sampler_fingerprint
from trgnn.trained_profile import (
    batch_plan,
    canonical_sha256,
    make_benchmark_contract,
    model_sha256,
)
from trgnn.trained_profile import (
    save_batch as save_legacy_batch,
)
from trgnn.training_pilot import file_sha256

pipeline = test_pipeline_profile.pipeline


@pytest.fixture(autouse=True)
def restore_gc_state():
    enabled, thresholds, callbacks = gc.isenabled(), gc.get_threshold(), list(gc.callbacks)
    gc.enable()
    try:
        yield
    finally:
        gc.callbacks[:] = callbacks
        gc.set_threshold(*thresholds)
        if enabled:
            gc.enable()
        else:
            gc.disable()


@pytest.fixture
def gc_run(tmp_path, pipeline, monkeypatch):
    """Four synthetic temporal queries permit a meaningful within-size partial resume."""
    fixture_loader, model = pipeline
    loader = NodeLoader(
        fixture_loader.data,
        test_pipeline_profile.OccurrenceSampler(),
        input_nodes=("users", torch.tensor([0, 0, 0, 0])),
        input_time=torch.tensor([1583020800, 1601510400, 1583020800, 1601510400]),
        transform=AttachTargetTransform("users", torch.tensor([0.0, 1.0, 0.0, 1.0])),
        num_workers=0,
        batch_size=2,
        filter_per_worker=False,
    )
    source_dir, training_dir = tmp_path / "source-inference", tmp_path / "source-training"
    source_dir.mkdir()
    training_dir.mkdir()
    checkpoint = tmp_path / "checkpoints/training.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"immutable checkpoint reconstructed by the external loader")
    training_report = training_dir / "training-run.json"
    training_report.write_text('{"status":"passed"}\n')
    args = SimpleNamespace(
        checkpoint=checkpoint,
        training_report=training_report,
        source_profile=source_dir / "trained-profile.json",
        output=tmp_path / "v2/gc-aware-profile.json",
        artifacts_dir=tmp_path / "v2/batches",
        data_dir=tmp_path / "unused-data",
        download=False,
        device="cpu",
        batch_sizes=[2],
        repeats=3,
        warmup=1,
        seed=42,
        bootstrap_samples=20,
        resume=False,
    )
    runtime = environment()
    runtime["gpu"] = None
    workload = SimpleNamespace(
        model=model,
        graph=loader.data,
        task=SimpleNamespace(entity_table="users"),
        val_queries=pd.DataFrame({"UserId": [0, 0, 0, 0]}),
        checkpoint_metadata={
            "sha256": file_sha256(checkpoint),
            "source_report_sha256": file_sha256(training_report),
            "input_contract_sha256": "a" * 64,
            "completed_steps": 2000,
        },
        contract={
            "settings": {"fanouts": [1, 1]},
            "runtime_versions": {"torch": torch.__version__},
        },
        environment=runtime,
        audits={"status": "passed"},
        setup_seconds={},
    )
    contract = make_benchmark_contract(workload, args, {"status": "unavailable"})
    contract_sha = canonical_sha256(contract)
    original_hash = model_sha256(model)
    plans = list(batch_plan(4, [2], 3, 42))
    artifacts = []
    for plan in plans:
        plan["model_state_sha256"] = original_hash
        profile = pipeline_profile.profile_batch(
            loader,
            plan["query_indices"],
            model,
            "users",
            repeats=3,
            warmup=1,
            seed=plan["sampling_seed"],
        )
        path, _ = save_legacy_batch(
            source_dir / "batches",
            plan,
            profile,
            contract_sha,
            "legacy-source-session",
            {"session_id": "legacy-source-session"},
        )
        artifacts.append({**plan, "sha256": file_sha256(path)})
    source = {
        "status": "passed",
        "measurements_complete": True,
        "completed_batches": 2,
        "expected_batches": 2,
        "benchmark_contract": contract,
        "benchmark_contract_sha256": contract_sha,
        "model_state_sha256_before": original_hash,
        "model_state_sha256_after": original_hash,
        "artifacts": artifacts,
    }
    args.source_profile.write_text(json.dumps(source, sort_keys=True) + "\n")
    inputs = {
        path: path.read_bytes()
        for path in [checkpoint, training_report, *source_dir.rglob("*.json")]
    }
    calls = {"loads": 0, "replays": 0, "profiles": 0}

    def load(*args, **kwargs):
        assert torch.get_num_threads() == 1
        assert torch.initial_seed() == 42
        calls["loads"] += 1
        return workload

    def replay(*args, **kwargs):
        assert training.sampler_fingerprint is acyclic_sampler_fingerprint
        calls["replays"] += 1
        return {"status": "passed", "validated_query_count": 4}

    original_profile = profiler.profile_batch_gc

    def measured(*args, **kwargs):
        calls["profiles"] += 1
        return original_profile(*args, **kwargs)

    monkeypatch.setattr(profiler, "parse_args", lambda: args)
    monkeypatch.setattr(profiler, "SOURCE_PROFILE_SHA256", file_sha256(args.source_profile))
    monkeypatch.setattr(profiler, "load_trained_workload", load)
    monkeypatch.setattr(profiler, "replay_saved_validation", replay)
    monkeypatch.setattr(profiler, "make_query_loader", lambda *args: loader)
    monkeypatch.setattr(profiler, "profile_batch_gc", measured)
    return SimpleNamespace(
        args=args,
        inputs=inputs,
        source=source,
        plans=plans,
        calls=calls,
        workload=workload,
        profile=measured,
    )


def _unchanged_inputs(run):
    assert {path: path.read_bytes() for path in run.inputs} == run.inputs


def _resign(record):
    record.pop("artifact_payload_sha256", None)
    record["artifact_payload_sha256"] = canonical_sha256(record)


def test_complete_collection_and_resume_preserve_sources_and_raw_measurements(gc_run, monkeypatch):
    run = gc_run
    torch.set_num_threads(2)
    torch.manual_seed(999)
    profiler.main()
    original_report = run.args.output.read_bytes()
    report = json.loads(original_report)
    assert report["status"] == "passed"
    assert report["measurements_complete"] and report["gc_version"] == 2
    assert report["completed_batches"] == report["completed_batches_this_attempt"] == 2
    assert report["model_state_sha256_before"] == report["model_state_sha256_after"]
    assert not report["training_performed"]
    assert report["benchmark_contract"]["gc_policy"]["forced_collection"] is False
    assert "overhead present" in report["latency_scope"]
    assert report["collection_program_window"]["wall_seconds"] > 0
    summary = report["summary_by_batch_size"]["2"]
    assert summary["query_count"] == 4 and summary["measured_query_count"] == 12
    assert summary["mixed_session_measurement"] is False
    assert summary["comparisons"]["hgam"]["cluster_bootstrap_95_percent_ci"] is not None
    sidecars = {path: path.read_bytes() for path in run.args.artifacts_dir.glob("*.json")}
    assert len(sidecars) == 2
    _unchanged_inputs(run)

    with pytest.raises(FileExistsError):
        profiler.main()
    assert run.args.output.read_bytes() == original_report
    assert {path: path.read_bytes() for path in sidecars} == sidecars

    def forbidden_measurement(*args, **kwargs):
        raise AssertionError("A complete resume must not collect new timings")

    monkeypatch.setattr(profiler, "profile_batch_gc", forbidden_measurement)
    run.args.resume = True
    profiler.main()
    resumed = json.loads(run.args.output.read_text())
    assert resumed["status"] == "passed"
    assert resumed["completed_batches_this_attempt"] == 0
    assert all(item["reused"] for item in resumed["artifacts"])
    assert resumed["summary_by_batch_size"] == report["summary_by_batch_size"]
    assert Path(resumed["archived_previous_report"]["path"]).read_bytes() == original_report
    assert {path: path.read_bytes() for path in sidecars} == sidecars
    assert run.calls["replays"] == 2
    _unchanged_inputs(run)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_partial_resume_preserves_first_case_and_suppresses_mixed_session_intervals(
    gc_run, monkeypatch, failure
):
    run = gc_run
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            model = args[2]
            with monkeypatch.context() as patch:

                def fail(*args, **kwargs):
                    raise failure("interrupted V2 encoder")

                patch.setattr(model, "encode", fail)
                return run.profile(*args, **kwargs)
        return run.profile(*args, **kwargs)

    monkeypatch.setattr(profiler, "profile_batch_gc", fail_second)
    with pytest.raises(failure, match="interrupted V2 encoder"):
        profiler.main()
    failed_report = json.loads(run.args.output.read_text())
    assert failed_report["status"] == "failed"
    assert failed_report["completed_batches"] == 1
    trace = failed_report["failed_case_gc_trace"]
    assert trace["batch_index"] == 1
    assert file_sha256(Path(trace["path"])) == trace["sha256"]
    json.loads(Path(trace["path"]).read_text())
    first = next(run.args.artifacts_dir.glob("*.json"))
    first_bytes = first.read_bytes()
    _unchanged_inputs(run)

    monkeypatch.setattr(profiler, "profile_batch_gc", run.profile)
    run.args.resume = True
    profiler.main()
    report = json.loads(run.args.output.read_text())
    assert report["status"] == "passed" and report["completed_batches_this_attempt"] == 1
    assert report["mixed_session_measurement"] is True
    summary = report["summary_by_batch_size"]["2"]
    assert summary["mixed_session_measurement"] is True
    assert summary["comparisons"]["hgam"]["cluster_bootstrap_95_percent_ci"] is None
    assert first.read_bytes() == first_bytes
    _unchanged_inputs(run)


@pytest.mark.parametrize(
    "target", ["source", "training", "checkpoint", "history_child", "ancestor", "overlap"]
)
def test_guard_rejects_protected_paths_before_writing_or_loading(gc_run, target):
    run = gc_run
    original_output = run.args.output
    if target == "source":
        run.args.output = run.args.source_profile
    elif target == "training":
        run.args.output = run.args.training_report
    elif target == "checkpoint":
        run.args.output = run.args.checkpoint
    elif target == "history_child":
        run.args.artifacts_dir = run.args.source_profile.parent / "new-batches"
    elif target == "ancestor":
        run.args.artifacts_dir = run.args.source_profile.parent.parent
    else:
        run.args.artifacts_dir = run.args.output.parent
    with pytest.raises(ValueError, match="protected|separate"):
        profiler.main()
    assert not original_output.exists()
    assert run.calls["loads"] == 0
    _unchanged_inputs(run)


def test_v1_resume_is_rejected_without_overwriting_the_historical_report(gc_run):
    run = gc_run
    run.args.output.parent.mkdir()
    old = b'{"status":"passed","format_version":1}\n'
    run.args.output.write_bytes(old)
    run.args.resume = True
    with pytest.raises(ValueError, match="old or diagnostic"):
        profiler.main()
    assert run.args.output.read_bytes() == old
    assert run.calls["loads"] == 0
    _unchanged_inputs(run)


@pytest.mark.parametrize("tamper", ["fingerprint", "gc_summary", "thresholds"])
def test_resigned_sidecar_tampering_is_rejected_beyond_its_checksum(gc_run, tamper):
    run = gc_run
    profiler.main()
    report = json.loads(run.args.output.read_text())
    path = Path(report["artifacts"][0]["path"])
    record = json.loads(path.read_text())
    planned = {
        **run.plans[0],
        "gc_thresholds": report["benchmark_contract"]["gc_policy"]["thresholds"],
    }
    _, refs = profiler.load_source_profile(run.args)
    source_record = refs[(2, 0)].load()
    if tamper == "fingerprint":
        record["trials"][0]["sampler_sha256"] = "0" * 64
    elif tamper == "gc_summary":
        record["gc_telemetry"]["summary"]["total_gc_ms"] += 0.1
    else:
        record["gc_thresholds"][0] += 1
    _resign(record)
    with pytest.raises(ValueError):
        profiler.validate_artifact(
            record, planned, report["benchmark_contract_sha256"], 3, source_record
        )
    _unchanged_inputs(run)


@pytest.mark.parametrize("mismatch", ["protocol", "gpu_class", "gc_disabled"])
def test_contract_rejects_runtime_or_natural_gc_policy_mismatch(gc_run, monkeypatch, mismatch):
    run = gc_run
    actual = copy.deepcopy(run.source["benchmark_contract"])
    if mismatch == "protocol":
        actual["protocol"]["seed"] += 1
    elif mismatch == "gpu_class":
        actual["hardware"]["gpu"] = {"name": "another GPU"}
    else:
        gc.disable()
    monkeypatch.setattr(profiler, "make_benchmark_contract", lambda *args: actual)
    with pytest.raises(ValueError, match="contract mismatch|GPU class|already be enabled"):
        profiler.make_gc_contract(run.workload, run.args, run.source, {})
    _unchanged_inputs(run)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_replay_audit_uses_identical_hash_and_restores_the_legacy_alias_on_error(pipeline, failure):
    loader, _ = pipeline
    torch.manual_seed(53)
    sample = loader.node_sampler.sample_from_nodes(loader.input_data[[0, 1]])
    original_alias = training.sampler_fingerprint
    original_pipeline_alias = pipeline_profile.sampler_fingerprint
    expected = original_alias(sample)
    with pytest.raises(failure, match="interrupted source replay"):
        with profiler.acyclic_replay_audit():
            assert training.sampler_fingerprint is acyclic_sampler_fingerprint
            assert training.sampler_fingerprint(sample) == expected
            assert pipeline_profile.sampler_fingerprint is original_pipeline_alias
            raise failure("interrupted source replay")
    assert training.sampler_fingerprint is original_alias
    assert pipeline_profile.sampler_fingerprint is original_pipeline_alias


def test_source_references_recheck_bytes_without_retaining_stale_batch_contents(gc_run):
    run = gc_run
    _, refs = profiler.load_source_profile(run.args)
    reference = refs[(2, 0)]
    path = run.args.source_profile.parent / "batches/batch-000002-000000.json"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="bytes|checksum"):
            reference.load()
    finally:
        path.write_bytes(original)
    assert reference.load()["batch_index"] == 0
    _unchanged_inputs(run)


def test_resume_rejects_corrupted_v2_case_before_collecting_more_measurements(gc_run, monkeypatch):
    run = gc_run
    profiler.main()
    original_report = run.args.output.read_bytes()
    report = json.loads(original_report)
    path = Path(report["artifacts"][0]["path"])
    record = json.loads(path.read_text())
    record["gc_telemetry"]["summary"]["total_gc_ms"] += 1
    _resign(record)
    path.write_text(json.dumps(record, sort_keys=True) + "\n")
    corrupted_bytes = path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid restored evidence must stop before any new timing")

    monkeypatch.setattr(profiler, "profile_batch_gc", forbidden)
    run.args.resume = True
    with pytest.raises(ValueError, match="GC collection|overlap summary"):
        profiler.main()
    failed = json.loads(run.args.output.read_text())
    assert failed["status"] == "failed" and failed["completed_batches_this_attempt"] == 0
    assert Path(failed["archived_previous_report"]["path"]).read_bytes() == original_report
    assert path.read_bytes() == corrupted_bytes
    _unchanged_inputs(run)
