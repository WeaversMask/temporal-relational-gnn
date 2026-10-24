"""Validate diagnostic attribution and restoration on real tiny CPU pipelines."""

import copy
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import test_pipeline_profile
import torch

from trgnn import pipeline_profile
from trgnn import spike_diagnostic as diagnostic
from trgnn.execution import MODES
from trgnn.spike_diagnostic import acyclic_sampler_fingerprint, observe_profile_batch
from trgnn.trained_profile import canonical_sha256, model_sha256, save_batch
from trgnn.training_pilot import file_sha256

pipeline = test_pipeline_profile.pipeline


@pytest.fixture(autouse=True)
def restore_gc_state():
    """Tests may force collection or fail mid-instrumentation; leave no GC policy behind."""
    enabled = gc.isenabled()
    thresholds = gc.get_threshold()
    callbacks = list(gc.callbacks)
    try:
        yield
    finally:
        gc.callbacks[:] = callbacks
        gc.set_threshold(*thresholds)
        if enabled:
            gc.enable()
        else:
            gc.disable()


def _method_snapshot(loader, model):
    targets = [
        (pipeline_profile, "execute_pipeline"),
        (pipeline_profile, "synchronize"),
        (pipeline_profile, "sampler_fingerprint"),
        (pipeline_profile, "time"),
        (loader.node_sampler, "sample_from_nodes"),
        (loader, "filter_fn"),
        (type(loader.data), "to"),
        *((model, name) for name in ("encode", "message_passing", "readout")),
    ]
    return [(owner, name, name in vars(owner), getattr(owner, name)) for owner, name in targets]


def _assert_methods_restored(snapshot):
    for owner, name, had_own_attribute, original in snapshot:
        assert (name in vars(owner)) == had_own_attribute
        assert getattr(owner, name) == original


def test_acyclic_hash_matches_original_temporal_occurrence_serialization(pipeline):
    loader, _ = pipeline
    torch.manual_seed(13)
    sample = loader.node_sampler.sample_from_nodes(loader.input_data[[0, 1]])
    variants = [sample]
    ownership = copy.deepcopy(sample)
    ownership.batch["events"] = torch.tensor([1, 0])
    variants.append(ownership)
    times = copy.deepcopy(sample)
    times.metadata = (sample.metadata[0], sample.metadata[1] + 1)
    variants.append(times)
    hashes = [pipeline_profile.sampler_fingerprint(value) for value in variants]
    assert len(set(hashes)) == 3
    assert [acyclic_sampler_fingerprint(value) for value in variants] == hashes


@pytest.mark.parametrize("fingerprint_policy", ["original", "acyclic"])
def test_observation_preserves_inputs_predictions_and_exact_executor_timer(
    pipeline, fingerprint_policy
):
    loader, model = pipeline
    original_state = copy.deepcopy(model.state_dict())
    snapshot = _method_snapshot(loader, model)
    baseline = pipeline_profile.profile_batch(
        loader, [0, 1], model, "users", repeats=3, warmup=0, seed=23
    )
    profile, trace = observe_profile_batch(
        loader,
        [0, 1],
        model,
        "users",
        repeats=3,
        warmup=1,
        seed=23,
        fingerprint_policy=fingerprint_policy,
    )
    for actual, expected in zip(profile["trials"], baseline["trials"], strict=True):
        for name in (
            "sampler_seed",
            "execution_order",
            "sampler_sha256",
            "sampled_counts",
            "seed_times_unix_seconds",
            "visibility_audit",
        ):
            assert actual[name] == expected[name]
    assert max(profile["max_absolute_logit_error"].values()) < 1e-5
    torch.testing.assert_close(model.state_dict(), original_state, atol=0, rtol=0)
    assert loader.data["users"].tf.device.type == "cpu"
    _assert_methods_restored(snapshot)

    passes = trace["pipeline_passes"]
    assert len(passes) == 3 + 3 * 2 * len(MODES)
    assert sum(row["pass_kind"] == "warmup" for row in passes) == 3
    assert sum(row["pass_kind"] == "online" for row in passes) == 9
    assert sum(row["pass_kind"] == "stage_replay" for row in passes) == 9
    for row in passes:
        assert (
            row["started_ns"]
            <= row["timed_started_ns"]
            <= row["timed_finished_ns"]
            <= row["finished_ns"]
        )
        assert row["elapsed_ms"] == (row["timed_finished_ns"] - row["timed_started_ns"]) / 1e6
        assert row["timer_read_count"] == (14 if row["pass_kind"] == "stage_replay" else 2)
        if row["pass_kind"] == "online":
            assert row["elapsed_ms"] == profile["trials"][row["repeat"]]["pipeline_ms"][row["mode"]]
    assert {span["operation"] for span in trace["host_spans"]} == {
        "sampling",
        "feature_gather",
        "transfer",
        "encode",
        "message_passing",
        "readout",
        "synchronize",
    }
    json.dumps({"profile": profile, "trace": trace}, allow_nan=False)


def test_gc_callback_identifies_collection_inside_the_original_timed_region(pipeline, monkeypatch):
    loader, model = pipeline
    original = model.message_passing
    collected = False

    def collect_once(*args, **kwargs):
        nonlocal collected
        if not collected:
            collected = True
            gc.collect(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "message_passing", collect_once)
    _, trace = observe_profile_batch(loader, [0, 1], model, "users", repeats=3, warmup=0, seed=29)
    events = [
        event
        for event in trace["gc_events"]
        if event["generation"] == 2 and event.get("stage") == "message_passing"
    ]
    assert len(events) == 2
    assert [event["phase"] for event in events] == ["start", "stop"]
    start, finish = events
    assert start["at_ns"] <= finish["at_ns"]
    assert start["pass_kind"] == finish["pass_kind"] == "online"
    assert start["mode"] == finish["mode"]
    matching = [
        row
        for row in trace["pipeline_passes"]
        if row["pass_kind"] == "online"
        and row["mode"] == start["mode"]
        and row["repeat"] == start["repeat"]
    ]
    assert len(matching) == 1
    assert matching[0]["timed_started_ns"] <= start["at_ns"]
    assert finish["at_ns"] <= matching[0]["timed_finished_ns"]


def test_control_collects_outside_executor_and_disables_only_automatic_gc(pipeline, monkeypatch):
    loader, model = pipeline
    original_executor = pipeline_profile.execute_pipeline
    in_executor = False
    calls = []

    def forced_collection(*args, **kwargs):
        calls.append((in_executor, gc.isenabled()))
        return 0

    def checked_executor(*args, **kwargs):
        nonlocal in_executor
        assert not gc.isenabled()
        in_executor = True
        try:
            return original_executor(*args, **kwargs)
        finally:
            in_executor = False

    monkeypatch.setattr(gc, "collect", forced_collection)
    monkeypatch.setattr(pipeline_profile, "execute_pipeline", checked_executor)
    _, trace = observe_profile_batch(
        loader,
        [0, 1],
        model,
        "users",
        repeats=3,
        warmup=1,
        gc_policy="collect_between_passes",
    )
    assert calls == [(False, False)] * len(trace["pipeline_passes"])
    assert len(calls) == 21
    assert pipeline_profile.execute_pipeline is checked_executor


@pytest.mark.parametrize("gc_policy", ["automatic", "collect_between_passes"])
@pytest.mark.parametrize("initially_enabled", [True, False])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failure_restores_gc_callbacks_thresholds_clock_and_all_wrapped_methods(
    pipeline, monkeypatch, gc_policy, initially_enabled, failure
):
    loader, model = pipeline
    if initially_enabled:
        gc.enable()
    else:
        gc.disable()
    gc.set_threshold(1099, 13, 7)

    def existing_callback(phase, info):
        return None

    def added_inside_failure(phase, info):
        return None

    gc.callbacks.append(existing_callback)
    callbacks = list(gc.callbacks)
    thresholds = gc.get_threshold()

    def fail_during_encode(*args, **kwargs):
        gc.set_threshold(1001, 3, 2)
        gc.callbacks.append(added_inside_failure)
        raise failure("simulated interrupted encoder")

    monkeypatch.setattr(model, "encode", fail_during_encode)
    snapshot = _method_snapshot(loader, model)
    with pytest.raises(failure, match="simulated interrupted encoder"):
        observe_profile_batch(
            loader,
            [0, 1],
            model,
            "users",
            repeats=3,
            warmup=0,
            gc_policy=gc_policy,
            fingerprint_policy="acyclic",
        )
    assert gc.callbacks == callbacks
    assert gc.get_threshold() == thresholds
    assert gc.isenabled() is initially_enabled
    _assert_methods_restored(snapshot)


@pytest.mark.parametrize(
    "options", [{"gc_policy": "unrecognized"}, {"fingerprint_policy": "unrecognized"}]
)
def test_invalid_policy_leaves_runtime_unmodified(pipeline, options):
    loader, model = pipeline
    snapshot = _method_snapshot(loader, model)
    callbacks, thresholds, enabled = list(gc.callbacks), gc.get_threshold(), gc.isenabled()
    with pytest.raises(ValueError, match="Unknown"):
        observe_profile_batch(loader, [0, 1], model, "users", **options)
    _assert_methods_restored(snapshot)
    assert gc.callbacks == callbacks
    assert gc.get_threshold() == thresholds
    assert gc.isenabled() is enabled


def test_gc_overlap_uses_exact_timer_intersections_and_ignores_incomplete_passes():
    def event(phase, at_ms, generation):
        return {
            "phase": phase,
            "at_ns": at_ms * 1_000_000,
            "generation": generation,
            "collected": 4 if phase == "stop" else 0,
            "uncollectable": 0,
        }

    trace = {
        "gc_events": [
            event("start", 10, 0),
            event("stop", 40, 0),
            event("start", 60, 1),
            event("stop", 80, 1),
            event("start", 240, 2),
            event("stop", 260, 2),
        ],
        "pipeline_passes": [
            {
                "pass_kind": "online",
                "repeat": 0,
                "mode": "hgam",
                "started_ns": 0,
                "finished_ns": 300_000_000,
                "timed_started_ns": 50_000_000,
                "timed_finished_ns": 250_000_000,
                "elapsed_ms": 200,
            },
            {"pass_kind": "stage_replay"},
            {"pass_kind": "online", "timed_finished_ns": None},
        ],
        "host_spans": [],
    }
    diagnostic._gc_overlap(trace)
    assert len(trace["gc_collections"]) == 3
    assert len(trace["online_gc_overlap"]) == 1
    overlap = trace["online_gc_overlap"][0]
    assert overlap["gc_overlap_ms"] == 30
    assert overlap["gc_generations"] == [1, 2]
    assert trace["summary"]["gc_total_ms"] == 70
    assert trace["summary"]["online_gc_overlap_ms"] == 30


@pytest.fixture
def diagnostic_cli(tmp_path, pipeline, monkeypatch):
    """Exercise real case collection while replacing only external reconstruction."""
    loader, model = pipeline
    source_dir = tmp_path / "historical-inference"
    source_dir.mkdir()
    training_dir = tmp_path / "historical-training"
    training_dir.mkdir()
    checkpoint = tmp_path / "checkpoints/training.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"immutable source checkpoint")
    training_report = training_dir / "training-run.json"
    training_report.write_text('{"status":"passed"}\n')
    source_path = source_dir / "trained-profile.json"
    protocol = {
        "batch_sizes": [32, 128, 512],
        "query_count": 2,
        "repeats": 3,
        "warmup": 0,
        "seed": 42,
        "bootstrap_samples": 2,
    }
    contract = {
        "protocol": protocol,
        "hardware": {"cpu_threads": 1},
        "checkpoint_sha256": file_sha256(checkpoint),
        "training_report_sha256": file_sha256(training_report),
    }
    contract_sha = canonical_sha256(contract)
    planned = {
        "batch_size": 512,
        "batch_index": 0,
        "query_indices": [0, 1],
        "sampling_seed": 2_000_042,
        "model_state_sha256": model_sha256(model),
    }
    profile = pipeline_profile.profile_batch(
        loader, [0, 1], model, "users", repeats=3, warmup=0, seed=planned["sampling_seed"]
    )
    sidecar, _ = save_batch(
        source_dir / "batches",
        planned,
        profile,
        contract_sha,
        "source-session",
        {"session_id": "source-session"},
    )
    source = {
        "status": "passed",
        "benchmark_contract": contract,
        "benchmark_contract_sha256": contract_sha,
        "model_state_sha256_before": model_sha256(model),
        "artifacts": [
            {
                "batch_size": 512,
                "batch_index": 0,
                "sha256": file_sha256(sidecar),
            }
        ],
    }
    source_path.write_text(json.dumps(source, sort_keys=True) + "\n")
    inputs = {
        path: path.read_bytes() for path in (checkpoint, training_report, source_path, sidecar)
    }
    args = SimpleNamespace(
        checkpoint=checkpoint,
        training_report=training_report,
        source_profile=source_path,
        output=tmp_path / "diagnostic/run.json",
        artifacts_dir=tmp_path / "diagnostic/cases",
        data_dir=tmp_path / "unused-data",
        download=False,
        device="cpu",
        batch_indices=[0],
        replay_prefix=False,
    )
    workload = SimpleNamespace(
        model=model,
        graph=loader.data,
        task=SimpleNamespace(entity_table="users"),
        val_queries=pd.DataFrame({"UserId": [0, 0]}),
        checkpoint_metadata={
            "sha256": file_sha256(checkpoint),
            "source_report_sha256": file_sha256(training_report),
        },
        contract={"settings": {"fanouts": [1, 1]}},
    )
    calls = {"loads": 0}

    def load(*args, **kwargs):
        assert torch.get_num_threads() == contract["hardware"]["cpu_threads"] == 1
        assert torch.initial_seed() == protocol["seed"] == 42
        calls["loads"] += 1
        return workload

    monkeypatch.setattr(diagnostic, "parse_args", lambda: args)
    monkeypatch.setattr(diagnostic, "SOURCE_PROFILE_SHA256", file_sha256(source_path))
    monkeypatch.setattr(diagnostic, "load_trained_workload", load)
    monkeypatch.setattr(diagnostic, "_check_source_contract", lambda *args: (contract, {}, []))
    monkeypatch.setattr(
        diagnostic,
        "replay_saved_validation",
        lambda *args: {"status": "passed", "validated_query_count": 2},
    )
    monkeypatch.setattr(diagnostic, "make_query_loader", lambda *args: loader)
    return SimpleNamespace(args=args, inputs=inputs, calls=calls, source=source)


def test_cli_preserves_historical_inputs_and_completed_output_on_rerun(diagnostic_cli):
    run = diagnostic_cli
    torch.set_num_threads(2)
    torch.manual_seed(999)
    diagnostic.main()
    report_bytes = run.args.output.read_bytes()
    report = json.loads(report_bytes)
    assert report["status"] == "passed"
    assert report["model_state_sha256_before"] == report["model_state_sha256_after"]
    assert len(report["cases"]) == report["expected_cases"] == 3
    assert all(case["archived_input_match"] for case in report["cases"])
    case_bytes = {path: path.read_bytes() for path in run.args.artifacts_dir.glob("*.json")}
    assert len(case_bytes) == 6
    with pytest.raises(ValueError, match="already exists"):
        diagnostic.main()
    assert run.args.output.read_bytes() == report_bytes
    assert {path: path.read_bytes() for path in case_bytes} == case_bytes
    assert {path: path.read_bytes() for path in run.inputs} == run.inputs
    assert run.calls["loads"] == 1


@pytest.mark.parametrize(
    "destination",
    [
        "source_profile",
        "training_report",
        "checkpoint",
        "historical_sibling",
        "artifact_ancestor",
        "artifact_history_child",
        "artifact_code",
    ],
)
def test_cli_rejects_protected_destinations_before_writing_or_loading(diagnostic_cli, destination):
    run = diagnostic_cli
    output = run.args.output
    if destination in ("source_profile", "training_report", "checkpoint"):
        run.args.output = getattr(run.args, destination)
    elif destination == "historical_sibling":
        run.args.output = run.args.source_profile.parent / "new-diagnostic.json"
    elif destination == "artifact_ancestor":
        run.args.artifacts_dir = run.args.source_profile.parent.parent
    elif destination == "artifact_history_child":
        run.args.artifacts_dir = run.args.source_profile.parent / "new-cases"
    else:
        run.args.artifacts_dir = Path(diagnostic.__file__).parent
    with pytest.raises(ValueError, match="already exists|protected"):
        diagnostic.main()
    assert not output.exists()
    assert {path: path.read_bytes() for path in run.inputs} == run.inputs
    assert run.calls["loads"] == 0


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_cli_persists_failure_trace_and_prior_cases_without_touching_inputs(
    diagnostic_cli, monkeypatch, failure
):
    run = diagnostic_cli
    original = diagnostic.observe_profile_batch
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            error = failure("interrupted second diagnostic case")
            error.diagnostic_trace = {"gc_events": [], "pipeline_passes": [], "summary": {}}
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(diagnostic, "observe_profile_batch", interrupted)
    with pytest.raises(failure, match="interrupted second diagnostic case"):
        diagnostic.main()
    report = json.loads(run.args.output.read_text())
    assert report["status"] == ("interrupted" if failure is KeyboardInterrupt else "failed")
    assert report["error"]["type"] == failure.__name__
    assert len(report["cases"]) == 1
    first_case = report["cases"][0]
    assert file_sha256(Path(first_case["profile_path"])) == first_case["profile_sha256"]
    assert file_sha256(Path(first_case["trace_path"])) == first_case["trace_sha256"]
    partial_path = Path(report["failed_case"]["trace_path"])
    assert file_sha256(partial_path) == report["failed_case"]["trace_sha256"]
    assert json.loads(partial_path.read_text())["status"] == "partial"
    assert {path: path.read_bytes() for path in run.inputs} == run.inputs


@pytest.mark.parametrize("changed", ["hardware", "source_runtime_versions", "protocol"])
def test_source_contract_allows_hardware_variation_but_rejects_input_protocol_changes(
    monkeypatch, changed
):
    expected = {
        "protocol": {
            "batch_sizes": [32, 128, 512],
            "repeats": 20,
            "warmup": 5,
            "seed": 42,
            "bootstrap_samples": 2000,
        },
        "hardware": {"cpu_model": "first CPU"},
        "source_runtime_versions": {"torch": "fixed"},
    }
    actual = copy.deepcopy(expected)
    actual[changed] = {"different": True}
    monkeypatch.setattr(diagnostic, "make_benchmark_contract", lambda *args: actual)
    _, differences, failed = diagnostic._check_source_contract(
        object(), SimpleNamespace(device="cpu"), {"benchmark_contract": expected}
    )
    assert set(differences) == {changed}
    assert failed == ([] if changed == "hardware" else [changed])
