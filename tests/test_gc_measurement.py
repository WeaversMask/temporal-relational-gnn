"""Check V2 operation parity and retained GC cost on a real tiny CPU pipeline."""

import copy
import gc
import json

import pytest
import test_pipeline_profile
import torch

from trgnn import gc_measurement as measurement
from trgnn import pipeline_profile
from trgnn.execution import MODES

pipeline = test_pipeline_profile.pipeline


@pytest.fixture(autouse=True)
def restore_gc_state():
    enabled, thresholds, callbacks = gc.isenabled(), gc.get_threshold(), list(gc.callbacks)
    try:
        yield
    finally:
        gc.callbacks[:] = callbacks
        gc.set_threshold(*thresholds)
        if enabled:
            gc.enable()
        else:
            gc.disable()


def test_acyclic_hash_preserves_temporal_ownership_times_and_dictionary_order(pipeline):
    loader, _ = pipeline
    torch.manual_seed(37)
    sample = loader.node_sampler.sample_from_nodes(loader.input_data[[0, 1]])
    ownership = copy.deepcopy(sample)
    ownership.batch["events"] = torch.tensor([1, 0])
    times = copy.deepcopy(sample)
    times.metadata = (sample.metadata[0], sample.metadata[1] + 1)
    reordered = copy.deepcopy(sample)
    reordered.node = dict(reversed(list(sample.node.items())))
    variants = [sample, ownership, times, reordered]
    expected = [pipeline_profile.sampler_fingerprint(value) for value in variants]
    assert len(set(expected[:3])) == 3
    assert expected[0] == expected[3]
    assert [measurement.acyclic_sampler_fingerprint(value) for value in variants] == expected


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stages", [False, True])
def test_new_executor_preserves_operations_waits_and_logits(pipeline, monkeypatch, mode, stages):
    loader, model = pipeline
    operations = []

    def observed(owner, name, label):
        original = getattr(owner, name)

        def call(*args, **kwargs):
            operations.append(label)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, call)

    observed(loader.node_sampler, "sample_from_nodes", "sampling")
    observed(loader, "filter_fn", "gather")
    observed(type(loader.data), "to", "transfer")
    observed(pipeline_profile, "synchronize", "wait")
    for name in ("encode", "message_passing", "readout"):
        observed(model, name, name)
    state = copy.deepcopy(model.state_dict())
    with torch.inference_mode():
        torch.manual_seed(41)
        old = pipeline_profile.execute_pipeline(
            loader, [0, 1], model, "users", mode, "cpu", stages=stages
        )
        original_order = operations[:]
        operations.clear()
        torch.manual_seed(41)
        new, bounds = measurement._execute_pipeline(
            loader, [0, 1], model, "users", mode, "cpu", stages=stages
        )
    assert operations == original_order
    assert operations.count("wait") == (14 if stages else 2)
    assert [value for value in operations if value != "wait"] == [
        "sampling",
        "gather",
        "transfer",
        "encode",
        "message_passing",
        "readout",
    ]
    assert measurement.acyclic_sampler_fingerprint(new.sample) == (
        pipeline_profile.sampler_fingerprint(old.sample)
    )
    torch.testing.assert_close(new.logits, old.logits, atol=0, rtol=0)
    torch.testing.assert_close(model.state_dict(), state, atol=0, rtol=0)
    assert bounds["elapsed_ms"] == new.elapsed_ms
    assert new.elapsed_ms == (bounds["timed_finished_ns"] - bounds["timed_started_ns"]) / 1e6
    assert bounds["timer_read_count"] == (14 if stages else 2)
    assert set(new.stage_ms) == (set(pipeline_profile.STAGES) if stages else set())


def test_natural_policy_never_forces_collection_or_subtracts_timed_gc(pipeline, monkeypatch):
    loader, model = pipeline
    original_message = model.message_passing
    original_collect = gc.collect
    collected = False

    def collect_once_inside_model(*args, **kwargs):
        nonlocal collected
        assert gc.isenabled()
        if not collected:
            collected = True
            original_collect(2)
        return original_message(*args, **kwargs)

    def forbidden_collection(*args, **kwargs):
        raise AssertionError("V2 must not force a collection")

    monkeypatch.setattr(model, "message_passing", collect_once_inside_model)
    monkeypatch.setattr(gc, "collect", forbidden_collection)
    original_executor = pipeline_profile.execute_pipeline
    original_fingerprint = pipeline_profile.sampler_fingerprint
    record = measurement.profile_batch_gc(
        loader, [0, 1], model, "users", repeats=3, warmup=0, seed=43
    )
    telemetry = record["gc_telemetry"]
    assert telemetry["policy"] == "automatic"
    assert pipeline_profile.execute_pipeline is original_executor
    assert pipeline_profile.sampler_fingerprint is original_fingerprint
    measurement.validate_gc_telemetry(record)
    full = [row for row in telemetry["collections"] if row["generation"] == 2]
    assert len(full) == 1
    collection = full[0]
    online = next(
        row for row in telemetry["passes"] if row["pass_id"] == collection["start_pass_id"]
    )
    assert online["pass_kind"] == "online"
    assert online["timed_started_ns"] <= collection["started_ns"]
    assert collection["finished_ns"] <= online["timed_finished_ns"]
    recorded_ms = record["trials"][online["repeat"]]["pipeline_ms"][online["mode"]]
    assert recorded_ms == online["elapsed_ms"]
    assert recorded_ms >= collection["duration_ms"] > 0
    overlap = telemetry["summary"]["online_gc_overlap_ms_by_mode"][online["mode"]]
    assert overlap >= collection["duration_ms"]
    assert telemetry["summary"]["total_gc_ms"] >= collection["duration_ms"]
    assert telemetry["collection_window"]["profile_wall_seconds"] * 1000 >= recorded_ms
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("initially_enabled", [False, True])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_error_restores_executor_fingerprint_callbacks_thresholds_and_enabled_state(
    pipeline, monkeypatch, initially_enabled, failure
):
    loader, model = pipeline
    gc.enable() if initially_enabled else gc.disable()
    gc.set_threshold(1199, 14, 8)

    def existing_callback(phase, info):
        return None

    gc.callbacks.append(existing_callback)
    callbacks, thresholds = list(gc.callbacks), gc.get_threshold()
    executor, fingerprint = pipeline_profile.execute_pipeline, pipeline_profile.sampler_fingerprint

    def fail(*args, **kwargs):
        assert gc.isenabled()
        gc.set_threshold(999, 2, 1)
        raise failure("interrupted measured model")

    monkeypatch.setattr(model, "encode", fail)
    with pytest.raises(failure, match="interrupted measured model") as raised:
        measurement.profile_batch_gc(loader, [0, 1], model, "users", repeats=3, warmup=0)
    assert gc.isenabled() is initially_enabled
    assert gc.get_threshold() == thresholds
    assert gc.callbacks == callbacks
    assert pipeline_profile.execute_pipeline is executor
    assert pipeline_profile.sampler_fingerprint is fingerprint
    assert raised.value.gc_measurement_trace["protocol"]["repeats"] == 3
    json.dumps(raised.value.gc_measurement_trace, allow_nan=False)


@pytest.fixture
def measured_case(pipeline):
    loader, model = pipeline
    return measurement.profile_batch_gc(
        loader, [0, 1], model, "users", repeats=3, warmup=1, seed=47
    )


@pytest.mark.parametrize(
    "tamper",
    ["mode", "exact_timer", "pipeline_time", "summary", "state", "wall", "events"],
)
def test_resume_validator_rejects_inconsistent_gc_measurements(measured_case, tamper):
    record = copy.deepcopy(measured_case)
    telemetry = record["gc_telemetry"]
    online = next(row for row in telemetry["passes"] if row["pass_kind"] == "online")
    if tamper == "mode":
        online["mode"] = "unknown"
    elif tamper == "exact_timer":
        online["timed_finished_ns"] += 1
    elif tamper == "pipeline_time":
        record["trials"][online["repeat"]]["pipeline_ms"][online["mode"]] += 0.01
    elif tamper == "summary":
        telemetry["summary"]["total_gc_ms"] += 0.01
    elif tamper == "state":
        telemetry["gc_enabled_during_case"] = False
    elif tamper == "wall":
        telemetry["collection_window"]["profile_wall_seconds"] += 0.01
    else:
        telemetry["events"].append(
            {
                "at_ns": telemetry["collection_window"]["finished_ns"] + 1,
                "phase": "start",
                "generation": 2,
                "collected": 0,
                "uncollectable": 0,
                "pass_id": -1,
            }
        )
    with pytest.raises(ValueError):
        measurement.validate_gc_telemetry(record)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an NVIDIA CUDA runtime")
def test_cuda_executor_matches_frozen_modes_and_records_real_allocator_peaks(pipeline):
    loader, cpu_model = pipeline
    model = cpu_model.cuda()
    original_state = copy.deepcopy(model.state_dict())
    original_executor = pipeline_profile.execute_pipeline
    original_fingerprint = pipeline_profile.sampler_fingerprint
    with torch.inference_mode():
        for mode in MODES:
            torch.manual_seed(59)
            old = original_executor(loader, [0, 1], model, "users", mode, "cuda")
            torch.manual_seed(59)
            new, bounds = measurement._execute_pipeline(
                loader, [0, 1], model, "users", mode, "cuda"
            )
            assert original_fingerprint(old.sample) == measurement.acyclic_sampler_fingerprint(
                new.sample
            )
            torch.testing.assert_close(new.logits, old.logits, atol=1e-5, rtol=1e-4)
            assert bounds["elapsed_ms"] == new.elapsed_ms > 0
            assert (
                new.cuda_memory["peak_allocated_bytes"]
                >= (new.cuda_memory["baseline_allocated_bytes"])
                > 0
            )
            del old, new
    record = measurement.profile_batch_gc(
        loader, [0, 1], model, "users", repeats=3, warmup=1, seed=61, device="cuda"
    )
    measurement.validate_gc_telemetry(record)
    for trial in record["trials"]:
        for mode in MODES:
            memory = trial["cuda_memory"][mode]
            assert memory["peak_allocated_bytes"] >= memory["baseline_allocated_bytes"] > 0
            assert memory["extra_peak_allocated_bytes"] > 0
            assert memory["peak_reserved_bytes"] >= memory["baseline_reserved_bytes"]
    torch.testing.assert_close(model.state_dict(), original_state, atol=0, rtol=0)
    assert loader.data["users"].tf.device.type == "cpu"
    assert pipeline_profile.execute_pipeline is original_executor
    assert pipeline_profile.sampler_fingerprint is original_fingerprint
