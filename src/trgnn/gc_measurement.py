"""Natural-GC measurement with acyclic input auditing and exact timer bounds.

The frozen paired protocol supplies sampling, ordering, audits, and stage replays.
Only its executor and fingerprint visitor are replaced temporarily. GC callbacks
add a small observer cost and can move collection cadence; no cost is subtracted.
This process-global instrumentation requires one serial caller.
"""

import gc
import hashlib
import json
import math
import time
from datetime import datetime, timezone

import torch

from trgnn import pipeline_profile
from trgnn.execution import MODES

FORMAT_VERSION = 1


def _fingerprint_value(value, digest):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Expected a CPU sampler output")
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.contiguous().numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _fingerprint_value(key, digest)
            _fingerprint_value(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(str((type(value).__name__, len(value))).encode())
        for item in value:
            _fingerprint_value(item, digest)
    else:
        digest.update(repr(value).encode())
    digest.update(b"\0")


def acyclic_sampler_fingerprint(sample):
    """Preserve every byte of the frozen hash without a recursive closure cycle."""
    digest = hashlib.sha256()
    for field in ("node", "row", "col", "edge", "batch", "num_sampled_nodes", "num_sampled_edges"):
        _fingerprint_value(field, digest)
        _fingerprint_value(getattr(sample, field), digest)
    _fingerprint_value(sample.metadata, digest)
    return digest.hexdigest()


def _execute_pipeline(loader, indices, model, entity, mode, device, *, stages=False):
    """Preserve frozen operations and waits; return exact bounds after timing."""
    timings = {}

    def stage(name, operation):
        if not stages:
            return operation()
        pipeline_profile.synchronize(device)
        started = time.perf_counter_ns()
        value = operation()
        pipeline_profile.synchronize(device)
        timings[name] = (time.perf_counter_ns() - started) / 1e6
        return value

    pipeline_profile.synchronize(device)
    initial = reserved_before = None
    if device == "cuda":
        initial = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter_ns()
    sample = stage(
        "sampling", lambda: loader.node_sampler.sample_from_nodes(loader.input_data[indices])
    )
    batch = stage("feature_gather", lambda: loader.filter_fn(sample))
    batch = stage("transfer", lambda: batch.to(device))
    if stages:
        features = stage("encoding", lambda: model.encode(batch, entity))
        result = stage("message_passing", lambda: model.message_passing(features, batch, mode))
        logits = stage("readout", lambda: model.readout(result, batch, entity))
    else:
        logits = model(batch, entity, mode)
    pipeline_profile.synchronize(device)
    finished = time.perf_counter_ns()
    elapsed_ms = (finished - started) / 1e6
    memory = None
    if device == "cuda":
        memory = {
            "baseline_allocated_bytes": initial,
            "baseline_reserved_bytes": reserved_before,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "extra_peak_allocated_bytes": torch.cuda.max_memory_allocated() - initial,
            "scope": "PyTorch allocator; excludes other GPU users and driver memory",
        }
    return (
        pipeline_profile.PipelineResult(sample, batch, logits, elapsed_ms, timings, memory),
        {
            "timed_started_ns": started,
            "timed_finished_ns": finished,
            "elapsed_ms": elapsed_ms,
            "timer_read_count": 2 + 2 * len(timings),
        },
    )


def _summarize(events, passes):
    pending, collections = None, []
    for event in events:
        if event["phase"] == "start":
            if pending is not None:
                raise ValueError("Overlapping or unmatched GC start events")
            pending = event
        else:
            if pending is None or pending["generation"] != event["generation"]:
                raise ValueError("Unmatched GC stop event")
            if pending["pass_id"] != event["pass_id"]:
                raise ValueError("GC changed pipeline pass while collecting")
            collections.append(
                {
                    "started_ns": pending["at_ns"],
                    "finished_ns": event["at_ns"],
                    "generation": event["generation"],
                    "collected": event["collected"],
                    "uncollectable": event["uncollectable"],
                    "start_pass_id": pending["pass_id"],
                    "duration_ms": (event["at_ns"] - pending["at_ns"]) / 1e6,
                }
            )
            pending = None
    if pending is not None:
        raise ValueError("Unmatched GC start event")
    overlaps = {mode: [] for mode in MODES}
    counts = {mode: 0 for mode in MODES}
    for item in passes:
        if item["pass_kind"] != "online":
            continue
        start, end = item["timed_started_ns"], item["timed_finished_ns"]
        overlap_ns = sum(
            max(0, min(end, event["finished_ns"]) - max(start, event["started_ns"]))
            for event in collections
        )
        overlaps[item["mode"]].append(overlap_ns / 1e6)
        counts[item["mode"]] += overlap_ns > 0
    by_mode = {mode: math.fsum(values) for mode, values in overlaps.items()}
    total_ms = math.fsum(event["duration_ms"] for event in collections)
    summary = {
        "collection_count": len(collections),
        "collections_by_generation": {
            str(generation): sum(event["generation"] == generation for event in collections)
            for generation in range(3)
        },
        "total_gc_ms": total_ms,
        "online_gc_overlap_ms_by_mode": by_mode,
        "online_passes_with_gc_by_mode": counts,
        "outside_online_gc_ms": max(0.0, total_ms - math.fsum(by_mode.values())),
    }
    return collections, summary


def profile_batch_gc(
    loader, indices, model, entity, *, repeats=20, warmup=5, seed=42, device="cpu"
):
    """Run natural automatic GC and retain its complete cost in online timings."""
    enabled, thresholds, callbacks = gc.isenabled(), gc.get_threshold(), list(gc.callbacks)
    executor, fingerprint = pipeline_profile.execute_pipeline, pipeline_profile.sampler_fingerprint
    telemetry = {
        "format_version": FORMAT_VERSION,
        "policy": "automatic",
        "protocol": {"repeats": repeats, "warmup": warmup, "seed": seed},
        "gc_enabled_before": enabled,
        "gc_enabled_during_case": True,
        "thresholds_before": list(thresholds),
        "callback_count_before": len(callbacks),
        "observer": "minimal GC callback; timestamp tuple retained per event; no cost subtracted",
        "limitation": "Observer allocations can move GC cadence; callbacks are not zero overhead",
        "state_snapshot_scope": (
            "counter/stat snapshots adjacent to callback attachment and detachment; "
            "event totals cover the observed case only, excluding later observer processing, "
            "reporting, and gaps between cases"
        ),
    }
    raw_events, passes = [], []
    active_pass, call_index = -1, 0
    started_ns = finished_ns = None
    started_at = finished_at = None

    def on_gc(phase, info):
        collected = info.get("collected", 0) if phase == "start" else info["collected"]
        uncollectable = info.get("uncollectable", 0) if phase == "start" else info["uncollectable"]
        raw_events.append(
            (
                time.perf_counter_ns(),
                phase,
                info["generation"],
                collected,
                uncollectable,
                active_pass,
            )
        )

    def execute(
        observed_loader,
        observed_indices,
        observed_model,
        observed_entity,
        mode,
        observed_device,
        *,
        stages=False,
    ):
        nonlocal active_pass, call_index
        index = call_index
        call_index += 1
        started = time.perf_counter_ns()
        active_pass = index
        try:
            result, bounds = _execute_pipeline(
                observed_loader,
                observed_indices,
                observed_model,
                observed_entity,
                mode,
                observed_device,
                stages=stages,
            )
        finally:
            active_pass = -1
        finished = time.perf_counter_ns()
        initial = index < warmup * len(MODES)
        passes.append(
            {
                **bounds,
                "pass_id": index,
                "started_ns": started,
                "finished_ns": finished,
                "pass_kind": "warmup" if initial else ("stage_replay" if stages else "online"),
                "mode": mode,
                "repeat": None if initial else (index - warmup * len(MODES)) // (2 * len(MODES)),
                "warmup_index": index // len(MODES) if initial else None,
            }
        )
        return result

    try:
        pipeline_profile.execute_pipeline = execute
        pipeline_profile.sampler_fingerprint = acyclic_sampler_fingerprint
        gc.enable()
        telemetry["count_before"] = list(gc.get_count())
        telemetry["stats_before"] = gc.get_stats()
        started_at = datetime.now(timezone.utc).isoformat()
        started_ns = time.perf_counter_ns()
        gc.callbacks.append(on_gc)
        profile = pipeline_profile.profile_batch(
            loader, indices, model, entity, repeats=repeats, warmup=warmup, seed=seed, device=device
        )
    except BaseException as error:
        error.gc_measurement_trace = {**telemetry, "raw_events": raw_events, "passes": passes}
        raise
    finally:
        gc.callbacks[:] = callbacks
        finished_ns = time.perf_counter_ns()
        finished_at = datetime.now(timezone.utc).isoformat()
        count_after = list(gc.get_count())
        stats_after = gc.get_stats()
        pipeline_profile.execute_pipeline = executor
        pipeline_profile.sampler_fingerprint = fingerprint
        gc.set_threshold(*thresholds)
        if enabled:
            gc.enable()
        else:
            gc.disable()
    processing_started = time.perf_counter_ns()
    telemetry.update(
        {
            "gc_enabled_after": gc.isenabled(),
            "thresholds_after": list(gc.get_threshold()),
            "count_after": count_after,
            "stats_after": stats_after,
            "callback_count_after": len(gc.callbacks),
            "callbacks_restored": gc.callbacks == callbacks,
            "patches_restored": pipeline_profile.execute_pipeline is executor
            and pipeline_profile.sampler_fingerprint is fingerprint,
            "passes": passes,
            "events": [
                dict(
                    zip(
                        ("at_ns", "phase", "generation", "collected", "uncollectable", "pass_id"),
                        event,
                        strict=True,
                    )
                )
                for event in raw_events
            ],
        }
    )
    telemetry["collections"], telemetry["summary"] = _summarize(telemetry["events"], passes)
    telemetry["collection_window"] = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "profile_wall_seconds": (finished_ns - started_ns) / 1e9,
        "observer_processing_seconds": 0.0,
        "scope": (
            "whole paired case: warmups, online trials, audits, stage replays and callbacks; "
            "observer processing recorded separately"
        ),
    }
    profile["gc_telemetry"] = telemetry
    validate_gc_telemetry(profile)
    telemetry["collection_window"]["observer_processing_seconds"] = (
        time.perf_counter_ns() - processing_started
    ) / 1e9
    return profile


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid {name}")
    return value


def _real(value, name, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(f"Invalid {name}")
    return value


def validate_gc_telemetry(record):
    """Recompute exact GC intersections and reject inconsistent natural-GC data."""
    try:
        data = record["gc_telemetry"]
        if (
            _integer(data["format_version"], "format version") != FORMAT_VERSION
            or data["policy"] != "automatic"
        ):
            raise ValueError("Unsupported GC telemetry protocol")
        protocol = data["protocol"]
        repeats = _integer(protocol["repeats"], "repeats", 3)
        warmup = _integer(protocol["warmup"], "warmup")
        seed = _integer(protocol["seed"], "seed")
        if len(record["trials"]) != repeats:
            raise ValueError("GC telemetry repeat count mismatch")
        expected = [("warmup", None, mode, index) for index in range(warmup) for mode in MODES]
        for repeat, trial in enumerate(record["trials"]):
            if _integer(trial["sampler_seed"], "sampler seed") != seed + repeat:
                raise ValueError("GC telemetry sampler seed mismatch")
            order = trial["execution_order"]
            if sorted(order) != sorted(MODES):
                raise ValueError("GC telemetry mode order mismatch")
            expected.extend(
                (kind, repeat, mode, None) for mode in order for kind in ("online", "stage_replay")
            )
        passes = data["passes"]
        if len(passes) != len(expected):
            raise ValueError("GC telemetry pass count mismatch")
        window = data["collection_window"]
        start = _integer(window["started_ns"], "collection start")
        end = _integer(window["finished_ns"], "collection finish", start)
        if _real(window["profile_wall_seconds"], "collection wall seconds") != (end - start) / 1e9:
            raise ValueError("GC collection wall duration mismatch")
        _real(window["observer_processing_seconds"], "observer processing duration")
        wall_start = datetime.fromisoformat(window["started_at_utc"])
        wall_end = datetime.fromisoformat(window["finished_at_utc"])
        if wall_start.tzinfo is None or wall_end.tzinfo is None or wall_start > wall_end:
            raise ValueError("Invalid GC collection wall-clock window")
        if abs((wall_end - wall_start).total_seconds() - (end - start) / 1e9) > 1:
            raise ValueError("GC wall-clock and monotonic collection windows differ")
        previous_end = start
        for index, (item, expected_pass) in enumerate(zip(passes, expected, strict=True)):
            if (
                _integer(item["pass_id"], "pass id") != index
                or (item["pass_kind"], item["repeat"], item["mode"], item["warmup_index"])
                != expected_pass
            ):
                raise ValueError("GC telemetry pass identity mismatch")
            for name in ("repeat", "warmup_index"):
                if item[name] is not None:
                    _integer(item[name], name)
            outer_start = _integer(item["started_ns"], "pass start", previous_end)
            timed_start = _integer(item["timed_started_ns"], "timer start", outer_start)
            timed_end = _integer(item["timed_finished_ns"], "timer finish", timed_start + 1)
            outer_end = _integer(item["finished_ns"], "pass finish", timed_end)
            if outer_end > end:
                raise ValueError("GC pipeline pass outside collection window")
            previous_end = outer_end
            if (
                _real(item["elapsed_ms"], "elapsed ms", positive=True)
                != (timed_end - timed_start) / 1e6
            ):
                raise ValueError("GC telemetry exact timer duration mismatch")
            expected_reads = 14 if item["pass_kind"] == "stage_replay" else 2
            if _integer(item["timer_read_count"], "timer read count") != expected_reads:
                raise ValueError("GC executor timer structure mismatch")
            if item["pass_kind"] == "online" and (
                item["elapsed_ms"]
                != _real(
                    record["trials"][item["repeat"]]["pipeline_ms"][item["mode"]],
                    "online pipeline ms",
                    positive=True,
                )
            ):
                raise ValueError("GC timer differs from online pipeline timing")
        previous = start
        for event in data["events"]:
            timestamp = _integer(event["at_ns"], "GC event timestamp", previous)
            if timestamp > end or event["phase"] not in ("start", "stop"):
                raise ValueError("GC event outside collection window or invalid phase")
            previous = timestamp
            if _integer(event["generation"], "GC generation") > 2:
                raise ValueError("Invalid GC generation")
            for name in ("collected", "uncollectable"):
                _integer(event[name], "GC " + name)
            pass_id = _integer(event["pass_id"], "GC event pass", -1)
            if pass_id >= len(passes):
                raise ValueError("GC event references unknown pipeline pass")
            if (
                pass_id >= 0
                and not passes[pass_id]["started_ns"] <= timestamp <= passes[pass_id]["finished_ns"]
            ):
                raise ValueError("GC event lies outside its labelled pipeline pass")
        collections, summary = _summarize(data["events"], passes)
        actual = json.dumps([data["collections"], data["summary"]], sort_keys=True, allow_nan=False)
        expected_summary = json.dumps([collections, summary], sort_keys=True, allow_nan=False)
        if actual != expected_summary:
            raise ValueError("GC collection or overlap summary mismatch")
        for name in (
            "gc_enabled_before",
            "gc_enabled_after",
            "gc_enabled_during_case",
            "callbacks_restored",
            "patches_restored",
        ):
            if not isinstance(data[name], bool):
                raise ValueError("Invalid GC state flag")
        if (
            not data["gc_enabled_during_case"]
            or not data["callbacks_restored"]
            or not data["patches_restored"]
        ):
            raise ValueError("GC observation did not restore state or use automatic collection")
        if data["gc_enabled_before"] != data["gc_enabled_after"]:
            raise ValueError("GC enabled state was not restored")
        if data["thresholds_before"] != data["thresholds_after"]:
            raise ValueError("GC thresholds were not restored")
        for name in ("thresholds_before", "thresholds_after", "count_before", "count_after"):
            if len(data[name]) != 3:
                raise ValueError("Invalid GC state counters")
            for value in data[name]:
                _integer(value, name)
        for name in ("stats_before", "stats_after"):
            if len(data[name]) != 3:
                raise ValueError("Invalid GC generation statistics")
            for generation in data[name]:
                for field in ("collections", "collected", "uncollectable"):
                    _integer(generation[field], "GC statistics " + field)
        for before, after in zip(data["stats_before"], data["stats_after"], strict=True):
            if any(
                after[field] < before[field]
                for field in ("collections", "collected", "uncollectable")
            ):
                raise ValueError("Cumulative GC statistics decreased")
        for name in ("callback_count_before", "callback_count_after"):
            _integer(data[name], name)
        if data["callback_count_before"] != data["callback_count_after"]:
            raise ValueError("GC callback count was not restored")
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Malformed GC telemetry") from error
