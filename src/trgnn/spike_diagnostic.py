"""Observe a reproducible inference pause without altering frozen benchmark code.

These timings include diagnostic callbacks and wrappers. They are not replacement
benchmark results. Host spans around asynchronous CUDA calls are not kernel times.
The process-global instrumentation requires a single serial caller.
"""

import argparse
import functools
import gc
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from trgnn import pipeline_profile
from trgnn.artifacts import environment, write_json
from trgnn.execution import MODES
from trgnn.gpu_telemetry import NvidiaSmiMonitor, gpu_identity
from trgnn.trained_checkpoint import load_trained_workload, replay_saved_validation
from trgnn.trained_profile import (
    artifact_path,
    batch_plan,
    canonical_sha256,
    make_benchmark_contract,
    model_sha256,
    validate_artifact,
)
from trgnn.training_pilot import file_sha256, make_query_loader

SOURCE_PROFILE_SHA256 = "0db16833c33b6d723779cd8cb39ef62f0abf2a0f63290a6bcf95875cd693c9c3"


class _PipelineClock:
    """Observe only the frozen executor's timer reads, leaving other clocks alone."""

    def __init__(self, clock):
        self.clock = clock
        self.reads = []

    def perf_counter_ns(self):
        value = self.clock.perf_counter_ns()
        self.reads.append(value)
        return value

    def __getattr__(self, name):
        return getattr(self.clock, name)


def _update_fingerprint(value, digest):
    """Match the original serialization using global recursion, without a closure."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Expected a CPU sampler output")
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.contiguous().numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _update_fingerprint(key, digest)
            _update_fingerprint(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(str((type(value).__name__, len(value))).encode())
        for item in value:
            _update_fingerprint(item, digest)
    else:
        digest.update(repr(value).encode())
    digest.update(b"\0")


def acyclic_sampler_fingerprint(sample):
    """Byte-identical diagnostic alternative to the recursive nested visitor."""
    digest = hashlib.sha256()
    for field in ("node", "row", "col", "edge", "batch", "num_sampled_nodes", "num_sampled_edges"):
        _update_fingerprint(field, digest)
        _update_fingerprint(getattr(sample, field), digest)
    _update_fingerprint(sample.metadata, digest)
    return digest.hexdigest()


def _gc_overlap(trace):
    """Intersect GC events with the exact frozen online timer, not wrapper bounds."""
    pending, collections = {}, []
    for event in trace["gc_events"]:
        generation = event["generation"]
        if event["phase"] == "start":
            pending[generation] = event
        elif generation in pending:
            start = pending.pop(generation)
            collections.append(
                {
                    "started_ns": start["at_ns"],
                    "finished_ns": event["at_ns"],
                    "duration_ms": (event["at_ns"] - start["at_ns"]) / 1e6,
                    "generation": generation,
                    "collected": event["collected"],
                    "uncollectable": event["uncollectable"],
                    "started_context": {
                        key: start.get(key) for key in ("pass_kind", "mode", "repeat", "stage")
                    },
                }
            )
    trace["gc_collections"] = collections
    overlaps = []
    for item in trace["pipeline_passes"]:
        if item["pass_kind"] != "online" or item["timed_finished_ns"] is None:
            continue
        start, end = item["timed_started_ns"], item["timed_finished_ns"]
        matching = [
            event
            for event in collections
            if event["started_ns"] < end and event["finished_ns"] > start
        ]
        duration = sum(
            max(0, min(end, event["finished_ns"]) - max(start, event["started_ns"]))
            for event in matching
        )
        overlaps.append(
            {
                "repeat": item["repeat"],
                "mode": item["mode"],
                "elapsed_ms": item["elapsed_ms"],
                "gc_overlap_ms": duration / 1e6,
                "gc_generations": [event["generation"] for event in matching],
                "timed_started_ns": start,
                "timed_finished_ns": end,
            }
        )
    trace["online_gc_overlap"] = overlaps
    long_passes = []
    for item in overlaps:
        if item["elapsed_ms"] <= 100:
            continue
        start, end = item["timed_started_ns"], item["timed_finished_ns"]
        spans = [
            {
                "operation": span["operation"],
                "host_ms": (span["finished_ns"] - span["started_ns"]) / 1e6,
            }
            for span in trace["host_spans"]
            if span["started_ns"] < end and span["finished_ns"] > start
        ]
        long_passes.append({**item, "overlapping_host_spans": spans})
    trace["summary"] = {
        "gc_collections": len(collections),
        "gc_total_ms": sum(event["duration_ms"] for event in collections),
        "online_gc_overlap_ms": sum(item["gc_overlap_ms"] for item in overlaps),
        "top_gc_pauses": sorted(collections, key=lambda event: event["duration_ms"], reverse=True)[
            :5
        ],
        "online_passes_over_100_ms": long_passes,
        "max_online_elapsed_ms": max((item["elapsed_ms"] for item in overlaps), default=None),
    }


def observe_profile_batch(
    loader,
    indices,
    model,
    entity,
    *,
    repeats=20,
    warmup=5,
    seed=42,
    device="cpu",
    gc_policy="automatic",
    fingerprint_policy="original",
):
    """Return the ordinary paired profile plus GC events and host-call spans.

    ``collect_between_passes`` disables automatic cyclic collection in this case
    and runs full collections immediately before each pipeline pass, outside its
    timed function. It does not disable reference-count reclamation. Both policies
    retain all original sampler, correctness, stage-replay and mode-order logic.
    All modified methods and GC settings are restored even on interruption.
    """
    if gc_policy not in ("automatic", "collect_between_passes"):
        raise ValueError("Unknown GC policy")
    if fingerprint_policy not in ("original", "acyclic"):
        raise ValueError("Unknown fingerprint policy")
    previous_enabled = gc.isenabled()
    previous_thresholds = gc.get_threshold()
    previous_callbacks = list(gc.callbacks)
    trace = {
        "gc_policy": gc_policy,
        "fingerprint_policy": fingerprint_policy,
        "gc_enabled_before": previous_enabled,
        "gc_thresholds_before": list(previous_thresholds),
        "gc_count_before": list(gc.get_count()),
        "gc_stats_before": gc.get_stats(),
        "clock": "time.perf_counter_ns; monotonic host clock",
        "host_span_scope": "host call duration, including existing waits; not CUDA kernel time",
        "timing_scope": "instrumented diagnostic; not comparable to benchmark latency estimates",
        "gc_events": [],
        "host_spans": [],
        "pipeline_passes": [],
    }
    context = {"pass_kind": "audits_or_between_passes", "mode": None, "repeat": None}
    replacements = []
    call_index = 0

    def on_gc(phase, info):
        trace["gc_events"].append(
            {
                "phase": phase,
                "at_ns": time.perf_counter_ns(),
                "generation": info.get("generation"),
                "collected": info.get("collected"),
                "uncollectable": info.get("uncollectable"),
                **context,
            }
        )

    def replace(owner, name, value):
        had_own_attribute = name in vars(owner)
        original = getattr(owner, name)
        replacements.append((owner, name, original, had_own_attribute))
        setattr(owner, name, value)

    def observed(operation, name):
        @functools.wraps(operation)
        def wrapped(*args, **kwargs):
            previous_stage = context.get("stage")
            context["stage"] = name
            started = time.perf_counter_ns()
            try:
                return operation(*args, **kwargs)
            finally:
                finished = time.perf_counter_ns()
                trace["host_spans"].append(
                    {"operation": name, "started_ns": started, "finished_ns": finished, **context}
                )
                if previous_stage is None:
                    context.pop("stage", None)
                else:
                    context["stage"] = previous_stage

        return wrapped

    original_pipeline = pipeline_profile.execute_pipeline

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
        nonlocal call_index
        position = call_index
        call_index += 1
        warmup_passes = warmup * len(MODES)
        label = {
            "pass_kind": "warmup"
            if position < warmup_passes
            else ("stage_replay" if stages else "online"),
            "mode": mode,
            "repeat": None
            if position < warmup_passes
            else ((position - warmup_passes) // (2 * len(MODES))),
            "warmup_index": position // len(MODES) if position < warmup_passes else None,
        }
        previous = dict(context)
        context.clear()
        context.update(label)
        try:
            if gc_policy == "collect_between_passes":
                context["pass_kind"] = "between_pass_collection"
                gc.collect()
                context.update(label)
            started = time.perf_counter_ns()
            original_clock = pipeline_profile.time
            clock = _PipelineClock(original_clock)
            pipeline_profile.time = clock
            result = None
            try:
                result = original_pipeline(
                    observed_loader,
                    observed_indices,
                    observed_model,
                    observed_entity,
                    mode,
                    observed_device,
                    stages=stages,
                )
                expected_reads = 2 + (2 * len(pipeline_profile.STAGES) if stages else 0)
                if len(clock.reads) != expected_reads:
                    raise AssertionError("Frozen executor timer-read structure changed")
                if result.elapsed_ms != (clock.reads[-1] - clock.reads[0]) / 1e6:
                    raise AssertionError("Diagnostic clock does not match the recorded timer")
                return result
            finally:
                pipeline_profile.time = original_clock
                trace["pipeline_passes"].append(
                    {
                        "started_ns": started,
                        "finished_ns": time.perf_counter_ns(),
                        "timed_started_ns": clock.reads[0] if clock.reads else None,
                        "timed_finished_ns": clock.reads[-1] if result is not None else None,
                        "timer_read_count": len(clock.reads),
                        "elapsed_ms": result.elapsed_ms if result is not None else None,
                        **label,
                    }
                )
        finally:
            context.clear()
            context.update(previous)

    try:
        gc.callbacks.append(on_gc)
        if gc_policy == "automatic":
            gc.enable()
        else:
            gc.disable()
        replace(pipeline_profile, "execute_pipeline", execute)
        replace(
            pipeline_profile, "synchronize", observed(pipeline_profile.synchronize, "synchronize")
        )
        if fingerprint_policy == "acyclic":
            replace(pipeline_profile, "sampler_fingerprint", acyclic_sampler_fingerprint)
        replace(
            loader.node_sampler,
            "sample_from_nodes",
            observed(loader.node_sampler.sample_from_nodes, "sampling"),
        )
        replace(loader, "filter_fn", observed(loader.filter_fn, "feature_gather"))
        batch_type = type(loader.data)
        replace(batch_type, "to", observed(batch_type.to, "transfer"))
        for name in ("encode", "message_passing", "readout"):
            replace(model, name, observed(getattr(model, name), name))
        profile = pipeline_profile.profile_batch(
            loader, indices, model, entity, repeats=repeats, warmup=warmup, seed=seed, device=device
        )
        gc.callbacks.remove(on_gc)
        context["pass_kind"] = "trace_analysis"
        trace["gc_count_after"] = list(gc.get_count())
        trace["gc_stats_after"] = gc.get_stats()
        _gc_overlap(trace)
        return profile, trace
    except BaseException as error:
        if on_gc in gc.callbacks:
            gc.callbacks.remove(on_gc)
        context["pass_kind"] = "trace_analysis"
        _gc_overlap(trace)
        error.diagnostic_trace = trace
        raise
    finally:
        for owner, name, original, had_own_attribute in reversed(replacements):
            if had_own_attribute:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        gc.callbacks[:] = previous_callbacks
        gc.set_threshold(*previous_thresholds)
        if previous_enabled:
            gc.enable()
        else:
            gc.disable()


def _source_record(source, source_path, planned, repeats):
    key = planned["batch_size"], planned["batch_index"]
    matches = [
        item for item in source["artifacts"] if (item["batch_size"], item["batch_index"]) == key
    ]
    if len(matches) != 1:
        raise ValueError("Source profile does not uniquely index the requested batch")
    path = artifact_path(source_path.parent / "batches", planned)
    if file_sha256(path) != matches[0]["sha256"]:
        raise ValueError("Source sidecar bytes differ from the archived profile index")
    return validate_artifact(
        json.loads(path.read_text()), planned, source["benchmark_contract_sha256"], repeats
    )


def _compare_source(profile, archived):
    for actual, expected in zip(profile["trials"], archived["trials"], strict=True):
        for name in (
            "sampler_seed",
            "execution_order",
            "sampler_sha256",
            "sampled_counts",
            "seed_times_unix_seconds",
            "visibility_audit",
        ):
            if actual[name] != expected[name]:
                raise ValueError(f"Diagnostic differs from archived paired inputs: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rel-stack"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-indices", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--replay-prefix", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/spike-diagnostic.json"))
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args()
    if any(index < 0 for index in args.batch_indices) or len(set(args.batch_indices)) != len(
        args.batch_indices
    ):
        parser.error("Use distinct nonnegative batch indices")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA device is unavailable")
    return args


def _guard_output_paths(args, directory):
    """Reject historical inputs and nonempty destinations before any write."""
    output, destination = args.output.resolve(), directory.resolve()
    if destination.is_relative_to(output):
        raise ValueError("Diagnostic output collides with its artifact directory")
    protected = [
        args.checkpoint.resolve(),
        args.training_report.resolve(),
        args.source_profile.resolve(),
    ]
    historical = [args.training_report.resolve().parent, args.source_profile.resolve().parent]
    code = Path(__file__).resolve().parent
    if output.exists():
        raise ValueError("Diagnostic output already exists; use a fresh run directory")
    if output in protected or any(output.is_relative_to(root) for root in (*historical, code)):
        raise ValueError("Diagnostic output would overwrite a protected historical input")
    if any(path.is_relative_to(destination) for path in protected):
        raise ValueError("Diagnostic artifact directory contains a protected input")
    if any(destination.is_relative_to(root) for root in (*historical, code)):
        raise ValueError("Diagnostic artifact directory overlaps protected history or source code")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Use an empty diagnostic artifacts directory for a fresh attempt")


def _check_source_contract(workload, args, source):
    expected = source["benchmark_contract"]
    protocol = expected["protocol"]
    settings = SimpleNamespace(
        device=args.device,
        batch_sizes=protocol["batch_sizes"],
        repeats=protocol["repeats"],
        warmup=protocol["warmup"],
        seed=protocol["seed"],
        bootstrap_samples=protocol["bootstrap_samples"],
    )
    actual = make_benchmark_contract(workload, settings, gpu_identity(args.device))
    differences = {
        name: {"source": value, "diagnostic": actual.get(name)}
        for name, value in expected.items()
        if actual.get(name) != value
    }
    required = [name for name in expected if name != "hardware"]
    failed = [name for name in required if actual.get(name) != expected[name]]
    return actual, differences, failed


def main():
    args = parse_args()
    directory = args.artifacts_dir or args.output.parent / "cases"
    _guard_output_paths(args, directory)
    report = {
        "status": "running",
        "scope": "instrumented_512_pause_diagnostic",
        "session_id": uuid.uuid4().hex,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "module_sha256": file_sha256(Path(__file__)),
        "source_profile_sha256": None,
        "cases": [],
        "causal_status": "unresolved; inspect observed GC overlap and controlled conditions",
        "limitations": [
            "Callbacks and wrappers perturb allocation and timing; not replacement benchmarks.",
            "Default short replay does not reproduce the original preceding allocation history.",
            "Prefix replay preserves workload order; instrumentation can move GC cadence.",
            "Controls follow the automatic phase; phase order and heap history can confound them.",
            "Collect-between-passes moves cyclic collection; reference counting remains.",
            "Host spans measure enqueue work and existing waits, not GPU kernel times.",
            "Absence of a pause in a short control does not establish its cause.",
        ],
    }
    write_json(args.output, report)
    started = time.perf_counter()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        source = json.loads(args.source_profile.read_text())
        report["source_profile_sha256"] = file_sha256(args.source_profile)
        if report["source_profile_sha256"] != SOURCE_PROFILE_SHA256:
            raise ValueError("Diagnostic source profile differs from the reviewed second run")
        contract = source["benchmark_contract"]
        if (
            source.get("status") != "passed"
            or canonical_sha256(contract) != source["benchmark_contract_sha256"]
        ):
            raise ValueError("Source profile is incomplete or its benchmark contract is altered")
        protocol = contract["protocol"]
        if protocol["batch_sizes"] != [32, 128, 512]:
            raise ValueError("Diagnostic expects the original 32/128/512 workload order")
        repeats, warmup, seed = protocol["repeats"], protocol["warmup"], protocol["seed"]
        torch.set_num_threads(contract["hardware"]["cpu_threads"])
        torch.manual_seed(seed)
        report["environment"] = environment()
        workload = load_trained_workload(
            args.checkpoint,
            args.training_report,
            args.data_dir,
            device=args.device,
            download=args.download,
        )
        actual_contract, differences, failed = _check_source_contract(workload, args, source)
        report["diagnostic_benchmark_contract"] = actual_contract
        report["source_contract_differences"] = differences
        if failed:
            raise ValueError("Diagnostic source contract mismatch: " + ", ".join(failed))
        for name, value in (
            ("checkpoint_sha256", workload.checkpoint_metadata["sha256"]),
            ("training_report_sha256", workload.checkpoint_metadata["source_report_sha256"]),
        ):
            if contract[name] != value:
                raise ValueError(f"Diagnostic source mismatch: {name}")
        if protocol["query_count"] != len(workload.val_queries):
            raise ValueError("Diagnostic source validation cohort size mismatch")
        report["validation_replay"] = replay_saved_validation(workload)
        original_hash = model_sha256(workload.model)
        if original_hash != source["model_state_sha256_before"]:
            raise ValueError("Diagnostic weights differ from the original benchmark")
        report["model_state_sha256_before"] = original_hash
        plans = list(batch_plan(len(workload.val_queries), [32, 128, 512], repeats, seed))
        for plan in plans:
            plan["model_state_sha256"] = original_hash
        targets = [
            p for p in plans if p["batch_size"] == 512 and p["batch_index"] in args.batch_indices
        ]
        if len(targets) != len(args.batch_indices):
            raise ValueError("Requested diagnostic batch index is outside the source cohort")
        automatic = (
            [
                p
                for p in plans
                if p["batch_size"] != 512 or p["batch_index"] <= max(args.batch_indices)
            ]
            if args.replay_prefix
            else targets
        )
        phases = (
            ("original_automatic", automatic, "automatic", "original"),
            ("original_collect_between_passes", targets, "collect_between_passes", "original"),
            ("acyclic_automatic", targets, "automatic", "acyclic"),
        )
        report["expected_cases"] = sum(len(items) for _, items, _, _ in phases)
        report["protocol"] = protocol
        monitor = NvidiaSmiMonitor(args.output.parent, report["session_id"], device=args.device)
        with monitor:
            report["telemetry"] = monitor.metadata
            for phase, items, gc_policy, fingerprint_policy in phases:
                loader, loader_size = None, None
                for plan in items:
                    archived = _source_record(source, args.source_profile, plan, repeats)
                    if loader_size != plan["batch_size"]:
                        loader_size = plan["batch_size"]
                        loader = make_query_loader(
                            workload.graph,
                            workload.val_queries,
                            workload.task,
                            workload.contract["settings"]["fanouts"],
                            loader_size,
                        )
                    torch.manual_seed(plan["sampling_seed"])
                    wall_start = datetime.now(timezone.utc).isoformat()
                    profile, trace = None, None
                    try:
                        profile, trace = observe_profile_batch(
                            loader,
                            plan["query_indices"],
                            workload.model,
                            workload.task.entity_table,
                            repeats=repeats,
                            warmup=warmup,
                            seed=plan["sampling_seed"],
                            device=args.device,
                            gc_policy=gc_policy,
                            fingerprint_policy=fingerprint_policy,
                        )
                        _compare_source(profile, archived)
                        if model_sha256(workload.model) != original_hash:
                            raise AssertionError("Diagnostic changed frozen weights or buffers")
                    except BaseException as error:
                        partial = (
                            trace if trace is not None else getattr(error, "diagnostic_trace", None)
                        )
                        if partial is not None:
                            path = directory / (
                                f"{phase}-batch-{loader_size:06d}-"
                                f"{plan['batch_index']:06d}-partial-trace.json"
                            )
                            write_json(
                                path, {**partial, **plan, "phase": phase, "status": "partial"}
                            )
                            report["failed_case"] = {
                                "trace_path": str(path),
                                "trace_sha256": file_sha256(path),
                            }
                            if profile is not None:
                                failed_profile = path.with_name(
                                    path.name.replace("trace", "profile")
                                )
                                write_json(failed_profile, {**profile, **plan, "status": "failed"})
                                report["failed_case"].update(
                                    {
                                        "profile_path": str(failed_profile),
                                        "profile_sha256": file_sha256(failed_profile),
                                    }
                                )
                        raise
                    stem = f"{phase}-batch-{plan['batch_size']:06d}-{plan['batch_index']:06d}"
                    profile_path, trace_path = (
                        directory / f"{stem}-profile.json",
                        directory / f"{stem}-trace.json",
                    )
                    write_json(
                        profile_path, {**profile, **plan, "status": "passed", "phase": phase}
                    )
                    write_json(
                        trace_path,
                        {
                            **trace,
                            **plan,
                            "phase": phase,
                            "started_at_utc": wall_start,
                            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    report["cases"].append(
                        {
                            "phase": phase,
                            "batch_size": loader_size,
                            "batch_index": plan["batch_index"],
                            "profile_path": str(profile_path),
                            "profile_sha256": file_sha256(profile_path),
                            "trace_path": str(trace_path),
                            "trace_sha256": file_sha256(trace_path),
                            "archived_input_match": True,
                            "model_state_match": True,
                            "summary": trace["summary"],
                        }
                    )
                    report["host_memory"] = pipeline_profile.host_memory()
                    write_json(args.output, report)
                    print(
                        f"{phase}: batch {loader_size}/{plan['batch_index']}; "
                        f"completed {len(report['cases'])}/{report['expected_cases']}",
                        flush=True,
                    )
                    del profile, trace
        report["model_state_sha256_after"] = model_sha256(workload.model)
        if report["model_state_sha256_after"] != original_hash:
            raise AssertionError("Diagnostic changed frozen model state after the final case")
        report["checkpoint_sha256_after"] = file_sha256(args.checkpoint)
        if report["checkpoint_sha256_after"] != contract["checkpoint_sha256"]:
            raise AssertionError("Diagnostic checkpoint bytes changed during the attempt")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["wall_seconds_this_attempt"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, report)


if __name__ == "__main__":
    main()
