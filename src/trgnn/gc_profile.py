"""GC-aware paired inference with natural automatic collection and fixed inputs.

This version is separate from the immutable notebook 06 benchmark. The online
timer retains naturally occurring collection, and lightweight GC observation
overhead is present. Collection-period wall time is reported separately rather
than added to online latency or used as an inference-throughput denominator.
"""

import argparse
import gc
import json
import math
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from trgnn import training
from trgnn.artifacts import environment, write_json
from trgnn.execution import MODES
from trgnn.gc_measurement import (
    acyclic_sampler_fingerprint,
    profile_batch_gc,
    validate_gc_telemetry,
)
from trgnn.gpu_telemetry import NvidiaSmiMonitor, gpu_identity
from trgnn.inference_statistics import summarize_profiles
from trgnn.pipeline_profile import host_memory
from trgnn.trained_checkpoint import load_trained_workload, replay_saved_validation
from trgnn.trained_profile import (
    ATOL,
    RTOL,
    artifact_path,
    batch_plan,
    canonical_sha256,
    make_benchmark_contract,
    model_sha256,
)
from trgnn.trained_profile import (
    validate_artifact as validate_legacy_artifact,
)
from trgnn.training_pilot import archive_previous_report, file_sha256, make_query_loader

FORMAT_VERSION = GC_VERSION = 2
SOURCE_PROFILE_SHA256 = "0db16833c33b6d723779cd8cb39ef62f0abf2a0f63290a6bcf95875cd693c9c3"
GC_SOURCE_FILES = ("gc_profile.py", "gc_measurement.py")
INPUT_FIELDS = (
    "sampler_seed",
    "execution_order",
    "sampler_sha256",
    "sampled_counts",
    "seed_times_unix_seconds",
    "visibility_audit",
)
LATENCY_SCOPE = "whole online pipeline with lightweight GC telemetry overhead present"


@dataclass(frozen=True)
class SourceRecordRef:
    """A small verified reference; retain no historical trial heap while measuring."""

    path: Path
    planned: dict
    contract_sha: str
    repeats: int
    sha256: str

    def load(self):
        if file_sha256(self.path) != self.sha256:
            raise ValueError("Source batch bytes differ from the pinned index")
        return validate_legacy_artifact(
            json.loads(self.path.read_text()), self.planned, self.contract_sha, self.repeats
        )


def _source_record(reference):
    return reference.load() if isinstance(reference, SourceRecordRef) else reference


@contextmanager
def acyclic_replay_audit():
    """Use identical hash bytes during replay without leaving old helper cycles."""
    original = training.sampler_fingerprint
    training.sampler_fingerprint = acyclic_sampler_fingerprint
    try:
        yield
    finally:
        training.sampler_fingerprint = original


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rel-stack"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/gc-aware-profile.json"))
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        any(size < 1 for size in args.batch_sizes)
        or len(set(args.batch_sizes)) != len(args.batch_sizes)
        or args.repeats < 3
        or args.warmup < 0
        or not 0 <= args.seed < 2**32
        or args.bootstrap_samples < 1
    ):
        parser.error("Use distinct positive sizes, >=3 repeats and nonnegative warmup/seed")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; the reviewed production source requires a T4")
    if args.artifacts_dir is None:
        args.artifacts_dir = args.output.parent / "batches"
    return args


def guard_output_paths(args):
    """Protect historical evidence before creating or archiving any output."""
    output, directory = args.output.resolve(), args.artifacts_dir.resolve()
    protected = [
        args.checkpoint.resolve(),
        args.training_report.resolve(),
        args.source_profile.resolve(),
    ]
    if any(not path.is_file() for path in protected):
        raise ValueError("Checkpoint, training report and source profile must be existing files")
    historical = [args.training_report.resolve().parent, args.source_profile.resolve().parent]
    code = Path(__file__).resolve().parent
    if output == directory or output.is_relative_to(directory) or directory.is_relative_to(output):
        raise ValueError("Summary and artifact directory must be separate paths")
    if output in protected or any(output.is_relative_to(path) for path in (*historical, code)):
        raise ValueError("Output would overwrite protected source/history")
    if any(path.is_relative_to(directory) for path in protected):
        raise ValueError("Artifact directory contains protected source/history")
    if any(directory.is_relative_to(path) for path in (*historical, code)):
        raise ValueError("Artifact directory overlaps protected source/history")
    if output.exists() and not output.is_file():
        raise ValueError("Summary output must be a file")
    if directory.exists() and not directory.is_dir():
        raise ValueError("Artifact output must be a directory")
    if not args.resume and (output.exists() or (directory.exists() and any(directory.iterdir()))):
        raise FileExistsError("Measurements already exist; use --resume or a fresh directory")
    if args.resume and output.is_file():
        previous = json.loads(output.read_text())
        if previous.get("gc_version") != GC_VERSION:
            raise ValueError("Cannot resume an old or diagnostic report as GC protocol v2")


def load_source_profile(args):
    """Read the pinned original index and every source batch without mutation."""
    if file_sha256(args.source_profile) != SOURCE_PROFILE_SHA256:
        raise ValueError("Source profile differs from the reviewed notebook 06 reproduction")
    source = json.loads(args.source_profile.read_text())
    contract = source.get("benchmark_contract", {})
    if (
        source.get("status") != "passed"
        or not source.get("measurements_complete")
        or canonical_sha256(contract) != source.get("benchmark_contract_sha256")
        or source.get("model_state_sha256_before") != source.get("model_state_sha256_after")
    ):
        raise ValueError("Source profile is incomplete or its contract/model identity differs")
    protocol = contract["protocol"]
    for name in ("batch_sizes", "repeats", "warmup", "seed", "bootstrap_samples"):
        if getattr(args, name) != protocol[name]:
            raise ValueError(f"Keep the original source protocol setting: {name}")
    plans = list(batch_plan(protocol["query_count"], args.batch_sizes, args.repeats, args.seed))
    artifacts = source.get("artifacts", [])
    keys = [(item["batch_size"], item["batch_index"]) for item in artifacts]
    expected = {(item["batch_size"], item["batch_index"]) for item in plans}
    if (
        len(keys) != len(set(keys))
        or set(keys) != expected
        or source.get("completed_batches") != len(plans)
        or source.get("expected_batches") != len(plans)
    ):
        raise ValueError("Source index must cover every query batch exactly once")
    index = dict(zip(keys, artifacts, strict=True))
    records = {}
    for plan in plans:
        plan["model_state_sha256"] = source["model_state_sha256_before"]
        key = plan["batch_size"], plan["batch_index"]
        path = artifact_path(args.source_profile.parent / "batches", plan)
        reference = SourceRecordRef(
            path, plan, source["benchmark_contract_sha256"], args.repeats, index[key]["sha256"]
        )
        reference.load()
        records[key] = reference
    return source, records


def make_gc_contract(workload, args, source, identity):
    """Bind source semantics, runtime, new code and active GC settings."""
    actual = make_benchmark_contract(workload, args, identity)
    expected = source["benchmark_contract"]
    differences = {
        name: {"source": value, "current": actual.get(name)}
        for name, value in expected.items()
        if actual.get(name) != value
    }
    failed = [
        name for name in expected if name != "hardware" and actual.get(name) != expected[name]
    ]
    if failed:
        raise ValueError("Source contract mismatch: " + ", ".join(failed))
    source_hardware, hardware = expected["hardware"], actual["hardware"]
    if hardware["device"] != source_hardware["device"] or hardware.get(
        "gpu"
    ) != source_hardware.get("gpu"):
        raise ValueError("Keep the source device and GPU class/memory configuration")
    if hardware["cpu_threads"] != source_hardware["cpu_threads"]:
        raise ValueError("Keep the source CPU thread count")
    if not gc.isenabled():
        raise ValueError("Natural automatic garbage collection must already be enabled")
    if gc.get_threshold()[0] <= 0:
        raise ValueError(
            "Natural automatic garbage collection needs a positive allocation threshold"
        )
    project = Path(__file__).resolve().parent
    return {
        "format_version": FORMAT_VERSION,
        "gc_version": GC_VERSION,
        "source_profile_sha256": SOURCE_PROFILE_SHA256,
        "source_benchmark_contract_sha256": source["benchmark_contract_sha256"],
        "legacy_benchmark_contract": actual,
        "new_profile_code_sha256": {name: file_sha256(project / name) for name in GC_SOURCE_FILES},
        "gc_policy": {
            "name": "natural_automatic",
            "enabled": True,
            "thresholds": list(gc.get_threshold()),
            "fingerprint": "acyclic byte-identical sampler serialization",
            "fingerprint_scope": "measurement audits and complete saved-validation replay",
            "forced_collection": False,
            "timing": "retain all natural collections; lightweight observer overhead present",
        },
    }, differences


def verify_source_match(profile, source_record):
    """Require every paired sampled occurrence and execution order to match."""
    try:
        if profile["sampler_pairing"] != source_record["sampler_pairing"]:
            raise ValueError("Sampler-pairing contract differs from the source")
        for actual, expected in zip(profile["trials"], source_record["trials"], strict=True):
            for name in INPUT_FIELDS:
                if actual[name] != expected[name]:
                    raise ValueError(f"Paired inputs differ from the source: {name}")
    except (KeyError, TypeError) as error:
        raise ValueError("Missing or malformed archived paired inputs") from error


def validate_artifact(record, planned, contract_sha, repeats, source_record):
    if not isinstance(record, dict):
        raise ValueError("Malformed GC measurement sidecar")
    payload = {key: value for key, value in record.items() if key != "artifact_payload_sha256"}
    if canonical_sha256(payload) != record.get("artifact_payload_sha256"):
        raise ValueError("GC measurement sidecar checksum mismatch")
    if (
        record.get("format_version") != FORMAT_VERSION
        or record.get("gc_version") != GC_VERSION
        or record.get("source_profile_sha256") != SOURCE_PROFILE_SHA256
        or record.get("source_artifact_payload_sha256") != source_record["artifact_payload_sha256"]
    ):
        raise ValueError("Sidecar is not the matching GC protocol v2/source batch")
    # Reuse the strict ordinary timing/occurrence validator without changing v1.
    ordinary = {**record, "format_version": 1}
    ordinary.pop("artifact_payload_sha256")
    ordinary["artifact_payload_sha256"] = canonical_sha256(ordinary)
    validate_legacy_artifact(ordinary, planned, contract_sha, repeats)
    verify_source_match(record, source_record)
    validate_gc_telemetry(record)
    telemetry = record["gc_telemetry"]
    if (
        record.get("gc_thresholds") != planned.get("gc_thresholds")
        or telemetry["thresholds_before"] != record.get("gc_thresholds")
        or telemetry["thresholds_after"] != record.get("gc_thresholds")
        or not all(
            telemetry[name]
            for name in ("gc_enabled_before", "gc_enabled_during_case", "gc_enabled_after")
        )
    ):
        raise ValueError("GC telemetry differs from the contracted active settings")
    return record


def save_batch(directory, planned, profile, contract_sha, session_id, origin, source_record):
    record = {
        **profile,
        **planned,
        "status": "passed",
        "format_version": FORMAT_VERSION,
        "gc_version": GC_VERSION,
        "source_profile_sha256": SOURCE_PROFILE_SHA256,
        "source_artifact_payload_sha256": source_record["artifact_payload_sha256"],
        "benchmark_contract_sha256": contract_sha,
        "query_count": len(planned["query_indices"]),
        "session_id": session_id,
        "origin_environment": origin,
    }
    record["artifact_payload_sha256"] = canonical_sha256(record)
    validate_artifact(record, planned, contract_sha, len(profile["trials"]), source_record)
    path = artifact_path(directory, planned)
    if path.exists():
        raise FileExistsError("A complete GC measurement sidecar already exists")
    write_json(path, record)
    return path, record


def collect_size_summary(
    directory, plans, contract_sha, repeats, batch_size, bootstrap, seed, source_records
):
    records = []
    for plan in plans:
        if plan["batch_size"] != batch_size:
            continue
        records.append(
            validate_artifact(
                json.loads(artifact_path(directory, plan).read_text()),
                plan,
                contract_sha,
                repeats,
                _source_record(source_records[(plan["batch_size"], plan["batch_index"])]),
            )
        )
    summary = summarize_profiles(
        records, batch_size, bootstrap_samples=bootstrap, seed=seed + batch_size
    )
    sessions = sorted({record["session_id"] for record in records})
    summary.update(
        latency_scope=LATENCY_SCOPE,
        source_session_ids=sessions,
        mixed_session_measurement=len(sessions) > 1,
        gc_version=GC_VERSION,
    )
    if len(sessions) > 1:
        for comparison in summary["comparisons"].values():
            comparison["cluster_bootstrap_95_percent_ci"] = None
            comparison["limitation"] = (
                "Mixed-session aggregate is descriptive; no between-session confidence interval"
            )
        summary["bootstrap"]["scope"] = "Confidence intervals suppressed for mixed sessions"
    else:
        summary["bootstrap"]["scope"] = "Query-batch variation within one source session only"
    summary["max_absolute_logit_error"] = {
        mode: max(record["max_absolute_logit_error"][mode] for record in records) for mode in MODES
    }
    summary["max_absolute_loss_error"] = {
        mode: max(record["max_absolute_loss_error"][mode] for record in records) for mode in MODES
    }
    gc_summaries = [record["gc_telemetry"]["summary"] for record in records]
    summary["gc_telemetry"] = {
        "collection_count": sum(item["collection_count"] for item in gc_summaries),
        "collections_by_generation": {
            str(generation): sum(
                item["collections_by_generation"].get(str(generation), 0) for item in gc_summaries
            )
            for generation in range(3)
        },
        "total_gc_ms": math.fsum(item["total_gc_ms"] for item in gc_summaries),
        "outside_online_gc_ms": math.fsum(item["outside_online_gc_ms"] for item in gc_summaries),
        "online_gc_overlap_ms_by_mode": {
            mode: math.fsum(item["online_gc_overlap_ms_by_mode"][mode] for item in gc_summaries)
            for mode in MODES
        },
        "online_passes_with_gc_by_mode": {
            mode: sum(item["online_passes_with_gc_by_mode"][mode] for item in gc_summaries)
            for mode in MODES
        },
        "scope": "Observed case collections; online overlap is already included in online times",
    }
    return summary


def main():
    args = parse_args()
    guard_output_paths(args)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    session_id = uuid.uuid4().hex
    report = {
        "status": "started",
        "scope": "gc_aware_trained_real_relational_online_inference",
        "gc_version": GC_VERSION,
        "trained": False,
        "training_performed": False,
        "measurements_complete": False,
        "end_to_end": True,
        "latency_scope": LATENCY_SCOPE,
        "online_boundary": "serial sampling + gather + transfer + encoders + GNN + readout",
        "excluded_from_online_timing": [
            "setup and graph construction",
            "saved-validation replay",
            "query scheduling and worker/DataLoader overhead",
            "correctness audits, RNG seeding, reporting and telemetry setup",
        ],
        "session_id": session_id,
        "environment": environment(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source_profile_sha256": None,
        "summary_by_batch_size": {},
        "artifacts": [],
        "completed_batches": 0,
        "completed_batches_this_attempt": 0,
        "notes": [
            "Natural automatic GC stays enabled; no forced collection or managed condition.",
            "Acyclic fingerprints retain the exact archived sampled occurrence serialization.",
            "All observations and outliers remain in the predefined summed-time estimator.",
            "GC overlap is already inside online timings; never add it again.",
            "Case GC observations exclude reporting gaps; program wall retains that cost.",
            "Program wall includes warmups, audits and stage replays; it is not query throughput.",
            "Callback/timer metadata observation overhead is present; no cost is subtracted.",
            "Graph inputs retain conditional snapshot semantics and constant/time features.",
            "The source activity baseline predicted better; this measures fixed-model execution.",
            "Within-session intervals do not establish session or training-seed variability.",
        ],
    }
    if args.resume and args.output.is_file():
        report["archived_previous_report"] = archive_previous_report(args.output)
    write_json(args.output, report)
    monitor = None
    collection_clock = None
    try:
        source, source_records = load_source_profile(args)
        report["source_profile_sha256"] = SOURCE_PROFILE_SHA256
        workload = load_trained_workload(
            args.checkpoint,
            args.training_report,
            args.data_dir,
            device=args.device,
            download=args.download,
        )
        report["reconstruction"] = {
            **workload.audits,
            "checkpoint": workload.checkpoint_metadata,
            "setup_seconds": workload.setup_seconds,
        }
        report["trained"] = True
        identity = gpu_identity(args.device)
        contract, differences = make_gc_contract(workload, args, source, identity)
        report["source_contract_differences"] = differences
        report["benchmark_contract"] = contract
        contract_sha = canonical_sha256(contract)
        report["benchmark_contract_sha256"] = contract_sha
        print(
            "Replaying all saved validation predictions before GC-aware measurement...", flush=True
        )
        with acyclic_replay_audit():
            report["validation_replay"] = replay_saved_validation(workload, atol=ATOL, rtol=RTOL)
        report["validation_replay"]["fingerprint_policy"] = "acyclic byte-identical serialization"
        workload.model.eval()
        original_hash = model_sha256(workload.model)
        if original_hash != source["model_state_sha256_before"]:
            raise ValueError("Frozen model differs from the pinned source profile")
        report["model_state_sha256_before"] = original_hash
        plans = list(
            batch_plan(len(workload.val_queries), args.batch_sizes, args.repeats, args.seed)
        )
        for plan in plans:
            plan["model_state_sha256"] = original_hash
            plan["gc_thresholds"] = contract["gc_policy"]["thresholds"]
        report["expected_batches"] = len(plans)
        expected_paths = {artifact_path(args.artifacts_dir, plan) for plan in plans}
        present = set(args.artifacts_dir.iterdir()) if args.artifacts_dir.exists() else set()
        if present - expected_paths:
            raise ValueError("GC artifact directory contains records outside this v2 protocol")
        existing = {}
        for plan in plans:
            path = artifact_path(args.artifacts_dir, plan)
            if path.exists():
                existing[path] = validate_artifact(
                    json.loads(path.read_text()),
                    plan,
                    contract_sha,
                    args.repeats,
                    _source_record(source_records[(plan["batch_size"], plan["batch_index"])]),
                )["session_id"]
        origin = {
            "session_id": session_id,
            "environment": workload.environment,
            "gpu_identity": identity,
        }
        write_json(args.output, report)
        collection_clock = time.perf_counter()
        report["collection_program_window"] = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "gc_stats_before": gc.get_stats(),
            "gc_count_before": list(gc.get_count()),
            "observed_case_gc_scope": (
                "Event durations cover profile_batch_gc case windows only; natural GC during "
                "loader creation, reporting, summaries and gaps is in program wall "
                "but is not event-timed"
            ),
            "scope": (
                "This attempt after setup, source/resume audits and saved-validation replay; "
                "includes collection, warmups, audits, separate stage replays, progress "
                "reporting and monitor teardown; "
                "excludes final summary serialization"
            ),
        }
        monitor = NvidiaSmiMonitor(args.output.parent, session_id, device=args.device)
        with monitor:
            report["telemetry"] = monitor.metadata
            for size in args.batch_sizes:
                loader = make_query_loader(
                    workload.graph,
                    workload.val_queries,
                    workload.task,
                    workload.contract["settings"]["fanouts"],
                    size,
                )
                selected = [plan for plan in plans if plan["batch_size"] == size]
                for plan in selected:
                    path = artifact_path(args.artifacts_dir, plan)
                    reused = path in existing
                    if reused:
                        source_session = existing[path]
                    else:
                        if (
                            not gc.isenabled()
                            or list(gc.get_threshold()) != contract["gc_policy"]["thresholds"]
                        ):
                            raise ValueError(
                                "Natural GC enabled state/thresholds changed during collection"
                            )
                        torch.manual_seed(plan["sampling_seed"])
                        case_started = datetime.now(timezone.utc).isoformat()
                        case_clock = time.perf_counter()
                        result = profile_batch_gc(
                            loader,
                            plan["query_indices"],
                            workload.model,
                            workload.task.entity_table,
                            repeats=args.repeats,
                            warmup=args.warmup,
                            seed=plan["sampling_seed"],
                            device=args.device,
                        )
                        result["collection_window"] = {
                            "started_at_utc": case_started,
                            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                            "wall_seconds": time.perf_counter() - case_clock,
                            "scope": (
                                "paired case: warmups, online trials, audits, stage replays, "
                                "GC observation and observer processing"
                            ),
                        }
                        source_record = _source_record(source_records[(size, plan["batch_index"])])
                        verify_source_match(result, source_record)
                        if model_sha256(workload.model) != original_hash:
                            raise AssertionError(
                                "Frozen weights or buffers changed during collection"
                            )
                        save_batch(
                            args.artifacts_dir,
                            plan,
                            result,
                            contract_sha,
                            session_id,
                            origin,
                            source_record,
                        )
                        del result, source_record
                        source_session = session_id
                        report["completed_batches_this_attempt"] += 1
                    report["artifacts"].append(
                        {
                            "path": str(path),
                            "sha256": file_sha256(path),
                            "batch_size": size,
                            "batch_index": plan["batch_index"],
                            "query_count": len(plan["query_indices"]),
                            "source_session_id": source_session,
                            "reused": reused,
                        }
                    )
                    report["completed_batches"] += 1
                    report["host_memory"] = host_memory()
                    write_json(args.output, report)
                    print(
                        f"GC batch size {size}: {plan['batch_index'] + 1}/{len(selected)}; "
                        f"overall {report['completed_batches']}/{len(plans)} "
                        f"({'restored' if reused else 'measured'})",
                        flush=True,
                    )
                report["summary_by_batch_size"][str(size)] = collect_size_summary(
                    args.artifacts_dir,
                    plans,
                    contract_sha,
                    args.repeats,
                    size,
                    args.bootstrap_samples,
                    args.seed,
                    source_records,
                )
                write_json(args.output, report)
        report["telemetry"] = monitor.metadata
        report["model_state_sha256_after"] = model_sha256(workload.model)
        if report["model_state_sha256_after"] != original_hash:
            raise AssertionError("Frozen model changed during GC-aware inference")
        for path, expected in (
            (args.checkpoint, workload.checkpoint_metadata["sha256"]),
            (args.training_report, workload.checkpoint_metadata["source_report_sha256"]),
            (args.source_profile, SOURCE_PROFILE_SHA256),
        ):
            if file_sha256(path) != expected:
                raise AssertionError("A protected source input changed during collection")
        report["status"] = "passed"
        report["measurements_complete"] = True
        report["source_session_ids"] = sorted(
            {item["source_session_id"] for item in report["artifacts"]}
        )
        report["mixed_session_measurement"] = len(report["source_session_ids"]) > 1
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        if hasattr(error, "gc_measurement_trace"):
            trace_path = args.output.parent / f"failed-gc-trace-{session_id}.json"
            write_json(trace_path, error.gc_measurement_trace)
            report["failed_case_gc_trace"] = {
                "path": str(trace_path),
                "sha256": file_sha256(trace_path),
                "batch_size": plan["batch_size"],
                "batch_index": plan["batch_index"],
                "query_indices": plan["query_indices"],
                "sampling_seed": plan["sampling_seed"],
            }
        if monitor is not None:
            report["telemetry"] = monitor.metadata
        raise
    finally:
        if collection_clock is not None:
            report["collection_program_window"].update(
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
                wall_seconds=time.perf_counter() - collection_clock,
                gc_stats_after=gc.get_stats(),
                gc_count_after=list(gc.get_count()),
            )
        write_json(args.output, report)
    print(
        f"Saved {args.output} and {report['completed_batches']} GC-aware batch sidecars.",
        flush=True,
    )


if __name__ == "__main__":
    main()
