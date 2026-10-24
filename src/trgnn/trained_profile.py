"""Paired trained inference on the frozen source validation cohort.

Reconstruction and complete saved-validation replay precede every attempt.
Each completed query batch is an atomic sidecar; progress reports contain an
index and summaries rather than repeatedly serializing the complete trial set.
"""

import argparse
import hashlib
import json
import math
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from trgnn.artifacts import environment, write_json
from trgnn.execution import MODES
from trgnn.gpu_telemetry import NvidiaSmiMonitor, gpu_identity
from trgnn.inference_statistics import summarize_profiles
from trgnn.pipeline_profile import STAGES, host_memory, profile_batch
from trgnn.trained_checkpoint import load_trained_workload, replay_saved_validation
from trgnn.training_pilot import archive_previous_report, file_sha256, make_query_loader

FORMAT_VERSION = 1
ATOL, RTOL = 1e-5, 1e-4
PROFILE_SOURCE_FILES = (
    "trained_profile.py",
    "trained_checkpoint.py",
    "inference_statistics.py",
    "gpu_telemetry.py",
)
ONLINE_BOUNDARY = "serial sampling + gather + transfer + encoders + GNN + readout"


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def model_sha256(model):
    """Hash weights and buffers outside timing to detect inference mutation."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cpu_model():
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def make_benchmark_contract(workload, args, identity):
    project = Path(__file__).resolve().parents[2]
    runtime = workload.environment
    return {
        "format_version": FORMAT_VERSION,
        "checkpoint_sha256": workload.checkpoint_metadata["sha256"],
        "training_report_sha256": workload.checkpoint_metadata["source_report_sha256"],
        "training_input_contract_sha256": workload.checkpoint_metadata["input_contract_sha256"],
        "profile_code_sha256": {
            name: file_sha256(project / "src/trgnn" / name) for name in PROFILE_SOURCE_FILES
        },
        "source_runtime_versions": workload.contract["runtime_versions"],
        "profiling_runtime": {
            "python": runtime["python"],
            "cuda_runtime": runtime["cuda_runtime"],
        },
        "hardware": {
            "device": args.device,
            "gpu": runtime["gpu"],
            "driver_version": identity.get("driver_version"),
            "cpu_model": cpu_model(),
            "cpu_threads": runtime["cpu_threads"],
            "platform_family": platform.system(),
        },
        "protocol": {
            "batch_sizes": args.batch_sizes,
            "primary_batch_size": args.batch_sizes[0],
            "query_count": len(workload.val_queries),
            "cohort": "complete source-report validation cohort in saved selected-row order",
            "repeats": args.repeats,
            "warmup": args.warmup,
            "seed": args.seed,
            "seed_scheme": "seed + batch-size position * 1000000 + batch_index * repeats + repeat",
            "warmup_seed": "reset torch RNG to the query-batch sampling seed before warmups",
            "atol": ATOL,
            "rtol": RTOL,
            "online_boundary": ONLINE_BOUNDARY,
            "stage_timing": "separate synchronized replay; not added to online latency",
            "bootstrap_samples": args.bootstrap_samples,
            "telemetry_sampling_interval_ms": 1000,
        },
    }


def batch_plan(query_count, batch_sizes, repeats, seed):
    for position, size in enumerate(batch_sizes):
        for index, start in enumerate(range(0, query_count, size)):
            yield {
                "batch_size": size,
                "batch_index": index,
                "query_indices": list(range(start, min(start + size, query_count))),
                "sampling_seed": seed + position * 1_000_000 + index * repeats,
            }


def artifact_path(directory, planned):
    return directory / f"batch-{planned['batch_size']:06d}-{planned['batch_index']:06d}.json"


def validate_artifact(record, planned, contract_sha, repeats):
    """Reject partial, altered or incompatible results before accepting resume."""
    if not isinstance(record, dict):
        raise ValueError("Malformed measurement sidecar")
    payload = {key: value for key, value in record.items() if key != "artifact_payload_sha256"}
    if canonical_sha256(payload) != record.get("artifact_payload_sha256"):
        raise ValueError("Measurement sidecar checksum mismatch")
    if (
        record.get("status") != "passed"
        or record.get("format_version") != FORMAT_VERSION
        or record.get("benchmark_contract_sha256") != contract_sha
        or any(record.get(key) != value for key, value in planned.items())
        or record.get("query_count") != len(planned["query_indices"])
        or not record.get("session_id")
        or not isinstance(record.get("origin_environment"), dict)
        or record["origin_environment"].get("session_id") != record["session_id"]
    ):
        raise ValueError("Measurement sidecar protocol or query cohort mismatch")
    trials = record.get("trials")
    if not isinstance(trials, list) or len(trials) != repeats:
        raise ValueError("Measurement sidecar has incomplete paired trials")
    try:
        for offset, trial in enumerate(trials):
            if (
                trial["sampler_seed"] != planned["sampling_seed"] + offset
                or sorted(trial["execution_order"]) != sorted(MODES)
                or trial["visibility_audit"]["status"] != "passed"
                or len(trial["sampler_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in trial["sampler_sha256"])
            ):
                raise ValueError("Measurement sidecar pairing or occurrence audit mismatch")
            for mode in MODES:
                times = [trial["pipeline_ms"][mode]]
                times.extend(trial["stage_ms"][mode][stage] for stage in STAGES)
                if any(
                    isinstance(value, bool) or not math.isfinite(value) or value <= 0
                    for value in times
                ):
                    raise ValueError("Measurement sidecar contains invalid timings")
                for field in ("max_absolute_logit_error", "max_absolute_loss_error"):
                    error = record[field][mode]
                    if not math.isfinite(error) or error < 0:
                        raise ValueError("Measurement sidecar contains invalid equivalence errors")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed paired measurement sidecar") from exc
    summarize_profiles([record], planned["batch_size"], bootstrap_samples=1)
    return record


def save_batch(directory, planned, profile, contract_sha, session_id, origin):
    record = {
        **profile,
        **planned,
        "status": "passed",
        "format_version": FORMAT_VERSION,
        "benchmark_contract_sha256": contract_sha,
        "query_count": len(planned["query_indices"]),
        "session_id": session_id,
        "origin_environment": origin,
    }
    record["artifact_payload_sha256"] = canonical_sha256(record)
    validate_artifact(record, planned, contract_sha, len(profile["trials"]))
    path = artifact_path(directory, planned)
    write_json(path, record)
    return path, record


def collect_size_summary(directory, plans, contract_sha, repeats, batch_size, bootstrap, seed):
    records = [
        validate_artifact(
            json.loads(artifact_path(directory, item).read_text()), item, contract_sha, repeats
        )
        for item in plans
        if item["batch_size"] == batch_size
    ]
    summary = summarize_profiles(
        records, batch_size, bootstrap_samples=bootstrap, seed=seed + batch_size
    )
    summary["source_session_ids"] = sorted({record["session_id"] for record in records})
    summary["max_absolute_logit_error"] = {
        mode: max(record["max_absolute_logit_error"][mode] for record in records) for mode in MODES
    }
    summary["max_absolute_loss_error"] = {
        mode: max(record["max_absolute_loss_error"][mode] for record in records) for mode in MODES
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rel-stack"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/trained-profile.json"))
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
        parser.error("Use distinct positive batch sizes, >=3 repeats, nonnegative warmup/seed")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Select a CUDA runtime or explicitly use --device cpu")
    if args.artifacts_dir is None:
        args.artifacts_dir = args.output.parent / "batches"
    protected = [args.checkpoint.resolve(), args.training_report.resolve()]
    if args.output.resolve() in protected or args.artifacts_dir.resolve() in protected:
        parser.error("Output paths must not overwrite source checkpoint or training report")
    if args.output.resolve() == args.artifacts_dir.resolve():
        parser.error("Summary and measurement directory paths must differ")
    if any(
        path.is_relative_to(args.artifacts_dir.resolve())
        for path in [*protected, args.output.resolve()]
    ):
        parser.error("Use a separate artifact folder from source files and the summary")
    return args


def main():
    args = parse_args()
    if not args.resume and (args.output.exists() or list(args.artifacts_dir.glob("batch-*.json"))):
        raise FileExistsError("Measurements already exist; use --resume or a new run directory")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    session_id = uuid.uuid4().hex
    report = {
        "status": "started",
        "scope": "trained_real_relational_online_inference",
        "trained": False,
        "training_performed": False,
        "measurements_complete": False,
        "end_to_end": True,
        "online_boundary": ONLINE_BOUNDARY,
        "excluded_from_online_timing": [
            "setup and graph construction",
            "saved-validation replay",
            "query scheduling and worker/DataLoader overhead",
            "correctness audits, RNG seeding, reporting and telemetry setup",
        ],
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "session_id": session_id,
        "environment": environment(),
        "summary_by_batch_size": {},
        "artifacts": [],
        "completed_batches": 0,
        "completed_batches_this_attempt": 0,
        "notes": [
            "All execution modes use the same frozen weights and paired sampled occurrences.",
            "Graph inputs retain the source checkpoint's conditional exported-snapshot semantics.",
            "The source activity baseline scored better; this is an execution experiment.",
            "Batch-size comparisons may use different sampler draws across sizes.",
            "Stage replays are separate measurements; never add their medians to online latency.",
            "Intervals cover measured query batches, not session or training-seed variability.",
        ],
    }
    if args.resume and args.output.is_file():
        report["archived_previous_report"] = archive_previous_report(args.output)
    write_json(args.output, report)
    monitor = None
    try:
        print("Verifying the trained checkpoint, source files and pinned inputs...", flush=True)
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
            "training_input_contract_sha256": workload.checkpoint_metadata["input_contract_sha256"],
        }
        report["trained"] = True
        print("Replaying all saved validation predictions before measuring...", flush=True)
        replay_started = time.perf_counter()
        report["validation_replay"] = replay_saved_validation(workload, atol=ATOL, rtol=RTOL)
        report["validation_replay"]["wall_seconds"] = time.perf_counter() - replay_started
        workload.model.eval()
        original_model_hash = model_sha256(workload.model)
        report["model_state_sha256_before"] = original_model_hash
        identity = gpu_identity(args.device)
        contract = make_benchmark_contract(workload, args, identity)
        contract_sha = canonical_sha256(contract)
        report["benchmark_contract"] = contract
        report["benchmark_contract_sha256"] = contract_sha
        plans = list(
            batch_plan(len(workload.val_queries), args.batch_sizes, args.repeats, args.seed)
        )
        for item in plans:
            item["model_state_sha256"] = original_model_hash
        report["expected_batches"] = len(plans)
        expected_paths = {artifact_path(args.artifacts_dir, item) for item in plans}
        unexpected = set(args.artifacts_dir.glob("batch-*.json")) - expected_paths
        if unexpected:
            raise ValueError(
                "Measurement directory contains batches outside the requested protocol"
            )
        origin = {
            "session_id": session_id,
            "environment": workload.environment,
            "gpu_identity": identity,
        }
        # Verify every existing sidecar before collecting any new measurements.
        existing = {}
        for item in plans:
            path = artifact_path(args.artifacts_dir, item)
            if path.exists():
                existing[path] = validate_artifact(
                    json.loads(path.read_text()), item, contract_sha, args.repeats
                )["session_id"]
        write_json(args.output, report)
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
                for item in (plan for plan in plans if plan["batch_size"] == size):
                    path = artifact_path(args.artifacts_dir, item)
                    reused = path in existing
                    if reused:
                        source_session = existing[path]
                    else:
                        # Warmups must not depend on which earlier batches were restored.
                        torch.manual_seed(item["sampling_seed"])
                        collection_started = datetime.now(timezone.utc).isoformat()
                        collection_clock = time.perf_counter()
                        result = profile_batch(
                            loader,
                            item["query_indices"],
                            workload.model,
                            workload.task.entity_table,
                            repeats=args.repeats,
                            warmup=args.warmup,
                            seed=item["sampling_seed"],
                            device=args.device,
                        )
                        result["collection_window"] = {
                            "started_at_utc": collection_started,
                            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                            "wall_seconds": time.perf_counter() - collection_clock,
                            "scope": "warmup, online trials, audits and separate stage replays",
                        }
                        if model_sha256(workload.model) != original_model_hash:
                            raise AssertionError(
                                "Frozen model weights or buffers changed during a measured batch"
                            )
                        save_batch(
                            args.artifacts_dir, item, result, contract_sha, session_id, origin
                        )
                        del result
                        source_session = session_id
                        report["completed_batches_this_attempt"] += 1
                    report["artifacts"].append(
                        {
                            "path": str(path),
                            "sha256": file_sha256(path),
                            "batch_size": size,
                            "batch_index": item["batch_index"],
                            "query_count": len(item["query_indices"]),
                            "source_session_id": source_session,
                            "reused": reused,
                        }
                    )
                    report["completed_batches"] += 1
                    report["host_memory"] = host_memory()
                    write_json(args.output, report)
                    print(
                        f"Batch size {size}: {item['batch_index'] + 1}/"
                        f"{sum(plan['batch_size'] == size for plan in plans)} "
                        f"query batches {'restored' if reused else 'measured'}; "
                        f"overall {report['completed_batches']}/{len(plans)}",
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
                )
                write_json(args.output, report)
        report["telemetry"] = monitor.metadata
        report["model_state_sha256_after"] = model_sha256(workload.model)
        if report["model_state_sha256_after"] != original_model_hash:
            raise AssertionError("Frozen model weights or buffers changed during inference")
        if file_sha256(args.checkpoint) != workload.checkpoint_metadata["sha256"]:
            raise AssertionError("Source checkpoint changed during profiling")
        report["status"] = "passed"
        report["measurements_complete"] = True
        report["source_session_ids"] = sorted(
            {item["source_session_id"] for item in report["artifacts"]}
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if monitor is not None:
            report["telemetry"] = monitor.metadata
        write_json(args.output, report)
        raise
    write_json(args.output, report)
    print(
        f"Saved {args.output} and {report['completed_batches']} paired batch sidecars.", flush=True
    )


if __name__ == "__main__":
    main()
