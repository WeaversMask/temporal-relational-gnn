"""Reconstruct a frozen trained workload and replay its saved validation results.

Only inference is performed. The source training implementation remains immutable:
its checkpoint contract binds source files, pinned input bytes, query occurrences,
feature policy, architecture and dependency versions before any profiling starts.
"""

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from relbench import load_dataset
from relbench.modeling.graph import make_pkey_fkey_graph

from trgnn.artifacts import environment
from trgnn.feature_audit import audit_feature_availability
from trgnn.real_batch import RealBatchModel
from trgnn.real_data import (
    download_inputs,
    load_specs,
    manifest_hashes,
    projected_database,
    verify_inputs,
)
from trgnn.training import (
    CHECKPOINT_VERSION,
    QUERY_SCHEME,
    SEED_SCHEME,
    _architecture,
    _input_signature,
    evaluate,
)
from trgnn.training_data import select_training_queries, validate_query_entities
from trgnn.training_pilot import file_sha256, fixed_constant_statistics, make_query_loader

SOURCE_FILES = (
    "training_pilot.py",
    "training.py",
    "training_data.py",
    "activity_baseline.py",
    "real_batch.py",
    "real_data.py",
    "execution.py",
    "feature_audit.py",
    "pipeline_profile.py",
    "artifacts.py",
)
PACKAGE_NAMES = (
    "torch",
    "torch-geometric",
    "pyg_lib",
    "relbench",
    "pytorch-frame",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
)


@dataclass
class TrainedWorkload:
    model: RealBatchModel
    graph: object
    task: object
    train_queries: pd.DataFrame
    val_queries: pd.DataFrame
    checkpoint_metadata: dict
    training_report: dict
    contract: dict
    setup_seconds: dict
    environment: dict
    audits: dict


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _require_equal(actual, expected, description):
    if actual != expected:
        raise ValueError(f"Frozen checkpoint {description} mismatch")


def _read_source(checkpoint_path, training_report_path):
    report = json.loads(training_report_path.read_text())
    if (
        report.get("status") != "passed"
        or report.get("trained") is not True
        or report.get("training", {}).get("status") != "passed"
    ):
        raise ValueError("A passed trained source report is required")
    checksum = file_sha256(checkpoint_path)
    _require_equal(checksum, report["checkpoint"]["sha256"], "SHA256")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _require_equal(checkpoint.get("format_version"), CHECKPOINT_VERSION, "format version")
    contract = checkpoint["contract"]
    _require_equal(contract, report["input_contract"], "input contract")
    _require_equal(_canonical_sha256(contract), report["input_contract_sha256"], "contract digest")
    _require_equal(contract.get("format_version"), 1, "input contract format version")
    _require_equal(contract.get("training_execution_mode"), "reference", "training mode")
    completed = checkpoint["completed_steps"]
    if type(completed) is not int or completed < 1:
        raise ValueError("A trained checkpoint with completed updates is required")
    for saved in (report["training"]["completed_steps"], report["checkpoint"]["completed_steps"]):
        _require_equal(completed, saved, "completed steps")
    _require_equal(completed, report["config"]["steps"], "requested training steps")
    history = checkpoint["history"]
    _require_equal(history, report["training"]["history"], "training history")
    _require_equal([row["step"] for row in history], list(range(1, completed + 1)), "history steps")
    settings = contract["settings"]
    _require_equal(
        settings,
        {name: report["config"][name] for name in settings},
        "report configuration",
    )
    required = {
        "device",
        "batch_size",
        "train_queries",
        "val_queries",
        "channels",
        "fanouts",
        "learning_rate",
        "seed",
    }
    _require_equal(set(settings), required, "configuration keys")
    saved = checkpoint["settings"]
    expected = {
        "seed": settings["seed"],
        "batch_size": settings["batch_size"],
        "query_count": settings["train_queries"],
        "seed_scheme": SEED_SCHEME,
        "query_scheme": QUERY_SCHEME,
    }
    for name, value in expected.items():
        _require_equal(saved[name], value, f"training setting {name}")
    query_order = checkpoint["query_order"]
    generator = torch.Generator(device="cpu").manual_seed(settings["seed"])
    expected_order = torch.randperm(settings["train_queries"], generator=generator)
    if not isinstance(query_order, torch.Tensor) or not torch.equal(query_order, expected_order):
        raise ValueError("Frozen checkpoint query order mismatch")
    for index, row in enumerate(history):
        expected_indices = expected_order[
            (torch.arange(settings["batch_size"]) + index * settings["batch_size"])
            % settings["train_queries"]
        ].tolist()
        _require_equal(row["query_indices"], expected_indices, "history query occurrences")
        _require_equal(row["query_count"], settings["batch_size"], "history query count")
        _require_equal(row["sampler_seed"], settings["seed"] + index, "history sampler seed")
        if (
            row.get("gradient_finite") is not True
            or not math.isfinite(row["loss"])
            or row.get("visibility_audit", {}).get("status") != "passed"
        ):
            raise ValueError("Source training history contains an unsuccessful update")
    if (
        report.get("validation_after", {}).get("metrics", {}).get("count")
        != settings["val_queries"]
    ):
        raise ValueError("Source validation must cover the complete frozen validation cohort")
    return report, checkpoint, checksum


def _verify_source_runtime(contract, report, runtime, device, project):
    _require_equal(device.type, contract["settings"]["device"], "device family for source replay")
    _require_equal(set(contract["code_sha256"]), set(SOURCE_FILES), "source file set")
    for name in SOURCE_FILES:
        _require_equal(
            file_sha256(project / "src/trgnn" / name),
            contract["code_sha256"][name],
            f"source file {name}",
        )
    _require_equal(set(contract["runtime_versions"]), set(PACKAGE_NAMES), "runtime package set")
    for name in PACKAGE_NAMES:
        expected = contract["runtime_versions"][name]
        _require_equal(runtime["packages"].get(name), expected, f"runtime package {name}")
    if device.type == "cuda":
        if not runtime["cuda_available"]:
            raise ValueError("The frozen source checkpoint requires a CUDA runtime")
        _require_equal(
            runtime["cuda_runtime"], report["environment"]["cuda_runtime"], "CUDA runtime"
        )


def load_trained_workload(
    checkpoint_path, training_report_path, data_dir, *, device="cpu", download=False
):
    """Load verified fixed weights and the exact source query cohort for inference.

    The checkpoint is loaded with ``weights_only=True``. No optimizer is created,
    no baseline is refitted, and no test query file is requested or read.
    """
    checkpoint_path, training_report_path, data_dir = map(
        Path, (checkpoint_path, training_report_path, data_dir)
    )
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Frozen inference supports only CPU and CUDA devices")
    setup = {}
    started = time.perf_counter()
    report, checkpoint, checksum = _read_source(checkpoint_path, training_report_path)
    project = Path(__file__).resolve().parents[2]
    runtime = environment()
    contract = checkpoint["contract"]
    _verify_source_runtime(contract, report, runtime, device, project)
    pins = json.loads((project / "configs/upstream.json").read_text())
    _require_equal(pins["dataset"], contract["dataset"], "pinned dataset")
    _require_equal(
        file_sha256(project / "configs/feature_policy.json"),
        contract["feature_policy_sha256"],
        "feature policy",
    )
    setup["checkpoint_and_source_verification"] = time.perf_counter() - started
    started = time.perf_counter()
    if download:
        download_inputs(data_dir, pins, splits=("train", "val"))
    verify_inputs(data_dir, pins)
    _require_equal(manifest_hashes(data_dir), contract["manifest_sha256"], "manifest bytes")
    database_spec, task_spec = load_specs(data_dir)
    input_hashes = {
        f"db/{name}.parquet": file_sha256(data_dir / "db" / f"{name}.parquet")
        for name in database_spec["tables"]
    }
    _require_equal(input_hashes, contract["database_sha256"], "database bytes")
    feature_audit = audit_feature_availability(data_dir, pins)
    if feature_audit["status"] != "passed":
        raise ValueError("The frozen constant/time feature-policy audit failed")
    _require_equal(
        feature_audit["policy_sha256"], contract["feature_policy_sha256"], "feature policy"
    )
    settings = contract["settings"]
    train_queries, val_queries, split_audit = select_training_queries(
        data_dir, settings["train_queries"], settings["val_queries"], settings["seed"]
    )
    _require_equal(split_audit, contract["split_audit"], "official split bytes and selected rows")
    _require_equal(split_audit, report["split_audit"], "reported official split")
    cutoff = val_queries[task_spec["time_col"]].max()
    _require_equal(str(cutoff), contract["graph_cutoff"], "graph cutoff")
    _require_equal(str(cutoff), report["graph_visibility_cutoff"], "reported graph cutoff")
    setup["download_and_input_verification"] = time.perf_counter() - started
    started = time.perf_counter()
    database, table_audit = projected_database(data_dir, cutoff)
    _require_equal(table_audit, report["table_audit"], "projected tables")
    eligibility = {
        "train": validate_query_entities(database, train_queries, task_spec),
        "val": validate_query_entities(database, val_queries, task_spec),
    }
    _require_equal(eligibility, report["query_eligibility"], "query visibility")
    setup["projection_and_query_validation"] = time.perf_counter() - started
    started = time.perf_counter()
    graph, _ = make_pkey_fkey_graph(database, {name: {} for name in database.table_dict})
    del database
    stats = fixed_constant_statistics(graph)
    task = load_dataset(data_dir).load_task("user-badge")
    train_loader = make_query_loader(
        graph, train_queries, task, settings["fanouts"], settings["batch_size"]
    )
    _require_equal(checkpoint["settings"]["entity"], task.entity_table, "query entity")
    _require_equal(
        _input_signature(train_loader), checkpoint["settings"]["input_sha256"], "training inputs"
    )
    model = RealBatchModel(graph, stats, settings["channels"], len(settings["fanouts"]))
    _require_equal(_architecture(model), checkpoint["settings"]["architecture"], "architecture")
    for name, value in checkpoint["model"].items():
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite or invalid frozen model state: {name}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    setup["graph_and_frozen_model_construction"] = time.perf_counter() - started
    return TrainedWorkload(
        model=model,
        graph=graph,
        task=task,
        train_queries=train_queries,
        val_queries=val_queries,
        checkpoint_metadata={
            "path": str(checkpoint_path),
            "sha256": checksum,
            "completed_steps": checkpoint["completed_steps"],
            "source_report_path": str(training_report_path),
            "source_report_sha256": file_sha256(training_report_path),
            "input_contract_sha256": report["input_contract_sha256"],
        },
        training_report=report,
        contract=contract,
        setup_seconds=setup,
        environment=runtime,
        audits={
            "status": "passed",
            "feature_audit": feature_audit,
            "query_eligibility": eligibility,
            "table_audit": table_audit,
            "source_files_match": True,
            "runtime_versions_match": True,
            "input_bytes_and_cohorts_match": True,
            "baseline_refitted": False,
            "test_queries_read": False,
        },
    )


def _compare_metrics(replayed, saved, atol, rtol):
    _require_equal(set(replayed), set(saved), "validation metric keys")
    for name, expected in saved.items():
        value = replayed[name]
        if isinstance(expected, float):
            if not math.isfinite(value) or not math.isclose(
                value, expected, abs_tol=atol, rel_tol=rtol
            ):
                raise ValueError(f"Frozen checkpoint validation metric {name} mismatch")
        else:
            _require_equal(value, expected, f"validation metric {name}")


def replay_saved_validation(workload, *, atol=1e-5, rtol=1e-4):
    """Replay every saved source validation occurrence before timing new batches."""
    if not math.isfinite(atol) or not math.isfinite(rtol) or min(atol, rtol) < 0:
        raise ValueError("Use finite non-negative validation replay tolerances")
    settings = workload.contract["settings"]
    loader = make_query_loader(
        workload.graph,
        workload.val_queries,
        workload.task,
        settings["fanouts"],
        settings["batch_size"],
    )
    actual = evaluate(
        workload.model,
        loader,
        workload.task.entity_table,
        batch_size=settings["batch_size"],
        seed=settings["seed"] + 1_000_000,
        device=next(workload.model.parameters()).device,
    )
    expected = workload.training_report["validation_after"]
    for name in ("labels", "sampler_sha256", "sampled_counts", "visibility_audits"):
        _require_equal(actual[name], expected[name], f"validation {name}")
    for name in ("logits", "predictions"):
        values = torch.tensor(actual[name], dtype=torch.float64)
        saved = torch.tensor(expected[name], dtype=torch.float64)
        if not bool(torch.isfinite(values).all()) or values.shape != saved.shape:
            raise ValueError(f"Frozen checkpoint validation {name} shape/finiteness mismatch")
        try:
            torch.testing.assert_close(values, saved, atol=atol, rtol=rtol)
        except AssertionError as exc:
            raise ValueError(f"Frozen checkpoint validation {name} mismatch") from exc
    _compare_metrics(actual["metrics"], expected["metrics"], atol, rtol)
    error = float(
        (
            torch.tensor(actual["logits"], dtype=torch.float64)
            - torch.tensor(expected["logits"], dtype=torch.float64)
        )
        .abs()
        .max()
    )
    return {
        "status": "passed",
        "scope": "all frozen validation queries at the original training evaluation batch size",
        "validated_query_count": len(actual["labels"]),
        "validated_batch_count": len(actual["sampler_sha256"]),
        "batch_size": settings["batch_size"],
        "seed": settings["seed"] + 1_000_000,
        "sampler_sha256": actual["sampler_sha256"],
        "source_metrics": expected["metrics"],
        "replayed_metrics": actual["metrics"],
        "max_absolute_logit_error": error,
        "atol": atol,
        "rtol": rtol,
        "new_training_updates": 0,
        "baseline_refitted": False,
    }
