"""Bounded supervised training on the pinned exported-snapshot benchmark.

This trains the established reference architecture before comparing execution
schedules on the resulting weights. It does not approve additional attributes,
reconstruct deleted records, or establish final predictive quality.
"""

import argparse
import copy
import hashlib
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from relbench import load_dataset
from relbench.base import Table
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from torch import nn
from torch_frame import stype
from torch_frame.data.stats import StatType
from torch_geometric.loader import NeighborLoader

from trgnn.activity_baseline import fit_activity_baseline
from trgnn.artifacts import environment, write_json
from trgnn.execution import MODES
from trgnn.feature_audit import audit_feature_availability
from trgnn.pipeline_profile import host_memory, sampler_fingerprint, synchronize, validate_batch
from trgnn.preflight import temporal_sampler_smoke
from trgnn.real_batch import RealBatchModel
from trgnn.real_data import (
    download_inputs,
    load_specs,
    manifest_hashes,
    projected_database,
    verify_inputs,
)
from trgnn.training import evaluate, train_pilot
from trgnn.training_data import select_training_queries, validate_query_entities


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_previous_report(path: Path) -> dict:
    """Preserve a retry's source report byte-for-byte before replacing its path."""
    archived = path.with_name(f"{path.stem}.previous-{uuid.uuid4().hex}{path.suffix}")
    with archived.open("xb") as stream:
        stream.write(path.read_bytes())
    return {"path": str(archived), "sha256": file_sha256(archived)}


class TrainingProgressReporter:
    """Keep all updates in memory while bounding report writes and stdout updates."""

    def __init__(self, report, *, output, checkpoint, steps, checkpoint_every, report_every):
        self.report = report
        self.output = output
        self.checkpoint = checkpoint
        self.steps = steps
        self.checkpoint_every = checkpoint_every
        self.report_every = report_every

    def record_step(self, record):
        self.report["training"]["history"].append(record)
        self.report["training"]["completed_steps"] = record["step"]
        self.report["trained"] = True
        if record["step"] % self.checkpoint_every == 0 or record["step"] == self.steps:
            self.report["checkpoint"].update(
                {"sha256": file_sha256(self.checkpoint), "completed_steps": record["step"]}
            )
        elif "completed_steps" not in self.report["checkpoint"]:
            # Fresh training saves step zero before its first update. Resumed
            # progress already has the saved count from record_restore().
            self.report["checkpoint"].update(
                {"sha256": file_sha256(self.checkpoint), "completed_steps": 0}
            )
        if record["step"] % self.report_every == 0 or record["step"] == self.steps:
            write_json(self.output, self.report)
            print(f"Step {record['step']}/{self.steps}: loss={record['loss']:.6f}", flush=True)

    def record_restore(self, progress):
        self.report["training"].update(progress)
        self.report["trained"] = progress["completed_steps"] > 0
        self.report["checkpoint"]["completed_steps"] = progress["completed_steps"]
        self.report["checkpoint"]["sha256"] = file_sha256(self.checkpoint)
        write_json(self.output, self.report)


def fixed_constant_statistics(graph) -> dict:
    """Use literal statistics for literal ones, independent of validation data."""
    stats = {
        "__const__": {
            StatType.MEAN: 1.0,
            StatType.STD: 0.0,
            StatType.QUANTILES: [1.0] * 5,
        }
    }
    result = {}
    for name in graph.node_types:
        frame = graph[name].tf
        if frame.col_names_dict != {stype.numerical: ["__const__"]}:
            raise ValueError("The pilot only supports the approved constant node feature")
        values = frame.feat_dict[stype.numerical]
        if not bool((values == 1).all()):
            raise ValueError("Expected literal constant-one node features")
        result[name] = copy.deepcopy(stats)
    return result


def make_query_loader(graph, queries, task, fanouts, batch_size):
    table = Table(queries, {task.entity_col: task.entity_table}, time_col=task.time_col)
    inputs = get_node_train_table_input(table, task)
    return NeighborLoader(
        graph,
        input_nodes=inputs.nodes,
        input_time=inputs.time,
        transform=inputs.transform,
        time_attr="time",
        num_neighbors=fanouts,
        disjoint=True,
        subgraph_type="directional",
        temporal_strategy="uniform",
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        filter_per_worker=False,
    )


def trained_equivalence(model, loader, entity, batch_size, seed, device):
    """Same trained weights and one actual validation sample; no timing claim."""
    torch.manual_seed(seed)
    indices = list(range(min(batch_size, loader.input_data.node.numel())))
    sample = loader.node_sampler.sample_from_nodes(loader.input_data[indices])
    fingerprint = sampler_fingerprint(sample)
    batch = loader.filter_fn(sample)
    visibility = validate_batch(batch, entity, len(model.gnn.convs))
    batch = batch.to(device)
    model.eval()
    errors, loss_errors = {}, {}
    with torch.inference_mode():
        reference = model(batch, entity, "reference")
        target = batch[entity].y.to(reference.dtype)
        reference_loss = nn.functional.binary_cross_entropy_with_logits(reference, target)
        for mode in MODES:
            logits = model(batch, entity, mode)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            torch.testing.assert_close(logits, reference, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(loss, reference_loss, atol=1e-5, rtol=1e-4)
            errors[mode] = float((logits - reference).abs().max())
            loss_errors[mode] = float((loss - reference_loss).abs())
    return {
        "status": "passed",
        "scope": "one_validation_batch_after_pilot_training",
        "sampler_sha256": fingerprint,
        "visibility_audit": visibility,
        "max_absolute_logit_error": errors,
        "max_absolute_loss_error": loss_errors,
        "atol": 1e-5,
        "rtol": 1e-4,
        "speedup_measured": False,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rel-stack"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-queries", type=int, default=3200)
    parser.add_argument("--val-queries", type=int, default=512)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--fanouts", type=int, nargs="+", default=[16, 8])
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--report-every", type=int, default=1, help="Write progress JSON and stdout every N updates"
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/training-pilot.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/training-pilot.json"))
    args = parser.parse_args()
    dimensions = (
        args.steps,
        args.batch_size,
        args.train_queries,
        args.val_queries,
        args.channels,
        args.checkpoint_every,
        args.report_every,
    )
    if min(dimensions) < 1 or any(n < 0 for n in args.fanouts):
        parser.error("Use positive dimensions and bounded non-negative fanouts")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Use a finite positive learning rate")
    if not 0 <= args.seed < 2**32:
        parser.error("Use a seed between zero and 2**32 - 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Select a CUDA runtime or explicitly use --device cpu")
    if args.checkpoint.resolve() == args.output.resolve():
        parser.error("Checkpoint and JSON report paths must differ")
    return args


def main():
    args = parse_args()
    project = Path(__file__).resolve().parents[2]
    pins = json.loads((project / "configs/upstream.json").read_text())
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    runtime = environment()
    report = {
        "status": "started",
        "scope": "real_relational_supervised_training_pilot",
        "trained": False,
        "final_predictor": False,
        "end_to_end_speedup_measured": False,
        "environment": runtime,
        "dataset": pins["dataset"],
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "training": {"completed_steps": 0, "history": []},
        "checkpoint": {"path": str(args.checkpoint)},
        "setup_seconds": {},
        "notes": [
            "Reference-only training preserves the existing encoder dropout.",
            "GNN inputs are constants and relative record time; no raw attributes are enabled.",
            "Normalizer statistics are literal constants, independent of validation data.",
            "Official train/validation labels remain separate; the test split is untouched.",
            "Each sampled occurrence uses its query timestamp in the exported snapshot graph.",
            "Earlier foreign-key edits, merges, deletions and survivorship are not reconstructed.",
            "Validation uses a fixed subset and sampling seed; this is not final task evaluation.",
            "The short pilot measures supervised execution and cost, not a finished predictor.",
            "Checkpoint resume requires matching inputs, code, configuration and runtime versions.",
        ],
    }
    if args.resume and args.output.is_file():
        report["archived_previous_report"] = archive_previous_report(args.output)
    write_json(args.output, report)
    try:
        if args.resume and not args.checkpoint.is_file():
            raise FileNotFoundError("Resume requires the pilot's previously exported checkpoint")
        if not args.resume and args.checkpoint.exists():
            raise FileExistsError(
                "Checkpoint already exists; choose --resume or a new checkpoint path"
            )
        if args.resume:
            report["checkpoint"]["resume_source_sha256"] = file_sha256(args.checkpoint)
        print("Checking temporal sampler and pinned train/validation files...", flush=True)
        report["sampler_preflight"] = temporal_sampler_smoke()
        started = time.perf_counter()
        if args.download:
            download_inputs(args.data_dir, pins, splits=("train", "val"))
        verify_inputs(args.data_dir, pins)
        report["setup_seconds"]["download_and_verify"] = time.perf_counter() - started
        report["feature_audit"] = audit_feature_availability(args.data_dir, pins)
        if report["feature_audit"]["status"] != "passed":
            raise ValueError("The constant/time feature-policy metadata audit failed")
        train_queries, val_queries, split_audit = select_training_queries(
            args.data_dir, args.train_queries, args.val_queries, args.seed
        )
        report["split_audit"] = split_audit
        database_spec, task_spec = load_specs(args.data_dir)
        started = time.perf_counter()
        input_hashes = {
            f"db/{name}.parquet": file_sha256(args.data_dir / "db" / f"{name}.parquet")
            for name in database_spec["tables"]
        }
        report["setup_seconds"]["database_hashing"] = time.perf_counter() - started
        cutoff = val_queries[task_spec["time_col"]].max()
        started = time.perf_counter()
        database, table_audit = projected_database(args.data_dir, cutoff)
        report["setup_seconds"]["projection"] = time.perf_counter() - started
        report["table_audit"] = table_audit
        report["query_eligibility"] = {
            "train": validate_query_entities(database, train_queries, task_spec),
            "val": validate_query_entities(database, val_queries, task_spec),
        }
        report["graph_visibility_cutoff"] = str(cutoff)
        print("Fitting historical-activity and training-prevalence baselines...", flush=True)
        started = time.perf_counter()
        report["activity_baseline"] = fit_activity_baseline(
            database, train_queries, val_queries, task_spec
        )
        report["setup_seconds"]["activity_baseline"] = time.perf_counter() - started
        write_json(args.output, report)
        print("Building the constant/time historical graph...", flush=True)
        started = time.perf_counter()
        graph, _ = make_pkey_fkey_graph(database, {name: {} for name in database.table_dict})
        del database
        stats = fixed_constant_statistics(graph)
        report["setup_seconds"]["graph_construction"] = time.perf_counter() - started
        report["host_memory"] = {"after_graph": host_memory()}
        # Labels have already been read explicitly. Loading the task only reads
        # metadata; never call get_table(), whose fallback can regenerate labels.
        task = load_dataset(args.data_dir).load_task("user-badge")
        train_loader = make_query_loader(graph, train_queries, task, args.fanouts, args.batch_size)
        val_loader = make_query_loader(graph, val_queries, task, args.fanouts, args.batch_size)
        torch.manual_seed(args.seed)
        model = RealBatchModel(graph, stats, args.channels, len(args.fanouts)).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        package_names = (
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
        code_files = (
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
        contract = {
            "format_version": 1,
            "dataset": pins["dataset"],
            "manifest_sha256": manifest_hashes(args.data_dir),
            "database_sha256": input_hashes,
            "split_audit": split_audit,
            "feature_policy_sha256": report["feature_audit"]["policy_sha256"],
            "code_sha256": {name: file_sha256(project / "src/trgnn" / name) for name in code_files},
            "runtime_versions": {name: runtime["packages"].get(name) for name in package_names},
            "settings": {
                key: getattr(args, key)
                for key in (
                    "device",
                    "batch_size",
                    "train_queries",
                    "val_queries",
                    "channels",
                    "fanouts",
                    "learning_rate",
                    "seed",
                )
            },
            "graph_cutoff": str(cutoff),
            "preprocessing": "literal constant-one statistics; tabular scaler fitted to train only",
            "model": "RelBench feature/time encoders + HeteroGraphSAGE + linear head",
            "optimizer": "Adam",
            "training_execution_mode": "reference",
            "graph_semantics": "pinned exported snapshot with per-query record-time visibility",
        }
        report["input_contract"] = contract
        report["input_contract_sha256"] = hashlib.sha256(
            json.dumps(contract, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if args.resume:
            report["validation_before"] = {
                "status": "not_measured_on_resume",
                "reason": "Initial metrics are kept in the source run.",
            }
        else:
            print("Evaluating the initial model on the fixed validation subset...", flush=True)
            report["validation_before"] = evaluate(
                model,
                val_loader,
                task.entity_table,
                batch_size=args.batch_size,
                seed=args.seed + 1_000_000,
                device=args.device,
            )
        write_json(args.output, report)

        # Reporting and checkpoint intervals do not change the stochastic training
        # schedule, so their values are recorded in config rather than the contract.
        progress = TrainingProgressReporter(
            report,
            output=args.output,
            checkpoint=args.checkpoint,
            steps=args.steps,
            checkpoint_every=args.checkpoint_every,
            report_every=args.report_every,
        )

        started = time.perf_counter()
        report["training"] = train_pilot(
            model,
            optimizer,
            train_loader,
            task.entity_table,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
            contract=contract,
            checkpoint_every=args.checkpoint_every,
            on_step=progress.record_step,
            on_restore=progress.record_restore,
        )
        report["training"]["wall_seconds_this_attempt"] = time.perf_counter() - started
        report["trained"] = report["training"]["completed_steps"] > 0
        report["checkpoint"].update(
            {
                "sha256": file_sha256(args.checkpoint),
                "completed_steps": report["training"]["completed_steps"],
            }
        )
        print(
            "Evaluating the pilot checkpoint and checking all three execution modes...", flush=True
        )
        report["validation_after"] = evaluate(
            model,
            val_loader,
            task.entity_table,
            batch_size=args.batch_size,
            seed=args.seed + 1_000_000,
            device=args.device,
        )
        report["trained_mode_equivalence"] = trained_equivalence(
            model,
            val_loader,
            task.entity_table,
            args.batch_size,
            args.seed + 2_000_000,
            args.device,
        )
        synchronize(args.device)
        report["host_memory"]["after_training"] = host_memory()
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(args.output, report)
        raise
    write_json(args.output, report)
    print(
        f"Saved {args.output}; bounded training pilot, not final predictive evaluation.", flush=True
    )


if __name__ == "__main__":
    main()
