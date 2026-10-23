"""Paired, untrained online inference profiles on a resident relational graph.

Complete batch latency is measured without intermediate synchronizations. A
separate replay measures individual stages; those stage timings must not be
added together and presented as the uninstrumented latency.
"""

import argparse
import hashlib
import json
import platform
import random
import resource
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from relbench import load_dataset
from relbench.base import Table
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from torch import nn
from torch_geometric.loader import NeighborLoader

from trgnn.artifacts import environment, write_json
from trgnn.execution import MODES, validate_sampled_inputs
from trgnn.feature_audit import audit_feature_availability
from trgnn.preflight import temporal_sampler_smoke
from trgnn.real_batch import RealBatchModel
from trgnn.real_data import (
    audit_sampled_batch,
    download_inputs,
    load_specs,
    manifest_hashes,
    projected_database,
    select_queries,
    verify_inputs,
)

STAGES = ("sampling", "feature_gather", "transfer", "encoding", "message_passing", "readout")


def host_memory() -> dict:
    """A process-lifetime high-water mark, not a per-stage memory delta."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "process_lifetime_peak_rss_bytes": int(
            peak if platform.system() == "Darwin" else peak * 1024
        ),
        "scope": "process lifetime, including imports and one-time graph preparation",
    }


def sampler_fingerprint(sample) -> str:
    """Hash local occurrences, ownership, edges, hop counts, and seed metadata.

    Global node IDs alone cannot establish equality of temporal sampled batches.
    This is computed outside every timing region on the CPU sampler output.
    """
    digest = hashlib.sha256()

    def update(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("Expected a CPU sampler output")
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                update(key)
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(str((type(value).__name__, len(value))).encode())
            for item in value:
                update(item)
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")

    for field in ("node", "row", "col", "edge", "batch", "num_sampled_nodes", "num_sampled_edges"):
        update(field)
        update(getattr(sample, field))
    update(sample.metadata)
    return digest.hexdigest()


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


@dataclass
class PipelineResult:
    sample: object
    batch: object
    logits: torch.Tensor
    elapsed_ms: float
    stage_ms: dict
    cuda_memory: dict | None


def execute_pipeline(loader, indices, model, entity, mode, device, *, stages=False):
    """Use the same sampler/filter operations as a single-worker NodeLoader.

    Index slicing is inside sampling. DataLoader worker scheduling and collation
    overhead are outside this explicit serial online boundary.
    """
    timings = {}

    def stage(name, operation):
        if not stages:
            return operation()
        synchronize(device)
        started = time.perf_counter_ns()
        value = operation()
        synchronize(device)
        timings[name] = (time.perf_counter_ns() - started) / 1e6
        return value

    synchronize(device)
    initial = None
    reserved_before = None
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
    synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
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
    return PipelineResult(sample, batch, logits, elapsed_ms, timings, memory)


def validate_batch(batch, entity, layers):
    visibility = audit_sampled_batch(batch, entity)
    shapes = {name: torch.empty((batch[name].num_nodes, 0)) for name in batch.node_types}
    validate_sampled_inputs(
        shapes,
        batch.edge_index_dict,
        batch.num_sampled_nodes_dict,
        batch.num_sampled_edges_dict,
        layers,
    )
    return visibility


def profile_batch(loader, indices, model, entity, *, repeats=10, warmup=2, seed=42, device="cpu"):
    """Resample identically for each paired mode and validate outside timing."""
    if repeats < 3 or warmup < 0:
        raise ValueError("Use at least three repeats and non-negative warmup")
    if loader.num_workers != 0:
        raise ValueError("The serial profile requires num_workers=0")
    layers = len(model.gnn.convs)
    record = {
        "trials": [],
        "max_absolute_logit_error": {mode: 0.0 for mode in MODES},
        "max_absolute_loss_error": {mode: 0.0 for mode in MODES},
        "sampler_pairing": "CPU torch RNG reset per paired pass; full sampler fingerprints checked",
    }
    order_rng = random.Random(seed)
    with torch.inference_mode():
        for _ in range(warmup):
            for mode in MODES:
                result = execute_pipeline(loader, indices, model, entity, mode, device)
                del result
        for repeat in range(repeats):
            trial_seed = seed + repeat
            order = list(MODES)
            order_rng.shuffle(order)
            trial = {
                "sampler_seed": trial_seed,
                "execution_order": order,
                "pipeline_ms": {},
                "stage_ms": {},
                "cuda_memory": {},
            }
            paired_logits, paired_losses = {}, {}
            expected_fingerprint = None
            for mode in order:
                # Seeding and all correctness checks are outside timing.
                torch.manual_seed(trial_seed)
                result = execute_pipeline(loader, indices, model, entity, mode, device)
                fingerprint = sampler_fingerprint(result.sample)
                if expected_fingerprint is not None and fingerprint != expected_fingerprint:
                    raise AssertionError(
                        "Paired execution modes received different sampled occurrences"
                    )
                expected_fingerprint = fingerprint
                visibility = validate_batch(result.batch, entity, layers)
                logits = result.logits.detach().cpu().clone()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, result.batch[entity].y.detach().cpu().to(logits.dtype)
                )
                paired_logits[mode], paired_losses[mode] = logits, loss
                trial["pipeline_ms"][mode] = result.elapsed_ms
                trial["cuda_memory"][mode] = result.cuda_memory
                if "sampled_counts" not in trial:
                    trial["sampled_counts"] = {
                        "nodes": result.batch.num_sampled_nodes_dict,
                        "edges": {
                            "|".join(key): value
                            for key, value in result.batch.num_sampled_edges_dict.items()
                        },
                    }
                    trial["visibility_audit"] = visibility
                    trial["seed_times_unix_seconds"] = result.batch[entity].seed_time.cpu().tolist()
                del result
                torch.manual_seed(trial_seed)
                result = execute_pipeline(loader, indices, model, entity, mode, device, stages=True)
                if sampler_fingerprint(result.sample) != expected_fingerprint:
                    raise AssertionError("Stage replay received different sampled occurrences")
                torch.testing.assert_close(result.logits.cpu(), logits, atol=1e-5, rtol=1e-4)
                trial["stage_ms"][mode] = result.stage_ms
                del result
            for mode in MODES:
                torch.testing.assert_close(
                    paired_logits[mode], paired_logits["reference"], atol=1e-5, rtol=1e-4
                )
                torch.testing.assert_close(
                    paired_losses[mode], paired_losses["reference"], atol=1e-5, rtol=1e-4
                )
                record["max_absolute_logit_error"][mode] = max(
                    record["max_absolute_logit_error"][mode],
                    float((paired_logits[mode] - paired_logits["reference"]).abs().max()),
                )
                record["max_absolute_loss_error"][mode] = max(
                    record["max_absolute_loss_error"][mode],
                    float((paired_losses[mode] - paired_losses["reference"]).abs()),
                )
            trial["sampler_sha256"] = expected_fingerprint
            record["trials"].append(trial)
    record["median_pipeline_ms"] = {
        mode: statistics.median(trial["pipeline_ms"][mode] for trial in record["trials"])
        for mode in MODES
    }
    record["median_stage_ms"] = {
        mode: {
            stage: statistics.median(trial["stage_ms"][mode][stage] for trial in record["trials"])
            for stage in STAGES
        }
        for mode in MODES
    }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rel-stack"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--fanouts", type=int, nargs="+", default=[4, 2])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/pipeline-profile.json"))
    args = parser.parse_args()
    if min(args.batch_size, args.batches, args.channels) < 1 or args.repeats < 3 or args.warmup < 0:
        parser.error("Use positive dimensions, at least three repeats, and non-negative warmup")
    if any(n < 0 for n in args.fanouts):
        parser.error("Use bounded, non-negative fanouts")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Select a CUDA runtime or explicitly use --device cpu")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pins = json.loads((Path(__file__).resolve().parents[2] / "configs/upstream.json").read_text())
    report = {
        "status": "started",
        "scope": "untrained_real_relational_pipeline_profile",
        "trained": False,
        "end_to_end": True,
        "online_boundary": "serial sampling + gather + transfer + encoders + GNN + readout",
        "excluded_from_online_timing": [
            "download",
            "projection",
            "graph construction",
            "loader/CSC construction",
            "model initialization",
            "correctness checks",
            "DataLoader scheduling",
            "output serialization",
        ],
        "stage_timing_method": "separate identical replay, synchronized at each stage boundary",
        "notes": [
            "Uninstrumented pipeline totals are measured, not sums of isolated stage medians.",
            "The graph is resident on the CPU; the model is resident on the selected device.",
            "Constant features plus relative time; no trained or predictive-quality result.",
            "CUDA allocator peaks include transfer and forward; they are not total device VRAM.",
            "Reserved memory includes the warm cache from prior modes and stage replays.",
            "Host RSS is a process-lifetime high-water mark, not per-stage or incremental use.",
        ],
        "environment": environment(),
        "dataset": pins["dataset"],
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "correctness_tolerances": {"atol": 1e-5, "rtol": 1e-4},
        "setup_seconds": {},
        "host_memory": {"initial": host_memory()},
        "batches": [],
    }
    write_json(args.output, report)
    try:
        print("Checking sampler and pinned inputs...", flush=True)
        report["sampler_preflight"] = temporal_sampler_smoke()
        started = time.perf_counter()
        if args.download:
            download_inputs(args.data_dir, pins)
        verify_inputs(args.data_dir, pins)
        report["setup_seconds"]["download_and_verify"] = time.perf_counter() - started
        started = time.perf_counter()
        report["feature_audit"] = audit_feature_availability(args.data_dir, pins)
        report["setup_seconds"]["feature_audit"] = time.perf_counter() - started
        if report["feature_audit"]["status"] != "passed":
            raise ValueError("Feature-availability audit failed; inspect the embedded audit")
        started = time.perf_counter()
        _, task_spec = load_specs(args.data_dir)
        selected, selected_ids = select_queries(
            args.data_dir, args.batch_size * args.batches, args.seed
        )
        report["queries"] = {
            "validation_row_ids": selected_ids,
            "entity_ids": selected[task_spec["entity_col"]].astype(int).tolist(),
            "timestamps": selected[task_spec["time_col"]].astype(str).tolist(),
        }
        report["setup_seconds"]["query_selection"] = time.perf_counter() - started
        cutoff = selected[task_spec["time_col"]].max()
        started = time.perf_counter()
        database, audit = projected_database(args.data_dir, cutoff)
        report["setup_seconds"]["projection"] = time.perf_counter() - started
        report["host_memory"]["after_projection"] = host_memory()
        report["graph_visibility_cutoff"] = str(cutoff)
        report["table_audit"] = audit
        report["manifest_sha256"] = manifest_hashes(args.data_dir)
        entity = task_spec["entity_table"]
        ids = selected[task_spec["entity_col"]].astype(int).to_numpy()
        table = database.table_dict[entity]
        if ((ids < 0) | (ids >= len(table))).any():
            raise ValueError("A target entity is unavailable at the graph cutoff")
        created = table.df[table.time_col].iloc[ids].reset_index(drop=True)
        if not (created <= selected[task_spec["time_col"]]).all():
            raise ValueError("A target entity did not exist at its prediction timestamp")
        started = time.perf_counter()
        print("Building the constant-feature historical graph...", flush=True)
        graph, stats = make_pkey_fkey_graph(database, {name: {} for name in database.table_dict})
        del database, table
        report["setup_seconds"]["graph_construction"] = time.perf_counter() - started
        report["host_memory"]["after_graph_construction"] = host_memory()
        task = load_dataset(args.data_dir).load_task("user-badge")
        query_table = Table(selected, {task.entity_col: entity}, time_col=task.time_col)
        inputs = get_node_train_table_input(query_table, task)
        started = time.perf_counter()
        loader = NeighborLoader(
            graph,
            input_nodes=inputs.nodes,
            input_time=inputs.time,
            transform=inputs.transform,
            time_attr="time",
            num_neighbors=args.fanouts,
            disjoint=True,
            subgraph_type="directional",
            temporal_strategy="uniform",
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            filter_per_worker=False,
        )
        report["setup_seconds"]["loader_construction"] = time.perf_counter() - started
        report["host_memory"]["after_loader_construction"] = host_memory()
        started = time.perf_counter()
        model = (
            RealBatchModel(graph, stats, args.channels, len(args.fanouts)).to(args.device).eval()
        )
        synchronize(args.device)
        report["setup_seconds"]["model_initialization"] = time.perf_counter() - started
        write_json(args.output, report)
        for index in range(args.batches):
            indices = list(range(index * args.batch_size, (index + 1) * args.batch_size))
            result = profile_batch(
                loader,
                indices,
                model,
                entity,
                repeats=args.repeats,
                warmup=args.warmup,
                seed=args.seed + index * args.repeats,
                device=args.device,
            )
            report["batches"].append({"index": index, **result})
            report["host_memory"][f"after_batch_{index}"] = host_memory()
            write_json(args.output, report)
            print(f"Batch {index + 1}/{args.batches}: {result['median_pipeline_ms']}", flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(args.output, report)
        raise
    write_json(args.output, report)
    print(f"Saved {args.output}; untrained resident-graph online pipeline only.", flush=True)


if __name__ == "__main__":
    main()
