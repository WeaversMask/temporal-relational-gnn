"""Parity on actual relational batches with an explicitly untrained model."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from relbench import load_dataset
from relbench.base import Table
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch import nn
from torch_geometric.loader import NeighborLoader

from trgnn.artifacts import environment, write_json
from trgnn.execution import MODES, run_layers, validate_sampled_inputs
from trgnn.preflight import temporal_sampler_smoke
from trgnn.real_data import (
    audit_sampled_batch,
    download_inputs,
    load_specs,
    manifest_hashes,
    projected_database,
    select_queries,
    verify_inputs,
)


class RealBatchModel(nn.Module):
    """RelBench encoders/GraphSAGE plus a linear head; no trained weights yet."""

    def __init__(self, data, col_stats, channels=64, num_layers=2):
        super().__init__()
        self.channels = channels
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={name: data[name].tf.col_names_dict for name in data.node_types},
            node_to_col_stats=col_stats,
        )
        self.temporal_encoder = HeteroTemporalEncoder(data.node_types, channels)
        self.gnn = HeteroGraphSAGE(
            data.node_types, data.edge_types, channels, aggr="mean", num_layers=num_layers
        )
        self.head = nn.Linear(channels, 1)

    def encode(self, batch, entity):
        """Encode the same occurrence features for every execution schedule."""
        # Preserve zero-node types. Some table encoders cannot reshape a zero batch.
        features = {
            name: self.encoder.encoders[name](frame)
            if frame.num_rows
            else self.head.weight.new_empty((0, self.channels))
            for name, frame in batch.tf_dict.items()
        }
        temporal = self.temporal_encoder(batch[entity].seed_time, batch.time_dict, batch.batch_dict)
        return {name: value + temporal[name] for name, value in features.items()}

    def message_passing(self, features, batch, mode="reference"):
        if mode == "reference":
            result = self.gnn(
                features,
                batch.edge_index_dict,
                batch.num_sampled_nodes_dict,
                batch.num_sampled_edges_dict,
            )
        else:
            result = run_layers(
                self.gnn.convs,
                self.gnn.norms,
                features,
                batch.edge_index_dict,
                batch.num_sampled_nodes_dict,
                batch.num_sampled_edges_dict,
                mode,
            )
        return result

    def readout(self, result, batch, entity):
        return self.head(result[entity][: batch[entity].seed_time.numel()]).flatten()

    def forward(self, batch, entity, mode="reference"):
        features = self.encode(batch, entity)
        result = self.message_passing(features, batch, mode)
        return self.readout(result, batch, entity)


def check_and_measure(model, batch, entity, repeats, device):
    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    record = {"max_absolute_logit_error": {}, "timings_ms": {mode: [] for mode in MODES}}
    with torch.inference_mode():
        reference = model(batch, entity, "reference")
        reference_loss = nn.functional.binary_cross_entropy_with_logits(
            reference, batch[entity].y.to(reference.dtype)
        )
        for mode in MODES:
            candidate = model(batch, entity, mode)
            torch.testing.assert_close(candidate, reference, atol=1e-5, rtol=1e-4)
            loss = nn.functional.binary_cross_entropy_with_logits(
                candidate, batch[entity].y.to(candidate.dtype)
            )
            torch.testing.assert_close(loss, reference_loss, atol=1e-5, rtol=1e-4)
            record["max_absolute_logit_error"][mode] = float((candidate - reference).abs().max())
            for _ in range(3):
                model(batch, entity, mode)
        order = random.Random(42)
        for _ in range(repeats):
            modes = list(MODES)
            order.shuffle(modes)
            for mode in modes:
                synchronize()
                started = time.perf_counter_ns()
                model(batch, entity, mode)
                synchronize()
                record["timings_ms"][mode].append((time.perf_counter_ns() - started) / 1e6)
        record["median_ms"] = {
            mode: statistics.median(values) for mode, values in record["timings_ms"].items()
        }
        record["extra_peak_allocated_cuda_bytes"] = {}
        for mode in MODES:
            if device == "cuda":
                synchronize()
                initial = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                model(batch, entity, mode)
                synchronize()
                extra = torch.cuda.max_memory_allocated() - initial
            else:
                extra = None
            record["extra_peak_allocated_cuda_bytes"][mode] = extra
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/real-batch.json"))
    args = parser.parse_args()
    if min(args.batch_size, args.batches, args.channels) < 1 or args.repeats < 3:
        parser.error("Positive dimensions and at least three timing repetitions are required")
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
        "scope": "untrained_real_relational_batch_forward",
        "trained": False,
        "end_to_end": False,
        "dataset": pins["dataset"],
        "environment": environment(),
        "batches": [],
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "notes": [
            "All tables/relationships retained; node features are constant plus relative time.",
            "Uses one random-weight model and identical sampled batches across execution modes.",
            "Forward timing includes feature/time encoders, message passing, and readout.",
            "Excludes sampling, feature gathering, transfer, graph building, and download.",
            "Task horizon is 91 days; the prose description's two-year claim is inconsistent.",
            "No predictive-quality, trained-model, or complete inference speedup claim.",
        ],
    }
    write_json(args.output, report)
    try:
        print("Checking compiled temporal sampler before downloading data...", flush=True)
        report["sampler_preflight"] = temporal_sampler_smoke()
        if args.download:
            download_inputs(args.data_dir, pins)
        verify_inputs(args.data_dir, pins)
        _, task_spec = load_specs(args.data_dir)
        selected, selected_ids = select_queries(
            args.data_dir, args.batch_size * args.batches, args.seed
        )
        report["validation_row_ids"] = selected_ids
        cutoff = selected[task_spec["time_col"]].max()
        database, audit = projected_database(args.data_dir, cutoff)
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
        print("Building the projected heterogeneous graph...", flush=True)
        graph, stats = make_pkey_fkey_graph(database, {name: {} for name in database.table_dict})
        del database
        dataset = load_dataset(args.data_dir)
        task = dataset.load_task("user-badge")
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
        )
        report["loader_construction_seconds"] = time.perf_counter() - started
        model = (
            RealBatchModel(graph, stats, args.channels, len(args.fanouts)).to(args.device).eval()
        )
        iterator = iter(loader)
        for index in range(args.batches):
            started = time.perf_counter()
            batch = next(iterator)
            sample_seconds = time.perf_counter() - started
            visibility = audit_sampled_batch(batch, entity)
            shapes = {name: torch.empty((batch[name].num_nodes, 0)) for name in batch.node_types}
            validate_sampled_inputs(
                shapes,
                batch.edge_index_dict,
                batch.num_sampled_nodes_dict,
                batch.num_sampled_edges_dict,
                len(args.fanouts),
            )
            counts = {
                "nodes": batch.num_sampled_nodes_dict,
                "edges": {"|".join(k): v for k, v in batch.num_sampled_edges_dict.items()},
            }
            batch = batch.to(args.device)
            measurements = check_and_measure(model, batch, entity, args.repeats, args.device)
            report["batches"].append(
                {
                    "index": index,
                    "sampling_and_gather_seconds": sample_seconds,
                    "visibility_audit": visibility,
                    "sampled_counts": counts,
                    **measurements,
                }
            )
            write_json(args.output, report)
            print(
                f"Batch {index + 1}/{args.batches}: parity passed; {measurements['median_ms']}",
                flush=True,
            )
        report["status"] = "passed"
        write_json(args.output, report)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(args.output, report)
        raise
    print(f"Saved {args.output}. Untrained real-graph diagnostic only.", flush=True)


if __name__ == "__main__":
    main()
