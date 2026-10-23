"""Synthetic, model-core feasibility benchmark. Not an end-to-end result."""

import argparse
import random
import statistics
import time
from pathlib import Path

import torch

from trgnn.artifacts import environment, write_json
from trgnn.execution import MODES, RelationalGraphSAGE, seed_outputs
from trgnn.synthetic import EDGE_TYPES, NODE_TYPES, make_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--fanouts", nargs="+", type=int, default=[4, 2])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic.json"))
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup < 0 or args.threads < 1:
        parser.error("Use at least 3 repeats, non-negative warmup, and positive threads")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but no CUDA device is available")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    batch = make_batch(args.batch_size, args.channels, tuple(args.fanouts), args.seed).to(device)
    model = RelationalGraphSAGE(NODE_TYPES, EDGE_TYPES, args.channels, len(args.fanouts))
    model = model.to(device).eval()

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    def execute(mode):
        return model(*batch.inputs(), mode=mode)

    raw_timings = {mode: [] for mode in MODES}
    errors = {}
    peak_extra_bytes = {}
    with torch.inference_mode():
        reference = seed_outputs(execute("reference"), batch.node_counts)
        for mode in MODES:
            outputs = seed_outputs(execute(mode), batch.node_counts)
            torch.testing.assert_close(outputs, reference, atol=1e-5, rtol=1e-4)
            errors[mode] = max(float((outputs[k] - reference[k]).abs().max()) for k in outputs)
            del outputs
            for _ in range(args.warmup):
                execute(mode)
        synchronize()
        order = random.Random(args.seed)
        execution_order = []
        for _ in range(args.repeats):
            modes = list(MODES)
            order.shuffle(modes)
            execution_order.append(modes)
            for mode in modes:
                synchronize()
                started = time.perf_counter_ns()
                execute(mode)
                synchronize()
                raw_timings[mode].append((time.perf_counter_ns() - started) / 1_000_000)
        for mode in MODES:
            if device == "cuda":
                synchronize()
                baseline_bytes = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                execute(mode)
                synchronize()
                peak_extra_bytes[mode] = torch.cuda.max_memory_allocated() - baseline_bytes
            else:
                peak_extra_bytes[mode] = None
    medians = {mode: statistics.median(values) for mode, values in raw_timings.items()}
    record = {
        "scope": "synthetic_preencoded_message_passing_only",
        "trained": False,
        "end_to_end": False,
        "notes": [
            "Includes Python block construction, convolution, node normalization, and ReLU.",
            "Excludes sampling, feature gathering, transfer, encoders, and prediction head.",
            "Synthetic occurrence features are not predictions on RelBench data.",
            "CPU results do not establish GPU acceleration or real-workload performance.",
        ],
        "config": {**vars(args), "output": str(args.output), "resolved_device": device},
        "environment": environment(),
        "node_counts": batch.node_counts,
        "edge_counts": {"|".join(k): v for k, v in batch.edge_counts.items()},
        "equivalence": {"atol": 1e-5, "rtol": 1e-4, "max_absolute_error": errors},
        "timings_ms": raw_timings,
        "execution_order": execution_order,
        "median_ms": medians,
        "extra_peak_allocated_cuda_bytes": peak_extra_bytes,
        "blocks_latency_reduction_vs_hgam": 1 - medians["blocks"] / medians["hgam"],
    }
    write_json(args.output, record)
    print(f"Saved {args.output}; scope={record['scope']}; device={device}")
    for mode in MODES:
        print(f"{mode}: {medians[mode]:.3f} ms, max seed error={errors[mode]:.3g}")
    print("Preliminary synthetic measurements only; no real-data or end-to-end claim.")


if __name__ == "__main__":
    main()
