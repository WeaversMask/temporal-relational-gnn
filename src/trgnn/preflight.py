"""Record the runtime and optionally check actual PyG temporal disjoint sampling."""

import argparse
from pathlib import Path

import torch

from trgnn.artifacts import environment, write_json


def temporal_sampler_smoke() -> dict:
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import NeighborLoader

    data = HeteroData()
    data["user"].x = torch.ones(2, 8)
    data["user"].time = torch.tensor([0, 0])
    data["event"].x = torch.ones(3, 8)
    data["event"].time = torch.tensor([2, 5, 8])
    edges = torch.tensor([[0, 1, 2], [0, 0, 0]])
    data["event", "belongs_to", "user"].edge_index = edges
    data["user", "rev_belongs_to", "event"].edge_index = edges.flip(0)
    loader = NeighborLoader(
        data,
        input_nodes=("user", torch.tensor([0, 0])),
        input_time=torch.tensor([3, 7]),
        time_attr="time",
        num_neighbors=[-1, -1],
        disjoint=True,
        temporal_strategy="uniform",
        batch_size=2,
        num_workers=0,
    )
    batch = next(iter(loader))
    for name in batch.node_types:
        visible = batch["user"].seed_time[batch[name].batch]
        if not bool((batch[name].time <= visible).all()):
            raise AssertionError(f"Future node sampled for {name}")
    event_ids = batch["event"].n_id
    owners = batch["event"].batch
    if set(event_ids[owners == 0].tolist()) != {0}:
        raise AssertionError("The earlier query must see only the first event")
    if set(event_ids[owners == 1].tolist()) != {0, 1}:
        raise AssertionError("The later query must see the first two events")
    if batch["user"].n_id[:2].tolist() != [0, 0]:
        raise AssertionError("Repeated seed occurrences must remain separate")
    return {
        "status": "passed",
        "scope": "tiny actual PyG sampler, not a RelBench task audit",
        "node_counts": batch.num_sampled_nodes_dict,
        "edge_counts": {"|".join(k): v for k, v in batch.num_sampled_edges_dict.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/preflight.json"))
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--check-sampler", action="store_true")
    args = parser.parse_args()
    report = environment()
    errors = []
    if args.require_cuda and not torch.cuda.is_available():
        errors.append("No CUDA device: select a GPU runtime before GPU experiments")
    if args.check_sampler:
        try:
            report["temporal_sampler"] = temporal_sampler_smoke()
        except Exception as exc:
            errors.append(f"Temporal sampler failed: {type(exc).__name__}: {exc}")
    else:
        report["temporal_sampler"] = {"status": "not_checked"}
    report["errors"] = errors
    report["status"] = "failed" if errors else "passed"
    write_json(args.output, report)
    print(f"Saved {args.output}; status={report['status']}; cuda={report['cuda_available']}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
