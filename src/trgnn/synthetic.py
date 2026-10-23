"""Synthetic directed, disjoint BFS batches for implementation checks.

These are not RelBench data or samples from its temporal sampler. They include
repeated global IDs and per-query occurrences so accidental deduplication can be
tested independently of downloading a large database.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from trgnn.execution import EdgeType, validate_sampled_inputs

NODE_TYPES = ("user", "post", "badge")
EMPTY_RELATION: EdgeType = ("badge", "unused", "post")
EDGE_TYPES: tuple[EdgeType, ...] = (
    ("post", "author", "user"),
    ("user", "rev_author", "post"),
    ("badge", "owner", "user"),
    ("user", "rev_owner", "badge"),
    ("post", "parent", "post"),
    ("user", "peer", "user"),
    EMPTY_RELATION,
)


@dataclass
class SampledBatch:
    x_dict: dict[str, Tensor]
    edge_index_dict: dict[EdgeType, Tensor]
    node_counts: dict[str, list[int]]
    edge_counts: dict[EdgeType, list[int]]
    global_ids: dict[str, Tensor]
    query_owner: dict[str, Tensor]
    node_times: dict[str, Tensor]
    seed_times: Tensor

    def inputs(self):
        return self.x_dict, self.edge_index_dict, self.node_counts, self.edge_counts

    def to(self, device: str, dtype: torch.dtype = torch.float32) -> "SampledBatch":
        return SampledBatch(
            {k: x.to(device=device, dtype=dtype) for k, x in self.x_dict.items()},
            {k: x.to(device) for k, x in self.edge_index_dict.items()},
            self.node_counts,
            self.edge_counts,
            self.global_ids,
            self.query_owner,
            self.node_times,
            self.seed_times,
        )


def make_batch(
    batch_size: int = 4,
    channels: int = 8,
    fanouts: tuple[int, ...] = (2, 1),
    seed: int = 42,
) -> SampledBatch:
    if batch_size < 1 or channels < 1 or not fanouts or any(n < 0 for n in fanouts):
        raise ValueError("Invalid synthetic batch dimensions")
    generator = torch.Generator().manual_seed(seed)
    owners: dict[str, list[int]] = {name: [] for name in NODE_TYPES}
    depths: dict[str, list[int]] = {name: [] for name in NODE_TYPES}
    owners["user"] = list(range(batch_size))
    depths["user"] = [0] * batch_size
    node_counts = {name: [len(owners[name])] for name in NODE_TYPES}
    edge_counts: dict[EdgeType, list[int]] = {relation: [] for relation in EDGE_TYPES}
    edges: dict[EdgeType, list[tuple[int, int]]] = {relation: [] for relation in EDGE_TYPES}
    for hop, fanout in enumerate(fanouts):
        previous_sizes = {name: len(owners[name]) for name in NODE_TYPES}
        destinations = {
            name: [i for i, depth in enumerate(depths[name]) if depth == hop] for name in NODE_TYPES
        }
        for relation in EDGE_TYPES:
            src, _, dst = relation
            before = len(edges[relation])
            if relation != EMPTY_RELATION:
                for dst_index in destinations[dst]:
                    owner = owners[dst][dst_index]
                    # The last seed is deliberately isolated, retaining only
                    # the relationship-specific root/bias contributions.
                    if owner == batch_size - 1:
                        continue
                    for _ in range(fanout):
                        src_index = len(owners[src])
                        owners[src].append(owner)
                        depths[src].append(hop + 1)
                        edges[relation].append((src_index, dst_index))
            edge_counts[relation].append(len(edges[relation]) - before)
        for name in NODE_TYPES:
            node_counts[name].append(len(owners[name]) - previous_sizes[name])
    x_dict, global_ids, query_owner, node_times = {}, {}, {}, {}
    seed_times = 1_000_000 + torch.arange(batch_size, dtype=torch.long) * 100_000
    for name in NODE_TYPES:
        count = len(owners[name])
        global_ids[name] = torch.arange(count) % 3
        query_owner[name] = torch.tensor(owners[name], dtype=torch.long)
        depth = torch.tensor(depths[name], dtype=torch.long)
        ages = (depth + 1) * 100 + query_owner[name]
        node_times[name] = seed_times[query_owner[name]] - ages
        base = torch.randn(3, channels, generator=generator)
        # A toy pre-encoded temporal feature; the real RelBench encoder is not
        # reproduced here. Repeated IDs remain separate local occurrences.
        x_dict[name] = base[global_ids[name]] + ages[:, None].float() / 1000
    edge_index_dict = {
        relation: torch.tensor(values, dtype=torch.long).reshape(-1, 2).t().contiguous()
        for relation, values in edges.items()
    }
    validate_sampled_inputs(x_dict, edge_index_dict, node_counts, edge_counts, len(fanouts))
    return SampledBatch(
        x_dict,
        edge_index_dict,
        node_counts,
        edge_counts,
        global_ids,
        query_owner,
        node_times,
        seed_times,
    )
