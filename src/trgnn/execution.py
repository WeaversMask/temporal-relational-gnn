"""Three execution schedules over identical directed neighbor-sampled batches.

The layers intentionally match RelBench's GraphSAGE configuration. HGAM is reused
from PyG; the block path supplies bipartite feature pairs to the same operators.
Inputs are already encoded, including any temporal features. This module does
not sample neighbors, merge global IDs, or claim to introduce block execution.
"""

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.nn import HeteroConv, LayerNorm, SAGEConv
from torch_geometric.utils import trim_to_layer

EdgeType = tuple[str, str, str]
Mode = Literal["reference", "hgam", "blocks"]
MODES: tuple[Mode, ...] = ("reference", "hgam", "blocks")


def validate_sampled_inputs(
    x_dict: Mapping[str, Tensor],
    edge_index_dict: Mapping[EdgeType, Tensor],
    node_counts: Mapping[str, Sequence[int]],
    edge_counts: Mapping[EdgeType, Sequence[int]],
    num_layers: int,
) -> None:
    """Validate the BFS prefix contract outside timed execution.

    Only directed sampling is supported. Counts describe per-hop *occurrences*,
    not unique global IDs. Bidirectional or induced subgraphs are unsupported.
    """
    if set(x_dict) != set(node_counts):
        raise ValueError("Every feature type must have per-hop node counts")
    if set(edge_index_dict) != set(edge_counts):
        raise ValueError("Every supplied relation must have per-hop edge counts")
    for name, counts in node_counts.items():
        if len(counts) != num_layers + 1 or any(n < 0 for n in counts):
            raise ValueError(f"Invalid node counts for {name}")
        if sum(counts) != x_dict[name].size(0):
            raise ValueError(f"Node counts disagree with features for {name}")
    for relation, edges in edge_index_dict.items():
        counts = edge_counts[relation]
        if len(counts) != num_layers or any(n < 0 for n in counts):
            raise ValueError(f"Invalid edge counts for {relation}")
        if edges.dtype != torch.long or edges.ndim != 2 or edges.size(0) != 2:
            raise ValueError(f"Expected int64 COO edges for {relation}")
        if sum(counts) != edges.size(1):
            raise ValueError(f"Edge counts disagree with edges for {relation}")
        if relation[0] not in x_dict or relation[2] not in x_dict:
            raise ValueError(f"Unknown endpoint type in {relation}")
        if edges.numel() and bool((edges < 0).any()):
            raise ValueError(f"Negative local node index in {relation}")
        for remaining in range(1, num_layers + 1):
            prefix = edges[:, : sum(counts[:remaining])]
            src_size = sum(node_counts[relation[0]][: remaining + 1])
            dst_size = sum(node_counts[relation[2]][:remaining])
            if prefix.numel() and (
                int(prefix[0].max()) >= src_size or int(prefix[1].max()) >= dst_size
            ):
                raise ValueError(f"Edges violate directed BFS prefix contract: {relation}")


def run_layers(
    convs: Sequence[nn.Module],
    norms: Sequence[nn.ModuleDict],
    x_dict: dict[str, Tensor],
    edge_index_dict: dict[EdgeType, Tensor],
    node_counts: dict[str, list[int]],
    edge_counts: dict[EdgeType, list[int]],
    mode: Mode,
) -> dict[str, Tensor]:
    """Execute existing layers without changing their parameters or operators.

    Pass RelBench's ``gnn.convs`` and ``gnn.norms`` here for real-model integration.
    Output shapes differ by schedule; supervised seed prefixes must agree.
    All relationship keys, including zero-edge relationships, are preserved.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown execution mode: {mode}")
    if not convs or len(convs) != len(norms):
        raise ValueError("Expected equally sized non-empty convolution and norm lists")
    for layer, (conv, norm_dict) in enumerate(zip(convs, norms, strict=True)):
        if mode == "hgam":
            x_dict, edge_index_dict, _ = trim_to_layer(
                layer, node_counts, edge_counts, x_dict, edge_index_dict
            )
        if mode == "blocks":
            remaining = len(convs) - layer
            destinations = {name: sum(counts[:remaining]) for name, counts in node_counts.items()}
            edges = {
                relation: index[:, : sum(edge_counts[relation][:remaining])]
                for relation, index in edge_index_dict.items()
            }
            # HeteroConv accepts edge-type keyed inputs. This also handles
            # same-table relationships without conflating source/dest shapes.
            pairs = {
                relation: (
                    x_dict[relation[0]],
                    x_dict[relation[2]][: destinations[relation[2]]],
                )
                for relation in edges
            }
            x_dict = conv(pairs, edges)
        else:
            x_dict = conv(x_dict, edge_index_dict)
        x_dict = {name: norm_dict[name](x).relu() for name, x in x_dict.items()}
    return x_dict


def seed_outputs(
    outputs: Mapping[str, Tensor], node_counts: Mapping[str, Sequence[int]]
) -> dict[str, Tensor]:
    return {
        name: outputs[name][: counts[0]] for name, counts in node_counts.items() if counts[0] > 0
    }


class RelationalGraphSAGE(nn.Module):
    """Small standalone harness mirroring RelBench's message-passing layers."""

    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[EdgeType],
        channels: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.convs = nn.ModuleList(
            HeteroConv(
                {
                    relation: SAGEConv((channels, channels), channels, aggr="mean")
                    for relation in edge_types
                },
                aggr="sum",
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.ModuleDict({name: LayerNorm(channels, mode="node") for name in node_types})
            for _ in range(num_layers)
        )

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        node_counts: dict[str, list[int]],
        edge_counts: dict[EdgeType, list[int]],
        mode: Mode = "reference",
    ) -> dict[str, Tensor]:
        return run_layers(
            self.convs, self.norms, x_dict, edge_index_dict, node_counts, edge_counts, mode
        )
