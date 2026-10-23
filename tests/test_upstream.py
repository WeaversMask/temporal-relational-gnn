"""Optional integration check against the installed RelBench package."""

import importlib.util

import pytest
import torch

from trgnn.execution import MODES, RelationalGraphSAGE, seed_outputs
from trgnn.synthetic import EDGE_TYPES, NODE_TYPES, make_batch


@pytest.mark.skipif(
    importlib.util.find_spec("relbench") is None,
    reason="Install .[relbench] to check the published reference architecture",
)
def test_matches_published_relbench_graphsage():
    from relbench.modeling.nn import HeteroGraphSAGE

    torch.manual_seed(17)
    batch = make_batch().to("cpu", torch.float64)
    upstream = (
        HeteroGraphSAGE(
            node_types=list(NODE_TYPES),
            edge_types=list(EDGE_TYPES),
            channels=8,
            aggr="mean",
            num_layers=2,
        )
        .double()
        .eval()
    )
    candidate = RelationalGraphSAGE(NODE_TYPES, EDGE_TYPES, 8, 2).double().eval()
    candidate.load_state_dict(upstream.state_dict(), strict=True)
    with torch.inference_mode():
        reference = seed_outputs(upstream(*batch.inputs()), batch.node_counts)
        for mode in MODES:
            actual = seed_outputs(candidate(*batch.inputs(), mode=mode), batch.node_counts)
            torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-9)
