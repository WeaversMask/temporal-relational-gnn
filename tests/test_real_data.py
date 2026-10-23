import pandas as pd
import pytest
import torch
import yaml
from torch_geometric.typing import WITH_PYG_LIB

from trgnn.real_data import normalized_time, projected_database, select_queries


@pytest.fixture
def local_database(tmp_path):
    root = tmp_path / "rel-stack"
    (root / "db").mkdir(parents=True)
    (root / "tasks/user-badge").mkdir(parents=True)
    manifest = {
        "name": "rel-stack",
        "manifest_version": 1,
        "val_timestamp": "2020-10-01",
        "test_timestamp": "2021-01-01",
        "tables": {
            "users": {"pkey": "Id", "time_col": "Created", "fkeys": {}},
            "events": {"pkey": "Id", "time_col": "Created", "fkeys": {"UserId": "users"}},
        },
    }
    task = {
        "name": "user-badge",
        "kind": "forecast",
        "task_type": "binary_classification",
        "entity_table": "users",
        "entity_col": "UserId",
        "target_col": "Target",
        "time_col": "timestamp",
        "timedelta": "91 days",
        "sql": (
            "SELECT t.timestamp, u.Id AS UserId, 0 AS Target "
            "FROM timestamps t CROSS JOIN users u WHERE u.Created <= t.timestamp"
        ),
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    (root / "tasks/user-badge/manifest.yaml").write_text(yaml.safe_dump(task))
    pd.DataFrame(
        {
            "Id": [0, 1],
            "Created": pd.to_datetime(["2020-01-01", "2020-11-01"]),
            "IgnoredText": ["do not load", "do not load"],
        }
    ).to_parquet(root / "db/users.parquet")
    pd.DataFrame(
        {"Id": [0, 1], "Created": pd.to_datetime(["2020-02-01", "2020-03-01"]), "UserId": [0, 1]}
    ).to_parquet(root / "db/events.parquet")
    pd.DataFrame(
        {"UserId": [0, 0, 0], "timestamp": pd.to_datetime(["2020-10-01"] * 3), "Target": [0, 1, 0]}
    ).to_parquet(root / "tasks/user-badge/val.parquet")
    return root


def test_projection_censors_rows_and_dangling_keys(local_database):
    database, audit = projected_database(local_database, pd.Timestamp("2020-10-01"))
    assert len(database.table_dict["users"]) == 1
    assert "IgnoredText" not in database.table_dict["users"].df
    assert pd.isna(database.table_dict["events"].df.loc[1, "UserId"])
    assert audit["events"]["dangling_foreign_keys_removed"]["UserId"] == 1


def test_query_selection_does_not_depend_on_labels(local_database):
    original, indices = select_queries(local_database, 2, 42)
    path = local_database / "tasks/user-badge/val.parquet"
    frame = pd.read_parquet(path)
    frame["Target"] = 1 - frame["Target"]
    frame.to_parquet(path)
    changed, changed_indices = select_queries(local_database, 2, 42)
    assert indices == changed_indices
    assert original["UserId"].equals(changed["UserId"])


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
def test_timestamp_units_are_explicit(unit):
    series = pd.Series(pd.to_datetime(["2020-10-01"]).as_unit(unit))
    result = normalized_time(series)
    assert str(result.dtype) == "datetime64[ns]"
    assert int(result.astype("int64").iloc[0] // 1_000_000_000) == 1601510400


def test_unknown_record_time_needs_explicit_policy():
    with pytest.raises(ValueError, match="Missing timestamps"):
        normalized_time(pd.Series([pd.NaT]))


def test_local_task_loads_without_training_or_test_tables(local_database):
    from relbench import load_dataset
    from relbench.base import Table
    from relbench.modeling.graph import get_node_train_table_input

    dataset = load_dataset(local_database)
    task = dataset.load_task("user-badge")
    selected, _ = select_queries(local_database, 2, 42)
    query_table = Table(selected, {task.entity_col: task.entity_table}, time_col=task.time_col)
    inputs = get_node_train_table_input(query_table, task)
    assert inputs.nodes[1].tolist() == [0, 0]
    assert inputs.time.tolist() == [1601510400, 1601510400]


@pytest.mark.skipif(not WITH_PYG_LIB, reason="Requires the compiled temporal sampler")
def test_real_pipeline_on_small_local_relational_fixture(local_database):
    from relbench.modeling.graph import make_pkey_fkey_graph
    from torch_geometric.loader import NeighborLoader

    from trgnn.execution import MODES, validate_sampled_inputs
    from trgnn.real_batch import RealBatchModel, check_and_measure
    from trgnn.real_data import audit_sampled_batch

    torch.set_num_threads(1)
    database, _ = projected_database(local_database, pd.Timestamp("2020-10-01"))
    graph, stats = make_pkey_fkey_graph(database, {name: {} for name in database.table_dict})
    loader = NeighborLoader(
        graph,
        input_nodes=("users", torch.tensor([0, 0])),
        input_time=torch.tensor([1583020800, 1601510400]),
        time_attr="time",
        num_neighbors=[2, 2],
        disjoint=True,
        subgraph_type="directional",
        batch_size=2,
        num_workers=0,
    )
    batch = next(iter(loader))
    batch["users"].y = torch.tensor([0.0, 1.0])
    assert audit_sampled_batch(batch, "users")["status"] == "passed"
    features = {name: torch.empty((batch[name].num_nodes, 0)) for name in batch.node_types}
    validate_sampled_inputs(
        features,
        batch.edge_index_dict,
        batch.num_sampled_nodes_dict,
        batch.num_sampled_edges_dict,
        2,
    )
    model = RealBatchModel(graph, stats, channels=8).eval()
    with torch.inference_mode():
        reference = model(batch, "users", "reference")
        for mode in MODES:
            torch.testing.assert_close(model(batch, "users", mode), reference)
    result = check_and_measure(model, batch, "users", repeats=3, device="cpu")
    assert len(result["timings_ms"]["blocks"]) == 3
