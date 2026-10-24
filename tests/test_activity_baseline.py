import json

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Table

from trgnn.activity_baseline import activity_features, binary_metrics, fit_activity_baseline


@pytest.fixture
def activity_inputs():
    task = {
        "entity_table": "users",
        "entity_col": "UserId",
        "time_col": "timestamp",
        "target_col": "Target",
    }
    users = Table(
        pd.DataFrame(
            {
                "Id": [0, 1],
                "Created": pd.to_datetime(["2020-01-01", "2019-01-01"]),
                "Reputation": [999, 9999],
            }
        ),
        {},
        pkey_col="Id",
        time_col="Created",
    )
    events = Table(
        pd.DataFrame(
            {
                "Id": range(9),
                "UserId": [0, 0, 0, 0, 0, 0, 0, None, 99],
                "Date": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-01-02",
                        "2020-03-31",
                        "2020-04-01",
                        "2020-04-02",
                        "2020-07-01",
                        "2020-07-02",
                        "2020-02-01",
                        "2020-02-01",
                    ]
                ),
                "Name": ["must not become a feature"] * 9,
            }
        ),
        {"UserId": "users"},
        pkey_col="Id",
        time_col="Date",
    )
    queries = pd.DataFrame(
        {
            "UserId": [0, 0, 1, 1],
            "timestamp": pd.to_datetime(["2020-04-01", "2020-07-01"] * 2),
            "Target": [0, 1, 1, 0],
        }
    )
    return Database({"users": users, "badges": events}), queries, task


def test_historical_counts_exclude_future_events_and_preserve_query_occurrences(activity_inputs):
    database, queries, task = activity_inputs
    features, names = activity_features(database, queries, task)
    assert names == [
        "account_age_days",
        "badges.UserId.lifetime_count",
        "badges.UserId.last_91_days_count",
    ]
    np.testing.assert_array_equal(features[:, 1:], [[4, 3], [6, 2], [0, 0], [0, 0]])
    np.testing.assert_array_equal(features[:2, 0], [91, 182])
    changed = queries.copy()
    changed["Target"] = 1 - changed["Target"]
    database.table_dict["users"].df["Reputation"] = -999
    database.table_dict["badges"].df["Name"] = "changed snapshot category"
    np.testing.assert_array_equal(activity_features(database, changed, task)[0], features)


def test_future_event_addition_does_not_change_historical_features(activity_inputs):
    database, queries, task = activity_inputs
    before, _ = activity_features(database, queries, task)
    events = database.table_dict["badges"]
    events.df.loc[6, "Date"] = pd.Timestamp("2040-01-01")
    after, _ = activity_features(database, queries, task)
    np.testing.assert_array_equal(before, after)


def test_baseline_scaler_and_prevalence_use_training_only(activity_inputs):
    database, queries, task = activity_inputs
    train = queries.iloc[[0, 2]].copy()
    val = queries.iloc[[1, 3]].copy()
    val["timestamp"] = pd.Timestamp("2030-01-01")
    val["Target"] = 1
    train_features, _ = activity_features(database, train, task)
    result = fit_activity_baseline(database, train, val, task)
    np.testing.assert_allclose(
        result["preprocessing"]["mean"], np.log1p(train_features).mean(axis=0)
    )
    assert result["training_prevalence_baseline"]["probability"] == 0.5
    assert result["validation_labels"] == val["Target"].tolist()
    assert len(result["validation_probabilities"]) == len(val)
    assert result["validation_metrics"]["count"] == len(val)
    assert result["prediction_source"] == "fitted_logistic_regression"
    json.dumps(result, allow_nan=False)


def test_single_class_training_has_explicit_prevalence_fallback(activity_inputs):
    database, queries, task = activity_inputs
    train = queries.iloc[:2].copy()
    train["Target"] = 0
    result = fit_activity_baseline(database, train, queries.iloc[2:], task)
    assert result["status"] == "not_fitted_single_class"
    assert result["prediction_source"] == "training_prevalence_fallback"
    assert "model" not in result
    assert result["validation_probabilities"] == [0.0, 0.0]
    json.dumps(result, allow_nan=False)


def test_single_class_metrics_are_explicit_and_finite():
    negative = binary_metrics([0, 0], [0.0, 1.0])
    assert negative["roc_auc"] is None and negative["average_precision"] is None
    assert np.isfinite(negative["binary_cross_entropy"])
    positive = binary_metrics([1, 1], [0.5, 0.5])
    assert positive["roc_auc"] is None and positive["average_precision"] == 1.0
    mixed = binary_metrics([0, 1], [0.1, 0.9])
    assert mixed["roc_auc"] == mixed["average_precision"] == 1.0


@pytest.mark.parametrize("probabilities", [[0.5], [np.nan, 0.5], [-0.1, 0.5], [0.5, 1.1]])
def test_metrics_reject_invalid_probabilities(probabilities):
    with pytest.raises(ValueError, match="aligned non-empty binary labels"):
        binary_metrics([0, 1], probabilities)
