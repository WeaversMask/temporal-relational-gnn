"""Historical activity counts under the same pinned hosted-snapshot graph contract."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from trgnn.real_data import normalized_time
from trgnn.training_data import validate_query_entities


def binary_metrics(labels, probabilities) -> dict:
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if (
        labels.ndim != 1
        or probabilities.shape != labels.shape
        or not labels.size
        or not np.isin(labels, [0, 1]).all()
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError(
            "Expected aligned non-empty binary labels and finite probabilities in [0, 1]"
        )
    positive_count = int(labels.sum())
    epsilon = np.finfo(np.float64).eps
    clipped = probabilities.clip(epsilon, 1 - epsilon)
    loss = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    limitations = []
    auc = None
    if len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, probabilities))
    else:
        limitations.append("ROC-AUC is undefined for a single-class cohort.")
    ap = None
    if positive_count:
        ap = float(average_precision_score(labels, probabilities))
    else:
        limitations.append("Average precision is undefined without positive examples.")
    return {
        "count": len(labels),
        "positive_count": positive_count,
        "positive_rate": float(labels.mean()),
        "binary_cross_entropy": float(loss),
        "bce_probability_clip_epsilon": float(epsilon),
        "roc_auc": auc,
        "average_precision": ap,
        "limitations": limitations,
    }


def activity_features(database, queries, task_spec) -> tuple[np.ndarray, list[str]]:
    """Count incoming records using sorted (entity, time) searches, not query scans.

    Counts inherit the benchmark's snapshot/survivorship limitations. No raw
    categorical, text, mutable numerical, or task-label attributes are read.
    """
    validate_query_entities(database, queries, task_spec)
    entity = task_spec["entity_table"]
    users = database.table_dict[entity]
    ids = queries[task_spec["entity_col"]].to_numpy(dtype=np.int64)
    times = normalized_time(queries[task_spec["time_col"]]).astype("int64").to_numpy()
    created = normalized_time(users.df[users.time_col]).astype("int64").to_numpy()[ids]
    features = [(times - created) / (86_400 * 1e9)]
    names = ["account_age_days"]
    window = int(91 * 86_400 * 1e9)
    pair_dtype = np.dtype([("entity", np.int64), ("time", np.int64)])
    upper = np.empty(len(ids), dtype=pair_dtype)
    lower = np.empty(len(ids), dtype=pair_dtype)
    upper["entity"], upper["time"] = ids, times
    lower["entity"], lower["time"] = ids, times - window
    for name, table in sorted(database.table_dict.items()):
        incoming = sorted(
            key for key, target in table.fkey_col_to_pkey_table.items() if target == entity
        )
        if not incoming:
            continue
        if table.time_col is None:
            raise ValueError(f"{name}: activity counts require explicit record timestamps")
        event_times = normalized_time(table.df[table.time_col]).astype("int64").to_numpy()
        for key in incoming:
            owners = table.df[key].astype("Int64").fillna(-1).to_numpy(dtype=np.int64)
            valid = (owners >= 0) & (owners < len(users))
            records = np.empty(int(valid.sum()), dtype=pair_dtype)
            records["entity"], records["time"] = owners[valid], event_times[valid]
            records.sort(order=["entity", "time"])
            starts = np.searchsorted(records["entity"], ids, side="left")
            ends = np.searchsorted(records, upper, side="right")
            recent_starts = np.searchsorted(records, lower, side="right")
            features.extend([ends - starts, ends - recent_starts])
            names.extend([f"{name}.{key}.lifetime_count", f"{name}.{key}.last_91_days_count"])
    return np.column_stack(features).astype(np.float64, copy=False), names


def fit_activity_baseline(database, train_queries, val_queries, task_spec) -> dict:
    train_x, names = activity_features(database, train_queries, task_spec)
    val_x, val_names = activity_features(database, val_queries, task_spec)
    if names != val_names:
        raise AssertionError("Training and validation activity feature schemas differ")
    train_y = train_queries[task_spec["target_col"]].to_numpy()
    val_y = val_queries[task_spec["target_col"]].to_numpy()
    prevalence = float(train_y.mean())
    constant_val = np.full(len(val_y), prevalence)
    report = {
        "status": "passed",
        "scope": "historical_activity_logistic_baseline_hosted_snapshot",
        "feature_names": names,
        "feature_definition": "account age; incoming record counts at <=t and in (t-91 days,t]",
        "preprocessing": {"transform": "log1p then StandardScaler", "fit_split": "train"},
        "limitations": [
            "Counts inherit hosted snapshot foreign-key, deletion and survivorship assumptions.",
            "No raw categorical, text, mutable snapshot or task-label features are included.",
        ],
        "training_prevalence_baseline": {
            "probability": prevalence,
            "train_metrics": binary_metrics(train_y, np.full(len(train_y), prevalence)),
            "validation_metrics": binary_metrics(val_y, constant_val),
        },
        "validation_labels": val_y.tolist(),
    }
    if len(np.unique(train_y)) < 2:
        report["status"] = "not_fitted_single_class"
        report["limitations"].append("Logistic regression was not fitted: training has one class.")
        report["prediction_source"] = "training_prevalence_fallback"
        report["train_metrics"] = report["training_prevalence_baseline"]["train_metrics"]
        report["validation_metrics"] = report["training_prevalence_baseline"]["validation_metrics"]
        report["validation_probabilities"] = constant_val.tolist()
        return report
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(np.log1p(train_x))
    scaled_val = scaler.transform(np.log1p(val_x))
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaled_train, train_y)
    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
        report["limitations"].append("Logistic regression reached its convergence iteration limit.")
    train_prob = model.predict_proba(scaled_train)[:, 1]
    val_prob = model.predict_proba(scaled_val)[:, 1]
    report["preprocessing"].update({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()})
    report["model"] = {
        "type": "LogisticRegression",
        "C": 1.0,
        "max_iter": 1000,
        "coefficient": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "iterations": model.n_iter_.tolist(),
    }
    report["train_metrics"] = binary_metrics(train_y, train_prob)
    report["validation_metrics"] = binary_metrics(val_y, val_prob)
    report["validation_probabilities"] = val_prob.tolist()
    report["prediction_source"] = "fitted_logistic_regression"
    return report
