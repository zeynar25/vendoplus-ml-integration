"""
Tests for train.py (step 7) and evaluate.py (step 8): selection rules,
baseline rules and metrics. These use small made-up inputs and never touch
the test partition.

Run from the project folder:
    python -m pytest -v
"""

import json

import numpy as np
import pytest

import config
import evaluate
import train


def row(algorithm, params, macro_f1, high_recall, high_precision):
    return {"algorithm": algorithm, "params": params, "macro_f1": macro_f1,
            "high_recall": high_recall, "high_precision": high_precision}


# ---- Selection rules (D-11, D-12) ------------------------------------------------

def test_highest_macro_f1_wins():
    rows = [row("random_forest", {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5}, 0.70, 0.9, 0.4),
            row("random_forest", {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 20}, 0.71, 0.5, 0.3)]
    assert train.choose_best_configuration(rows)["macro_f1"] == 0.71


def test_tie_broken_by_high_recall_then_precision_then_size():
    rf = "random_forest"
    small = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 20}
    big = {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 5}
    assert train.choose_best_configuration([row(rf, big, 0.7, 0.8, 0.4), row(rf, small, 0.7, 0.9, 0.4)])["params"] == small
    assert train.choose_best_configuration([row(rf, small, 0.7, 0.9, 0.3), row(rf, big, 0.7, 0.9, 0.4)])["params"] == big
    assert train.choose_best_configuration([row(rf, big, 0.7, 0.9, 0.4), row(rf, small, 0.7, 0.9, 0.4)])["params"] == small


def test_smaller_model_definitions():
    assert train.size_key("logistic_regression", {"C": 0.1}) < train.size_key("logistic_regression", {"C": 10})
    assert train.size_key("hist_gradient_boosting", {"max_leaf_nodes": 15}) < \
        train.size_key("hist_gradient_boosting", {"max_leaf_nodes": 31})
    # Fewer trees first, then shallower, then LARGER leaf size.
    assert train.size_key("random_forest", {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 5}) < \
        train.size_key("random_forest", {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20})
    assert train.size_key("random_forest", {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 20}) < \
        train.size_key("random_forest", {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5})


def test_final_model_prefers_meeting_both_targets():
    meets = row("logistic_regression", {"C": 1}, 0.81, 0.92, 0.5)
    higher_f1_but_low_recall = row("random_forest", {}, 0.85, 0.70, 0.6)
    assert train.choose_final_model([higher_f1_but_low_recall, meets]) is meets
    # If none meets both targets, the highest macro F1 wins.
    a, b = row("logistic_regression", {"C": 1}, 0.70, 0.8, 0.4), row("random_forest", {}, 0.72, 0.7, 0.4)
    assert train.choose_final_model([a, b]) is b


# ---- Baselines --------------------------------------------------------------------

def test_stock_only_baseline_thresholds_in_grams():
    assert evaluate.stock_only_baseline([49.9, 50.0, 50.1, 200.0, 200.1]).tolist() == \
        ["High", "High", "Medium", "Medium", "Low"]


def test_consumption_baseline():
    # 40 g left at 10 g/h -> 4 h (High); 41 g -> 4.1 h (Medium); 240 g -> 24 h (Medium);
    # 241 g -> 24.1 h (Low); no use in the last 24 h -> Low.
    rate = 10 / config.HOPPER_CAPACITY_G  # 10 g per hour as a share of capacity
    stock = [40, 41, 240, 241, 10]
    rates = [rate, rate, rate, rate, 0.0]
    assert evaluate.consumption_baseline(stock, rates).tolist() == ["High", "Medium", "Medium", "Low", "Low"]


# ---- Metrics ----------------------------------------------------------------------

def test_metrics_by_hand():
    truth = ["Low", "Low", "Medium", "High", "High"]
    pred = ["Low", "Medium", "Medium", "High", "Low"]
    m = train.classification_metrics(truth, pred)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["high_recall"] == pytest.approx(1 / 2)
    assert m["high_precision"] == pytest.approx(1.0)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]  # rows: true Low, Medium, High
    expected_macro = np.mean([2 * 0.5 * 0.5 / 1.0, 2 * 0.5 * 1 / 1.5, 2 * 1 * 0.5 / 1.5])
    assert m["macro_f1"] == pytest.approx(expected_macro, abs=1e-6)


# ---- Real selection file (skipped until train.py has run) --------------------------

@pytest.mark.skipif(not config.SELECTION_PATH.exists(), reason="Run train.py first.")
def test_real_selection_used_validation_only():
    with open(config.SELECTION_PATH, encoding="utf-8") as f:
        selection = json.load(f)
    assert selection["selected_using"].startswith("validation")
    assert set(selection["kept_per_algorithm"]) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert selection["feature_columns_in_order"] == config.FEATURE_COLUMNS
    assert selection["final_model"] in selection["kept_per_algorithm"]
