"""
evaluate.py - Pipeline step 8: baselines and the single test-set evaluation

Reads the three models kept by train.py (results/selection.json) and scores
them, together with two simple rules, on the TEST partition. Nothing is
tuned or selected here: the choices were frozen on validation by train.py.

Baselines (they use the same noisy stock reading as the models):
  Stock-only     High if measured stock <= 50 g (5%), Medium if <= 200 g (20%), else Low
  Consumption    hours left = measured stock / average use over the last 24 h (g per hour);
                 High if <= 4 h, Medium if <= 24 h, else Low (no use in 24 h -> Low)

A learned model is called useful only if its test macro F1 is higher than
BOTH baselines' (D-13).

Outputs (results/):
  test_results.csv, per_stock_level_test.csv, baselines_validation.csv,
  confusion_matrices/*.csv, test_confusion_matrices.png, test_comparison.png,
  evaluation_report.json

How to run (from the project folder, after train.py):
    python evaluate.py
"""

import json
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from pipeline_utils import file_sha256, relative_path, save_csv, save_json
from train import classification_metrics, load_partitions

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist. Gradient Boosting",
}
STOCK_BASELINE = "Stock-only baseline"
CONSUMPTION_BASELINE = "Consumption baseline"


# =============================================================================
# Block 1. The two rule-based baselines
# =============================================================================

def stock_only_baseline(measured_stock_g):
    stock = np.asarray(measured_stock_g, dtype=float)
    return np.where(stock <= config.STOCK_BASELINE_HIGH_MAX_G, "High",
                    np.where(stock <= config.STOCK_BASELINE_MEDIUM_MAX_G, "Medium", "Low"))


def consumption_baseline(measured_stock_g, rate_24h):
    """rate_24h is a share of capacity per hour, so x capacity = grams per hour."""
    stock = np.asarray(measured_stock_g, dtype=float)
    grams_per_hour = np.asarray(rate_24h, dtype=float) * config.HOPPER_CAPACITY_G
    safe_rate = np.where(grams_per_hour > 0, grams_per_hour, 1.0)  # avoid dividing by zero
    hours_left = np.where(grams_per_hour > 0, stock / safe_rate, np.inf)
    return np.where(hours_left <= config.HIGH_HORIZON_HOURS, "High",
                    np.where(hours_left <= config.LABEL_HORIZON_HOURS, "Medium", "Low"))


# =============================================================================
# Block 2. Predictions and scores for every method
# =============================================================================

def predict_all(part, models):
    X = part[config.FEATURE_COLUMNS]
    predictions = {MODEL_NAMES[alg]: model.predict(X) for alg, model in models.items()}
    predictions[STOCK_BASELINE] = stock_only_baseline(part["measured_stock_g"])
    predictions[CONSUMPTION_BASELINE] = consumption_baseline(part["measured_stock_g"], part["rate_24h"])
    return predictions


def score_all(part, predictions):
    return {method: classification_metrics(part["label"], pred) for method, pred in predictions.items()}


def per_stock_level(part, predictions):
    """Accuracy and High recall for each stock level (macro F1 is undefined where a class is absent)."""
    rows = []
    for level in config.STOCK_LEVELS_G:
        at_level = (part["stock_g"] == level).to_numpy()
        truth = part["label"].to_numpy()[at_level]
        is_high = truth == "High"
        row = {"stock_g": level, "examples": int(at_level.sum()),
               **{f"true_{label.lower()}": int((truth == label).sum()) for label in config.LABELS}}
        for method, pred in predictions.items():
            p = pred[at_level]
            row[f"{method} accuracy"] = round(float((p == truth).mean()), 4)
            row[f"{method} High recall"] = round(float((p[is_high] == "High").mean()), 4) if is_high.any() else None
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Block 3. Figures
# =============================================================================

def plot_confusion_matrices(scores, path):
    methods = list(scores)
    fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 3.4), dpi=150)
    for ax, method in zip(axes, methods):
        counts = np.array(scores[method]["confusion_matrix"])
        row_share = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{counts[i, j]:,}\n{row_share[i, j]:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if row_share[i, j] > 0.6 else "#1a1a19")
        ax.set_xticks(range(3), config.LABELS, fontsize=7)
        ax.set_yticks(range(3), config.LABELS, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_title(method, fontsize=8)
    axes[0].set_ylabel("True label", fontsize=7)
    fig.suptitle("Test-set confusion matrices (counts and share of each true class)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_comparison(scores, path):
    methods = list(scores)
    x = np.arange(len(methods))
    macro = [scores[m]["macro_f1"] for m in methods]
    high = [scores[m]["high_recall"] for m in methods]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.bar(x - 0.18, macro, width=0.34, color="#2a78d6", label="Macro F1")
    ax.bar(x + 0.18, high, width=0.34, color="#eb6834", label="High-class recall")
    for xi, (m, h) in enumerate(zip(macro, high)):
        ax.text(xi - 0.18, m + 0.01, f"{m:.2f}", ha="center", fontsize=7)
        ax.text(xi + 0.18, h + 0.01, f"{h:.2f}", ha="center", fontsize=7)
    ax.axhline(config.TARGET_MACRO_F1, color="#2a78d6", lw=0.8, ls="--")
    ax.axhline(config.TARGET_HIGH_RECALL, color="#eb6834", lw=0.8, ls="--")
    ax.text(len(methods) - 0.5, config.TARGET_MACRO_F1 + 0.01, "macro F1 target", fontsize=7, ha="right",
            color="#52514e")
    ax.text(len(methods) - 0.5, config.TARGET_HIGH_RECALL + 0.01, "High recall target", fontsize=7, ha="right",
            color="#52514e")
    ax.set_xticks(x, methods, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score on the test set", fontsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# =============================================================================
# Block 4. The single test evaluation
# =============================================================================

def run_evaluation(selection_path=config.SELECTION_PATH, dataset_path=config.ML_DATASET_PATH,
                   split_report_path=config.SPLIT_REPORT_PATH, results_dir=config.RESULTS_DIR):
    with open(selection_path, encoding="utf-8") as f:
        selection = json.load(f)
    if selection["dataset_sha256"] != file_sha256(dataset_path):
        raise ValueError("The dataset changed after the models were selected. Re-run train.py first.")

    models = {}
    for algorithm, info in selection["kept_per_algorithm"].items():
        model_path = config.PROJECT_DIR / info["model_file"]
        if file_sha256(model_path) != info["model_sha256"]:
            raise ValueError(f"{model_path} is not the model recorded in {selection_path}.")
        models[algorithm] = joblib.load(model_path)

    parts = load_partitions(["validation", "test"], dataset_path, split_report_path)

    # Baselines on validation, for context only (they have nothing to tune).
    val_scores = score_all(parts["validation"], {
        STOCK_BASELINE: stock_only_baseline(parts["validation"]["measured_stock_g"]),
        CONSUMPTION_BASELINE: consumption_baseline(parts["validation"]["measured_stock_g"],
                                                   parts["validation"]["rate_24h"])})

    test = parts["test"]
    predictions = predict_all(test, models)
    scores = score_all(test, predictions)

    best_baseline_f1 = max(scores[STOCK_BASELINE]["macro_f1"], scores[CONSUMPTION_BASELINE]["macro_f1"])
    rows = []
    for method, m in scores.items():
        is_model = method in MODEL_NAMES.values()
        rows.append({"method": method,
                     "type": "kept model" if is_model else "baseline",
                     **{k: v for k, v in m.items() if k != "confusion_matrix"},
                     "meets_macro_f1_target": m["macro_f1"] >= config.TARGET_MACRO_F1,
                     "meets_high_recall_target": m["high_recall"] >= config.TARGET_HIGH_RECALL,
                     "beats_both_baselines_macro_f1": (m["macro_f1"] > best_baseline_f1) if is_model else None})
    results = pd.DataFrame(rows)

    results_dir.mkdir(parents=True, exist_ok=True)
    save_csv(results, results_dir / "test_results.csv")
    save_csv(per_stock_level(test, predictions), results_dir / "per_stock_level_test.csv")
    save_csv(pd.DataFrame([{"method": k, **{a: b for a, b in v.items() if a != "confusion_matrix"}}
                           for k, v in val_scores.items()]), results_dir / "baselines_validation.csv")
    confusion_dir = results_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    for method, m in scores.items():
        matrix = pd.DataFrame(m["confusion_matrix"], index=[f"true {l}" for l in config.LABELS],
                              columns=[f"predicted {l}" for l in config.LABELS])
        matrix.to_csv(confusion_dir / f"{method.lower().replace(' ', '_').replace('.', '')}_test.csv",
                      lineterminator="\n")
    plot_confusion_matrices(scores, results_dir / "test_confusion_matrices.png")
    plot_comparison(scores, results_dir / "test_comparison.png")

    final_name = MODEL_NAMES[selection["final_model"]]
    report = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "rule": "Models were selected on validation by train.py; this step only scores them on test.",
        "selection_sha256": file_sha256(selection_path),
        "test_rows": len(test),
        "test_class_counts": {l: int((test["label"] == l).sum()) for l in config.LABELS},
        "final_model": final_name,
        "final_model_test": {k: v for k, v in scores[final_name].items()},
        "final_model_meets_targets_on_test": bool(scores[final_name]["macro_f1"] >= config.TARGET_MACRO_F1
                                                  and scores[final_name]["high_recall"] >= config.TARGET_HIGH_RECALL),
        "best_baseline_macro_f1": best_baseline_f1,
        "models_beating_both_baselines": [r["method"] for r in rows if r["beats_both_baselines_macro_f1"]],
        "test_scores": scores,
        "baselines_on_validation": val_scores,
    }
    save_json(report, results_dir / "evaluation_report.json")
    return report, results


def main():
    report, results = run_evaluation()
    columns = ["method", "accuracy", "macro_f1", "high_precision", "high_recall", "medium_f1", "low_f1"]
    print(f"Test set: {report['test_rows']} examples {report['test_class_counts']}\n")
    print(results[columns].to_string(index=False))
    print("\nFinal model (chosen on validation):", report["final_model"])
    print("Meets both targets on test        :", report["final_model_meets_targets_on_test"])
    print("Models beating both baselines     :", report["models_beating_both_baselines"] or "none")
    print("Written to                        :", relative_path(config.RESULTS_DIR))


if __name__ == "__main__":
    main()
