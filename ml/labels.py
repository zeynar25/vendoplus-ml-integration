"""
labels.py - Pipeline step 4: restock urgency labels

Turns each simulated scenario's "hours to shortage" into an urgency label:
    High   : shortage at or before t + 4 h
    Medium : shortage after t + 4 h and at or before t + 24 h
    Low    : no shortage within 24 h

The labels come from replaying recorded sales against assumed stock levels
and recipes. They are NOT historical restock records.

Output: data/processed/labeled_scenarios.csv (the scenarios plus a "label" column)

How to run (from the project folder):
    python labels.py
"""

import numpy as np
import pandas as pd

import config
from pipeline_utils import (file_sha256, relative_path, save_csv, save_json,
                            stop_if_any_check_failed, verify_input)


def assign_labels(hours_to_shortage):
    """Map hours to shortage (blank = no shortage within 24 h) to Low / Medium / High."""
    hours = pd.Series(hours_to_shortage, dtype="float64")
    labels = np.where(hours.isna(), "Low",
                      np.where(hours <= config.HIGH_HORIZON_HOURS, "High", "Medium"))
    return pd.Series(labels, index=hours.index)


def class_counts(table, by=None):
    """Number of rows of each label, optionally split by one or more columns."""
    if by is None:
        return {label: int((table["label"] == label).sum()) for label in config.LABELS}
    counts = table.groupby(by)["label"].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=config.LABELS, fill_value=0)
    return {str(key): {label: int(n) for label, n in row.items()} for key, row in counts.iterrows()}


def run_labels(scenarios_path=config.SCENARIOS_PATH, scenarios_report_path=config.SCENARIOS_REPORT_PATH,
               labeled_path=config.LABELED_SCENARIOS_PATH, report_path=config.LABELS_REPORT_PATH):
    verify_input(scenarios_path, scenarios_report_path, "scenario_file")
    scenarios = pd.read_csv(scenarios_path, parse_dates=["decision_time", "shortage_time"])
    scenarios["label"] = assign_labels(scenarios["hours_to_shortage"])

    hours = scenarios["hours_to_shortage"]
    checks = {
        "every_row_has_a_known_label": scenarios["label"].isin(config.LABELS).all(),
        "high_means_within_4h": bool((hours[scenarios["label"] == "High"] <= config.HIGH_HORIZON_HOURS).all()),
        "medium_means_after_4h_within_24h": bool(
            hours[scenarios["label"] == "Medium"].between(config.HIGH_HORIZON_HOURS, config.LABEL_HORIZON_HOURS,
                                                          inclusive="right").all()),
        "low_means_no_shortage": bool(hours[scenarios["label"] == "Low"].isna().all()),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    save_csv(scenarios, labeled_path, datetime_format=config.OUTPUT_DATETIME_FORMAT)

    report = {
        "note": "Labels are simulated from sales and assumed recipes, not historical restock records.",
        "rules": {"High": f"shortage at or before t + {config.HIGH_HORIZON_HOURS} h",
                  "Medium": f"shortage after t + {config.HIGH_HORIZON_HOURS} h, at or before "
                            f"t + {config.LABEL_HORIZON_HOURS} h",
                  "Low": f"no shortage within {config.LABEL_HORIZON_HOURS} h"},
        "input": {"path": relative_path(scenarios_path), "sha256": file_sha256(scenarios_path)},
        "labeled_file": {"path": relative_path(labeled_path), "sha256": file_sha256(labeled_path),
                         "rows": len(scenarios)},
        "class_counts_all_decision_times": class_counts(scenarios),
        "class_counts_by_hopper": class_counts(scenarios, "hopper"),
        "class_counts_by_stock_level_g": class_counts(scenarios, "stock_g"),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save_json(report, report_path)
    stop_if_any_check_failed(checks, report_path)
    return report


def main():
    report = run_labels()
    print("Labeled rows :", report["labeled_file"]["rows"])
    print("Class counts :", report["class_counts_all_decision_times"])
    for level, counts in report["class_counts_by_stock_level_g"].items():
        print(f"  {level:>4} g    :", counts)
    print("Checks passed:", sum(report["checks"].values()), "of", len(report["checks"]))
    print("Written      :", report["labeled_file"]["path"])


if __name__ == "__main__":
    main()
