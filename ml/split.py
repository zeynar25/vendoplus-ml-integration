"""
split.py - Pipeline step 6: chronological train / validation / test partitions

Every row is assigned to a partition by the DATE of its decision time:
    train      : 3 March 2024 - 30 October 2024
    validation : 1 November 2024 - 30 December 2024
    test       : 1 January 2025 - 21 March 2025
Decision times on 31 October and 31 December are left out, so that no
training label uses validation sales and no validation label uses test sales.
All 15 rows of one decision time (3 hoppers x 5 stock levels) share its date,
so they always land in the same partition. Nothing is shuffled.

This step also produces the training-only class counts per stock level that
decision D-08 asks for (the stock-level review, P-03).

Output: data/processed/ml_dataset.csv (features + label + partition)

How to run (from the project folder):
    python split.py
"""

import pandas as pd

import config
from labels import class_counts
from pipeline_utils import (file_sha256, relative_path, save_csv, save_json,
                            stop_if_any_check_failed, verify_input)


def assign_partitions(decision_times):
    """Return 'train', 'validation', 'test' or 'excluded' for each decision time."""
    day = pd.Series(pd.DatetimeIndex(decision_times).normalize())
    partition = pd.Series("excluded", index=day.index)
    for name, (first_day, last_day) in config.PARTITIONS.items():
        inside = (day >= pd.Timestamp(first_day)) & (day <= pd.Timestamp(last_day))
        partition[inside] = name
    return partition.to_numpy()


def run_checks(dataset):
    times = dataset.groupby("partition")["decision_time"].agg(["min", "max"])
    horizon = pd.Timedelta(hours=config.LABEL_HORIZON_HOURS)
    rows_per_time = dataset.groupby("decision_time").size()
    partitions_per_time = dataset.groupby("decision_time")["partition"].nunique()

    expected_rows = {}
    for name, (first_day, last_day) in config.PARTITIONS.items():
        days = (pd.Timestamp(last_day) - pd.Timestamp(first_day)).days + 1
        expected_rows[name] = days * 24 * len(config.HOPPERS) * len(config.STOCK_LEVELS_G)

    checks = {
        "row_counts_match_the_planned_dates": all(
            int((dataset["partition"] == name).sum()) == n for name, n in expected_rows.items()),
        "each_decision_time_in_one_partition": bool((partitions_per_time == 1).all()),
        "each_decision_time_has_all_15_scenarios": bool(
            (rows_per_time == len(config.HOPPERS) * len(config.STOCK_LEVELS_G)).all()),
        "train_labels_end_before_validation_starts": times.loc["train", "max"] + horizon < times.loc["validation", "min"],
        "validation_labels_end_before_test_starts": times.loc["validation", "max"] + horizon < times.loc["test", "min"],
        "partitions_in_time_order": times.loc["train", "max"] < times.loc["validation", "min"]
                                    and times.loc["validation", "max"] < times.loc["test", "min"],
    }
    return {name: bool(passed) for name, passed in checks.items()}, expected_rows


def run_split(features_path=config.FEATURES_PATH, features_report_path=config.FEATURES_REPORT_PATH,
              dataset_path=config.ML_DATASET_PATH, report_path=config.SPLIT_REPORT_PATH):
    verify_input(features_path, features_report_path, "features_file")
    table = pd.read_csv(features_path, parse_dates=["decision_time"])
    table["partition"] = assign_partitions(table["decision_time"])

    excluded = table[table["partition"] == "excluded"]
    dataset = table[table["partition"] != "excluded"].reset_index(drop=True)
    checks, expected_rows = run_checks(dataset)
    save_csv(dataset, dataset_path)

    train = dataset[dataset["partition"] == "train"]
    high_from_10g = (train["label"].eq("High") & train["stock_g"].eq(10)).sum() / max(train["label"].eq("High").sum(), 1)
    report = {
        "partitions": config.PARTITIONS,
        "excluded_days": sorted(excluded["decision_time"].dt.strftime("%Y-%m-%d").unique().tolist()),
        "excluded_rows": len(excluded),
        "rows_per_partition": {p: int((dataset["partition"] == p).sum()) for p in config.PARTITIONS},
        "expected_rows_per_partition": expected_rows,
        "class_counts_per_partition": class_counts(dataset, "partition"),
        "class_counts_per_partition_and_hopper": class_counts(dataset, ["partition", "hopper"]),
        "stock_level_review_training_only": {
            "class_counts_by_stock_level_g": class_counts(train, "stock_g"),
            "share_of_training_high_labels_at_10g": round(float(high_from_10g), 4),
        },
        "dataset_file": {"path": relative_path(dataset_path), "sha256": file_sha256(dataset_path),
                         "rows": len(dataset)},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save_json(report, report_path)
    stop_if_any_check_failed(checks, report_path)
    return report


def main():
    report = run_split()
    print("Excluded days      :", ", ".join(report["excluded_days"]), f"({report['excluded_rows']} rows)")
    for name, n in report["rows_per_partition"].items():
        print(f"{name:11s}        : {n:6d} rows  {report['class_counts_per_partition'][name]}")
    print("Training rows per stock level (Low, Medium, High):")
    for level, counts in report["stock_level_review_training_only"]["class_counts_by_stock_level_g"].items():
        print(f"  {level:>4} g          :", counts)
    print("Checks passed      :", sum(report["checks"].values()), "of", len(report["checks"]))
    print("Written            :", report["dataset_file"]["path"])


if __name__ == "__main__":
    main()
