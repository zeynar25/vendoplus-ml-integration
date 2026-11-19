"""
features.py - Pipeline step 5: feature engineering

Builds the 11 model features for every labeled scenario, using ONLY
information available at the decision time t:

  stock_ratio            measured stock / 1,000 g capacity. "Measured" = the
                         true simulated stock + Gaussian noise (sd 2 g, seed 42),
                         added in every partition (D-09). Labels keep using the
                         TRUE stock.
  rate_1h, rate_6h,      grams this hopper used in the last 1, 6 and 24 hours,
  rate_24h               i.e. sales in (t - window, t], as a share of capacity
                         per hour. Computed from exact sale times (P-05).
  hour_sin, hour_cos     hour of day on a circle: sin/cos(2*pi*hour/24)
  weekday_sin,           day of week on a circle (Monday = 0):
  weekday_cos            sin/cos(2*pi*weekday/7)
  hopper_coffee, ...     1 for this row's hopper, 0 for the others

Output: data/processed/features.csv

How to run (from the project folder):
    python features.py
"""

import numpy as np
import pandas as pd

import config
from pipeline_utils import (file_sha256, load_transactions, relative_path, save_csv, save_json,
                            stop_if_any_check_failed, verify_input)


# =============================================================================
# Block 1. Consumption rates from the past only
# =============================================================================

def consumption_rates(sale_times, grams, times):
    """
    For each decision time t: grams used in (t - window, t] for each window,
    divided by capacity and by the window length in hours.
    Sales after t are never counted, because searchsorted(..., side="right")
    stops at t itself.
    """
    sale_times = pd.DatetimeIndex(sale_times)
    times = pd.DatetimeIndex(times)
    used_before = np.concatenate([[0], np.cumsum(np.asarray(grams))])
    sold_by_t = sale_times.searchsorted(times, side="right")
    rates = {}
    for hours in config.RATE_WINDOWS_HOURS:
        sold_by_window_start = sale_times.searchsorted(times - pd.Timedelta(hours=hours), side="right")
        grams_in_window = used_before[sold_by_t] - used_before[sold_by_window_start]
        rates[f"rate_{hours}h"] = grams_in_window / config.HOPPER_CAPACITY_G / hours
    return pd.DataFrame(rates, index=times)


# =============================================================================
# Block 2. All 11 features
# =============================================================================

def add_features(scenarios, transactions, seed=config.RANDOM_SEED):
    table = scenarios.copy()
    times = pd.DatetimeIndex(table["decision_time"].unique())

    # Consumption rates: computed once per (decision time, hopper), then attached
    # to all five stock levels of that decision time.
    rate_tables = []
    for hopper in config.HOPPERS:
        rates = consumption_rates(transactions["datetime"], transactions[f"{hopper}_g"], times)
        rates = rates.rename_axis("decision_time").reset_index()
        rates["hopper"] = hopper
        rate_tables.append(rates)
    table = table.merge(pd.concat(rate_tables), on=["decision_time", "hopper"], how="left",
                        validate="many_to_one")

    # Measured stock = true stock + noise. One draw per row, in the file's fixed
    # row order, from a generator started with the fixed seed: same numbers every run.
    rng = np.random.default_rng(seed)
    table["measured_stock_g"] = table["stock_g"] + rng.normal(0.0, config.NOISE_SD_G, size=len(table))
    table["stock_ratio"] = table["measured_stock_g"] / config.HOPPER_CAPACITY_G

    hour = table["decision_time"].dt.hour
    weekday = table["decision_time"].dt.dayofweek  # Monday = 0
    table["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    table["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    table["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    table["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    for hopper in config.HOPPERS:
        table[f"hopper_{hopper}"] = (table["hopper"] == hopper).astype(int)

    keep = ["decision_time", "hopper", "stock_g", "measured_stock_g", "hours_to_shortage", "label"]
    return table[keep + config.FEATURE_COLUMNS]


# =============================================================================
# Block 3. Checks
# =============================================================================

def run_checks(scenarios, transactions, table, sample_size=200):
    features = table[config.FEATURE_COLUMNS]
    noise = table["measured_stock_g"] - table["stock_g"]

    # Recompute the rates for random decision times with a plain filter that
    # keeps only sales in (t - window, t]. They must match Block 1.
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample = table.iloc[rng.choice(len(table), size=min(sample_size, len(table)), replace=False)]
    rates_match = True
    for _, row in sample.iterrows():
        t = row["decision_time"]
        for hours in config.RATE_WINDOWS_HOURS:
            in_window = (transactions["datetime"] > t - pd.Timedelta(hours=hours)) & (transactions["datetime"] <= t)
            expected = transactions.loc[in_window, f"{row['hopper']}_g"].sum() / config.HOPPER_CAPACITY_G / hours
            rates_match &= bool(np.isclose(row[f"rate_{hours}h"], expected))

    same_rates_for_all_levels = (table.groupby(["decision_time", "hopper"])[["rate_1h", "rate_6h", "rate_24h"]]
                                 .nunique().max().max() == 1)
    checks = {
        "one_row_per_scenario": len(table) == len(scenarios),
        "no_missing_feature_values": not features.isna().any().any(),
        "rates_are_not_negative": bool((table[["rate_1h", "rate_6h", "rate_24h"]] >= 0).all().all()),
        "rates_use_only_sales_up_to_t": rates_match,
        "rates_shared_by_the_five_stock_levels": bool(same_rates_for_all_levels),
        "exactly_one_hopper_flag_per_row": bool((table[[f"hopper_{h}" for h in config.HOPPERS]].sum(axis=1) == 1).all()),
        "noise_mean_close_to_0": bool(abs(noise.mean()) < 0.05),
        "noise_sd_close_to_2g": bool(abs(noise.std() - config.NOISE_SD_G) < 0.05),
        "labels_unchanged": bool((table["label"].to_numpy() == scenarios["label"].to_numpy()).all()),
    }
    return checks


# =============================================================================
# Block 4. Run and write the outputs
# =============================================================================

def run_features(labeled_path=config.LABELED_SCENARIOS_PATH, labels_report_path=config.LABELS_REPORT_PATH,
                 transactions_path=None, consumption_report_path=config.CONSUMPTION_REPORT_PATH,
                 features_path=config.FEATURES_PATH, report_path=config.FEATURES_REPORT_PATH):
    transactions_path = transactions_path or config.TRANSACTION_CONSUMPTION_PATH
    verify_input(labeled_path, labels_report_path, "labeled_file")
    scenarios = pd.read_csv(labeled_path, parse_dates=["decision_time"])
    transactions = load_transactions(transactions_path, consumption_report_path)

    table = add_features(scenarios, transactions)
    checks = run_checks(scenarios, transactions, table)
    save_csv(table, features_path)

    noise = table["measured_stock_g"] - table["stock_g"]
    report = {
        "feature_columns_in_model_order": config.FEATURE_COLUMNS,
        "noise": {"sd_setting_g": config.NOISE_SD_G, "seed": config.RANDOM_SEED,
                  "observed_mean_g": round(float(noise.mean()), 4), "observed_sd_g": round(float(noise.std()), 4),
                  "applied_to": "every row in every partition; labels use the true stock"},
        "input": {"labeled_path": relative_path(labeled_path), "labeled_sha256": file_sha256(labeled_path),
                  "transactions_sha256": file_sha256(transactions_path)},
        "features_file": {"path": relative_path(features_path), "sha256": file_sha256(features_path),
                          "rows": len(table)},
        "feature_summary": table[config.FEATURE_COLUMNS].describe().loc[["mean", "min", "max"]].round(5).to_dict(),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save_json(report, report_path)
    stop_if_any_check_failed(checks, report_path)
    return report


def main():
    report = run_features()
    print("Feature rows  :", report["features_file"]["rows"])
    print("Features      :", ", ".join(report["feature_columns_in_model_order"]))
    print("Noise (g)     : mean", report["noise"]["observed_mean_g"], " sd", report["noise"]["observed_sd_g"])
    print("Checks passed :", sum(report["checks"].values()), "of", len(report["checks"]))
    print("Written       :", report["features_file"]["path"])


if __name__ == "__main__":
    main()
