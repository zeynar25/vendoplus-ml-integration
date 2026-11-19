"""
stock_simulation.py - Pipeline step 3: hopper stock simulation

For every hourly decision time t, every hopper and every stock level s:
  - start the hopper with s grams at time t (the TRUE simulated stock, no noise);
  - replay the sales after t, in order, with no refill;
  - the shortage is the first sale that needs more grams than are left;
  - record when that happens, if it happens within 24 hours.

Output: data/processed/stock_scenarios.csv, one row per
(decision time, hopper, stock level), with the shortage time and the hours
from t to the shortage (blank when there is no shortage within 24 hours).
Labels are assigned in the next step (labels.py).

How to run (from the project folder):
    python stock_simulation.py
"""

import numpy as np
import pandas as pd

import config
from pipeline_utils import (file_sha256, load_transactions, relative_path, save_csv,
                            save_json, stop_if_any_check_failed)


# =============================================================================
# Block 1. Which decision times have complete 24-hour windows
# =============================================================================

def decision_times(start=config.DECISION_START, end=config.DECISION_END):
    """Every hour from start to end, both included."""
    return pd.date_range(start, end, freq="h")


def window_is_complete(times, period_start=config.STUDY_PERIOD_START,
                       period_end_exclusive=config.STUDY_PERIOD_END_EXCLUSIVE):
    """
    A decision time is usable only if the 24 hours before it (for the features)
    and the 24 hours after it (for the label) both lie inside the study period.
    """
    before_ok = times - pd.Timedelta(hours=24) >= pd.Timestamp(period_start)
    after_ok = times + pd.Timedelta(hours=config.LABEL_HORIZON_HOURS) <= pd.Timestamp(period_end_exclusive)
    return np.asarray(before_ok & after_ok)


# =============================================================================
# Block 2. The replay: first shortage after each decision time
# =============================================================================

def first_shortage_times(sale_times, grams, times, stock_g, horizon_hours=config.LABEL_HORIZON_HOURS):
    """
    For each decision time in `times`, return the time of the first sale that
    cannot be served from `stock_g` grams (NaT if none within horizon_hours).

    How it works, without looping over every sale:
    used_before[i] = grams used by the first i sales (a running total).
    A hopper that holds `stock_g` at time t runs short at the first later sale
    where (running total at that sale) - (running total at t) > stock_g.
    Because the running total only goes up, np.searchsorted finds that sale
    directly, like looking up a word in a sorted dictionary.
    """
    sale_times = pd.DatetimeIndex(sale_times)
    times = pd.DatetimeIndex(times)
    grams = np.asarray(grams)

    used_before = np.concatenate([[0], np.cumsum(grams)])
    # Number of sales at or before t. A sale exactly at t counts as the past.
    sold_by_t = sale_times.searchsorted(times, side="right")
    limit = used_before[sold_by_t] + stock_g
    # First i with used_before[i] > limit: the failing sale is sale number i - 1.
    i = np.searchsorted(used_before, limit, side="right")

    shortage = pd.Series(pd.NaT, index=range(len(times)), dtype="datetime64[ns]")
    has_failing_sale = i <= len(grams)
    shortage[has_failing_sale] = sale_times[i[has_failing_sale] - 1]
    # Only shortages up to and including t + horizon count.
    too_late = shortage > pd.Series(times + pd.Timedelta(hours=horizon_hours))
    shortage[too_late] = pd.NaT
    return pd.DatetimeIndex(shortage)


def simulate_all(transactions, times):
    """One row per (decision time, hopper, stock level)."""
    pieces = []
    for hopper in config.HOPPERS:
        grams = transactions[f"{hopper}_g"].to_numpy()
        for stock_g in config.STOCK_LEVELS_G:
            shortage = first_shortage_times(transactions["datetime"], grams, times, stock_g)
            pieces.append(pd.DataFrame({
                "decision_time": times,
                "hopper": hopper,
                "stock_g": stock_g,
                "shortage_time": shortage,
                "hours_to_shortage": (shortage - times) / pd.Timedelta(hours=1),
            }))
    scenarios = pd.concat(pieces, ignore_index=True)
    # Fixed row order: time, then hopper (coffee, milk, chocolate), then stock level.
    scenarios["hopper_order"] = scenarios["hopper"].map({h: k for k, h in enumerate(config.HOPPERS)})
    scenarios = (scenarios.sort_values(["decision_time", "hopper_order", "stock_g"])
                 .drop(columns="hopper_order").reset_index(drop=True))
    return scenarios


# =============================================================================
# Block 3. Checks, including a slow, simple replay of a random sample
# =============================================================================

def slow_replay(sale_times, grams, t, stock_g, horizon_hours=config.LABEL_HORIZON_HOURS):
    """The same rule written as a plain loop, used only to double-check Block 2."""
    left = stock_g
    end = t + pd.Timedelta(hours=horizon_hours)
    for when, need in zip(sale_times, grams):
        if when <= t or need == 0:
            continue
        if when > end:
            return pd.NaT
        if need > left:
            return when
        left -= need
    return pd.NaT


def run_checks(transactions, times, scenarios, sample_size=500):
    expected_rows = len(times) * len(config.HOPPERS) * len(config.STOCK_LEVELS_G)
    hours = scenarios["hours_to_shortage"].dropna()

    rng = np.random.default_rng(config.RANDOM_SEED)
    sample = scenarios.iloc[rng.choice(len(scenarios), size=min(sample_size, len(scenarios)), replace=False)]
    sample_agrees = True
    for _, row in sample.iterrows():
        slow = slow_replay(transactions["datetime"], transactions[f"{row['hopper']}_g"],
                           row["decision_time"], row["stock_g"])
        fast = row["shortage_time"]
        if not ((pd.isna(slow) and pd.isna(fast)) or slow == fast):
            sample_agrees = False
            break

    checks = {
        "one_row_per_time_hopper_and_level": len(scenarios) == expected_rows,
        "no_duplicate_rows": not scenarios.duplicated(["decision_time", "hopper", "stock_g"]).any(),
        "shortages_happen_after_t_and_within_24h": bool(((hours > 0) & (hours <= config.LABEL_HORIZON_HOURS)).all()),
        "more_stock_never_runs_out_sooner": more_stock_never_runs_out_sooner(scenarios),
        "fast_replay_matches_slow_replay_on_sample": sample_agrees,
    }
    return checks


def more_stock_never_runs_out_sooner(scenarios):
    """At one decision time and hopper, a fuller hopper cannot run out earlier."""
    hours = scenarios["hours_to_shortage"].fillna(np.inf)
    previous = hours.groupby([scenarios["decision_time"], scenarios["hopper"]]).shift()
    return bool((hours[previous.notna()] >= previous[previous.notna()]).all())


# =============================================================================
# Block 4. Run and write the outputs
# =============================================================================

def run_stock_simulation(transactions_path=None, consumption_report_path=config.CONSUMPTION_REPORT_PATH,
                         scenarios_path=config.SCENARIOS_PATH, report_path=config.SCENARIOS_REPORT_PATH):
    transactions_path = transactions_path or config.TRANSACTION_CONSUMPTION_PATH
    transactions = load_transactions(transactions_path, consumption_report_path)

    all_times = decision_times()
    complete = window_is_complete(all_times)
    times = all_times[complete]

    scenarios = simulate_all(transactions, times)
    checks = run_checks(transactions, times, scenarios)
    save_csv(scenarios, scenarios_path, datetime_format=config.OUTPUT_DATETIME_FORMAT)

    has_24h = scenarios["hours_to_shortage"].notna()
    has_4h = scenarios["hours_to_shortage"] <= config.HIGH_HORIZON_HOURS
    share = (scenarios.assign(short_24h=has_24h, short_4h=has_4h)
             .groupby(["hopper", "stock_g"])[["short_24h", "short_4h"]].mean().round(4))
    report = {
        "note": "Simulated with the TRUE stock (no noise) and recipe "
                f"{config.RECIPE_VERSION}; no refill during the 24-hour replay.",
        "input": {"path": relative_path(transactions_path), "sha256": file_sha256(transactions_path)},
        "decision_times": {"first": str(times.min()), "last": str(times.max()),
                           "candidates": len(all_times), "kept": len(times),
                           "excluded_incomplete_window": int((~complete).sum())},
        "stock_levels_g": config.STOCK_LEVELS_G,
        "scenario_file": {"path": relative_path(scenarios_path), "sha256": file_sha256(scenarios_path),
                          "rows": len(scenarios)},
        "share_with_shortage": {f"{h} {s} g": {"within_24h": r.short_24h, "within_4h": r.short_4h}
                                for (h, s), r in share.iterrows()},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save_json(report, report_path)
    stop_if_any_check_failed(checks, report_path)
    return report


def main():
    report = run_stock_simulation()
    d = report["decision_times"]
    print("Decision times     :", d["kept"], f"({d['first']} to {d['last']})")
    print("Incomplete windows :", d["excluded_incomplete_window"], "excluded")
    print("Scenario rows      :", report["scenario_file"]["rows"])
    print("Checks passed      :", sum(report["checks"].values()), "of", len(report["checks"]))
    print("Written            :", report["scenario_file"]["path"])


if __name__ == "__main__":
    main()
