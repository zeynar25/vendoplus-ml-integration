"""
consumption.py - Pipeline step 2: ingredient consumption calculation

What this script does:
  1. Reads the cleaned sales (data/processed/coffee_sales_cleaned.csv) and
     confirms it is the same file that preprocessing.py produced.
  2. Looks up each sale's beverage in the recipe table (config.RECIPES_G) and
     records how many grams it takes from the coffee, milk and chocolate hoppers.
  3. Adds the sales up hour by hour, keeping hours with no sales as zeros.
  4. Writes three files to data/processed/:
       transaction_consumption.csv - one row per sale, with grams per hopper
       hourly_consumption.csv      - one row per hour of the study period
       consumption_report.json     - totals, the recipe used, and checks

The grams are ESTIMATES from the recipe table, not measurements from the
Kaggle machine. No stock levels, noise or labels are produced here.

How to run (from the project folder):
    python consumption.py
"""

import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

import config

# Column names for the grams taken from each hopper, e.g. "coffee_g".
GRAM_COLUMNS = [f"{hopper}_g" for hopper in config.HOPPERS]


# =============================================================================
# Block 1. Small helpers
# =============================================================================

def file_sha256(path):
    """Return the SHA-256 fingerprint of a file (reading only)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(path):
    """Path relative to the project folder, written with / on every OS."""
    try:
        return path.resolve().relative_to(config.PROJECT_DIR).as_posix()
    except ValueError:  # file outside the project folder (happens in the tests)
        return path.resolve().as_posix()


# =============================================================================
# Block 2. Load the cleaned sales and confirm they are the approved file
# =============================================================================

def load_cleaned_sales(cleaned_path, cleaning_report_path=None):
    """
    Read the cleaned sales. If the cleaning report is given, check that the
    file's SHA-256 still matches the one recorded when it was cleaned, so we
    never calculate consumption from a file that was edited by hand.
    """
    if cleaning_report_path is not None:
        with open(cleaning_report_path, encoding="utf-8") as f:
            expected_hash = json.load(f)["cleaned_file"]["sha256"]
        if file_sha256(cleaned_path) != expected_hash:
            raise ValueError(f"{cleaned_path} does not match the SHA-256 in {cleaning_report_path}. "
                             f"Re-run preprocessing.py instead of editing the cleaned file.")

    sales = pd.read_csv(cleaned_path)
    missing = [c for c in ("datetime", "coffee_name") if c not in sales.columns]
    if missing:
        raise ValueError(f"Cleaned file is missing columns: {missing}")

    sales["datetime"] = pd.to_datetime(sales["datetime"], format=config.OUTPUT_DATETIME_FORMAT)
    return sales


# =============================================================================
# Block 3. Transaction level: grams taken from each hopper by each sale
# =============================================================================

def recipe_table():
    """The recipe dict from config.py as a table: one row per beverage, one column per hopper."""
    table = pd.DataFrame.from_dict(config.RECIPES_G, orient="index")[config.HOPPERS]
    table.columns = GRAM_COLUMNS
    return table


def calculate_transaction_consumption(sales):
    """
    Return one row per sale with the grams it takes from each hopper.
    A beverage that is not in the recipe table stops the script with an error:
    it is never silently given zero consumption.
    """
    recipes = recipe_table()

    unknown = sales.loc[~sales["coffee_name"].isin(recipes.index), "coffee_name"]
    if len(unknown) > 0:
        counts = unknown.value_counts().to_dict()
        raise ValueError(f"Beverages with no recipe in config.RECIPES_G: {counts}")

    transactions = sales[["datetime", "coffee_name"]].copy()
    # Number the sales 1, 2, 3, ... in time order, so we can prove later that
    # none are lost or duplicated.
    transactions.insert(0, "transaction_id", range(1, len(transactions) + 1))
    # Look up each sale's recipe row and attach its grams (like VLOOKUP in Excel).
    transactions = transactions.join(recipes, on="coffee_name")
    return transactions


# =============================================================================
# Block 4. Hourly totals, including hours with no sales
# =============================================================================

def aggregate_hourly(transactions, period_start, period_end_exclusive):
    """
    Add up the sales in each clock hour. The hour labelled 10:00 holds the
    sales from 10:00:00 up to (but not including) 11:00:00.
    Every hour of the study period appears, and hours with no sales get 0.
    """
    hour_start = transactions["datetime"].dt.floor("h")
    grouped = transactions.groupby(hour_start)
    hourly = grouped[GRAM_COLUMNS].sum()
    hourly.insert(0, "transactions", grouped.size())

    all_hours = pd.date_range(start=pd.Timestamp(period_start),
                              end=pd.Timestamp(period_end_exclusive) - pd.Timedelta(hours=1),
                              freq="h")
    # reindex adds the missing hours; fill_value=0 makes them 0 instead of empty.
    hourly = hourly.reindex(all_hours, fill_value=0)
    hourly.index.name = "hour_start"
    return hourly.reset_index()


# =============================================================================
# Block 5. Checks: every check must be True, or the script stops with an error
# =============================================================================

def run_checks(sales, transactions, hourly, period_start, period_end_exclusive):
    recipes = recipe_table()
    expected_hours = int((pd.Timestamp(period_end_exclusive) - pd.Timestamp(period_start))
                         / pd.Timedelta(hours=1))
    no_sale_hours = hourly["transactions"] == 0

    # Expected totals worked out a second way: (number of sales of each
    # beverage) x (its recipe), summed over the beverages.
    sales_per_beverage = transactions["coffee_name"].value_counts()
    expected_totals = recipes.loc[sales_per_beverage.index].mul(sales_per_beverage, axis=0).sum()

    checks = {
        "one_transaction_row_per_cleaned_sale": len(transactions) == len(sales),
        "transaction_ids_are_unique": transactions["transaction_id"].is_unique,
        "no_missing_grams": not transactions[GRAM_COLUMNS].isna().any().any(),
        "no_negative_grams": bool((transactions[GRAM_COLUMNS] >= 0).all().all()),
        "transaction_totals_match_recipe_x_count": all(
            np.isclose(transactions[c].sum(), expected_totals[c]) for c in GRAM_COLUMNS),
        "every_hour_of_the_period_present_once": (len(hourly) == expected_hours
                                                  and hourly["hour_start"].is_unique),
        "hours_are_consecutive": bool((hourly["hour_start"].diff().dropna()
                                       == pd.Timedelta(hours=1)).all()),
        "no_missing_values_in_hourly": not hourly.isna().any().any(),
        "hourly_transactions_add_up_to_all_sales": int(hourly["transactions"].sum()) == len(transactions),
        "hourly_grams_add_up_to_transaction_grams": all(
            np.isclose(hourly[c].sum(), transactions[c].sum()) for c in GRAM_COLUMNS),
        "no_sale_hours_have_zero_grams": bool((hourly.loc[no_sale_hours, GRAM_COLUMNS] == 0).all().all()),
        "all_sales_inside_the_hourly_range": bool(
            (transactions["datetime"] >= pd.Timestamp(period_start)).all()
            and (transactions["datetime"] < pd.Timestamp(period_end_exclusive)).all()),
    }
    return {name: bool(passed) for name, passed in checks.items()}


# =============================================================================
# Block 6. Run everything in order and write the output files
# =============================================================================

def run_consumption(cleaned_path, cleaning_report_path, transactions_path, hourly_path, report_path,
                    period_start=config.STUDY_PERIOD_START,
                    period_end_exclusive=config.STUDY_PERIOD_END_EXCLUSIVE):
    cleaned_hash_before = file_sha256(cleaned_path)

    sales = load_cleaned_sales(cleaned_path, cleaning_report_path)                # Block 2
    transactions = calculate_transaction_consumption(sales)                      # Block 3
    hourly = aggregate_hourly(transactions, period_start, period_end_exclusive)  # Block 4
    checks = run_checks(sales, transactions, hourly, period_start, period_end_exclusive)  # Block 5

    transactions_path.parent.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(transactions_path, index=False, lineterminator="\n", encoding="utf-8",
                        date_format=config.OUTPUT_DATETIME_FORMAT)
    hourly_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(hourly_path, index=False, lineterminator="\n", encoding="utf-8",
                  date_format="%Y-%m-%d %H:%M:%S")

    # Prove the cleaned input file was only read, never changed.
    checks["cleaned_file_unchanged"] = file_sha256(cleaned_path) == cleaned_hash_before

    has_sales = hourly["transactions"] > 0
    report = {
        "report_created_at": datetime.now().isoformat(timespec="seconds"),
        "software": {"pandas": pd.__version__, "numpy": np.__version__},
        "note": "Grams are estimates from the recipe table (simulation inputs), "
                "not measurements from the Kaggle machine.",
        "recipe_version": config.RECIPE_VERSION,
        "recipes_g_used": config.RECIPES_G,
        "input": {"path": relative_path(cleaned_path), "sha256": cleaned_hash_before,
                  "sales": len(sales)},
        "transaction_file": {
            "path": relative_path(transactions_path),
            "sha256": file_sha256(transactions_path),
            "rows": len(transactions),
            "columns": list(transactions.columns),
            "sales_per_beverage": {k: int(v) for k, v in
                                   transactions["coffee_name"].value_counts().items()},
        },
        "hourly_file": {
            "path": relative_path(hourly_path),
            "sha256": file_sha256(hourly_path),
            "rows": len(hourly),
            "columns": list(hourly.columns),
            "first_hour": str(hourly["hour_start"].min()),
            "last_hour": str(hourly["hour_start"].max()),
            "hour_rule": "hour H holds sales from H:00:00 up to but not including (H+1):00:00",
            "hours_with_sales": int(has_sales.sum()),
            "hours_without_sales": int((~has_sales).sum()),
            "most_sales_in_one_hour": int(hourly["transactions"].max()),
        },
        "total_grams": {c: float(transactions[c].sum()) for c in GRAM_COLUMNS},
        "max_grams_in_one_hour": {c: float(hourly[c].max()) for c in GRAM_COLUMNS},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Consumption checks failed: {failed}. See {report_path}")
    return report


def main():
    report = run_consumption(config.CLEANED_DATA_PATH, config.CLEANING_REPORT_PATH,
                             config.TRANSACTION_CONSUMPTION_PATH, config.HOURLY_CONSUMPTION_PATH,
                             config.CONSUMPTION_REPORT_PATH)
    hourly = report["hourly_file"]
    print("Recipe version     :", report["recipe_version"])
    print("Sales read         :", report["input"]["sales"])
    print("Transaction rows   :", report["transaction_file"]["rows"])
    print("Hourly rows        :", hourly["rows"], f"({hourly['first_hour']} to {hourly['last_hour']})")
    print("Hours with sales   :", hourly["hours_with_sales"])
    print("Hours without sales:", hourly["hours_without_sales"])
    for column, grams in report["total_grams"].items():
        print(f"Total {column:12s}: {grams:,.0f} g")
    print("Checks passed      :", sum(report["checks"].values()), "of", len(report["checks"]))
    print("Written            :", report["transaction_file"]["path"])
    print("                    ", hourly["path"])
    print("                    ", relative_path(config.CONSUMPTION_REPORT_PATH))


if __name__ == "__main__":
    main()
