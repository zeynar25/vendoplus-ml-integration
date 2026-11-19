"""
Tests for consumption.py

Part A uses tiny hand-made sales whose grams can be checked with pen and paper.
Part B checks the real output files made by `python consumption.py`,
recomputing everything a second, simpler way.

Run from the project folder:
    python -m pytest -v
"""

import hashlib
import json

import pandas as pd
import pytest

import config
import consumption

# This copy of the approved recipe table is deliberate. The pipeline reads ONLY
# config.RECIPES_G; this test fails if someone edits config.py without also
# updating the approved table here and in docs/decision_log.md (D-24).
APPROVED_RECIPES = {
    "Espresso":            (7, 0, 0),
    "Americano":           (7, 0, 0),
    "Americano with Milk": (7, 6, 0),
    "Cortado":             (7, 6, 0),
    "Cappuccino":          (7, 10, 0),
    "Latte":               (7, 12, 0),
    "Cocoa":               (0, 6, 20),
    "Hot Chocolate":       (0, 8, 20),
}


def make_sales(rows):
    """Build a small sales table from (timestamp text, beverage) pairs."""
    sales = pd.DataFrame(rows, columns=["datetime", "coffee_name"])
    sales["datetime"] = pd.to_datetime(sales["datetime"])
    return sales


# =============================================================================
# Part A. Hand-made examples
# =============================================================================

def test_config_recipes_are_the_approved_recipes():
    actual = {name: (r["coffee"], r["milk"], r["chocolate"]) for name, r in config.RECIPES_G.items()}
    assert actual == APPROVED_RECIPES


def test_menu_is_taken_from_the_recipe_table():
    assert config.MENU_ITEMS == list(config.RECIPES_G)
    for recipe in config.RECIPES_G.values():
        assert set(recipe) == set(config.HOPPERS)
        assert all(grams >= 0 for grams in recipe.values())


def test_every_beverage_maps_to_its_own_recipe():
    sales = make_sales([("2024-03-02 09:00:00", name) for name in config.MENU_ITEMS])
    transactions = consumption.calculate_transaction_consumption(sales)
    for _, row in transactions.iterrows():
        expected = APPROVED_RECIPES[row["coffee_name"]]
        assert (row["coffee_g"], row["milk_g"], row["chocolate_g"]) == expected


def test_missing_ingredients_give_zero_not_blank():
    sales = make_sales([("2024-03-02 09:00:00", "Espresso"), ("2024-03-02 09:10:00", "Cocoa")])
    transactions = consumption.calculate_transaction_consumption(sales)
    espresso, cocoa = transactions.iloc[0], transactions.iloc[1]
    assert espresso["milk_g"] == 0 and espresso["chocolate_g"] == 0
    assert cocoa["coffee_g"] == 0
    assert not transactions[consumption.GRAM_COLUMNS].isna().any().any()


def test_unknown_beverage_is_reported_not_given_zero():
    sales = make_sales([("2024-03-02 09:00:00", "Latte"), ("2024-03-02 09:05:00", "Matcha")])
    with pytest.raises(ValueError, match="Matcha"):
        consumption.calculate_transaction_consumption(sales)


# The worked example from the explanation, small enough to check by hand:
#
#   sale time        beverage        coffee  milk  chocolate
#   09:05:00.000     Latte              7     12      0
#   09:40:00.000     Espresso           7      0      0
#   09:59:59.999     Cocoa              0      6     20
#   10:00:00.000     Cappuccino         7     10      0   <- exactly 10:00 belongs to the 10:00 hour
#   12:30:00.000     Hot Chocolate      0      8     20
#
#   hour     sales  coffee  milk  chocolate
#   09:00      3      14     18      20
#   10:00      1       7     10       0
#   11:00      0       0      0       0   <- no sales: zeros, not blanks
#   12:00      1       0      8      20
#   total      5      21     36      40
EXAMPLE_SALES = [
    ("2024-03-02 09:05:00.000", "Latte"),
    ("2024-03-02 09:40:00.000", "Espresso"),
    ("2024-03-02 09:59:59.999", "Cocoa"),
    ("2024-03-02 10:00:00.000", "Cappuccino"),
    ("2024-03-02 12:30:00.000", "Hot Chocolate"),
]
EXPECTED_HOURLY = pd.DataFrame({
    "hour_start": pd.to_datetime(["2024-03-02 09:00", "2024-03-02 10:00",
                                  "2024-03-02 11:00", "2024-03-02 12:00"]),
    "transactions": [3, 1, 0, 1],
    "coffee_g": [14, 7, 0, 0],
    "milk_g": [18, 10, 0, 8],
    "chocolate_g": [20, 0, 0, 20],
})


@pytest.fixture
def example():
    transactions = consumption.calculate_transaction_consumption(make_sales(EXAMPLE_SALES))
    hourly = consumption.aggregate_hourly(transactions, "2024-03-02 09:00:00", "2024-03-02 13:00:00")
    return transactions, hourly


def test_worked_example_hourly_values(example):
    _, hourly = example
    for column in EXPECTED_HOURLY.columns:
        assert hourly[column].tolist() == EXPECTED_HOURLY[column].tolist(), column


def test_worked_example_totals_are_preserved(example):
    transactions, hourly = example
    assert hourly["transactions"].sum() == len(transactions) == 5
    assert hourly["coffee_g"].sum() == transactions["coffee_g"].sum() == 21
    assert hourly["milk_g"].sum() == transactions["milk_g"].sum() == 36
    assert hourly["chocolate_g"].sum() == transactions["chocolate_g"].sum() == 40


def test_worked_example_no_sale_hour_is_zero(example):
    _, hourly = example
    eleven = hourly[hourly["hour_start"] == pd.Timestamp("2024-03-02 11:00")].iloc[0]
    assert eleven["transactions"] == 0
    assert eleven[consumption.GRAM_COLUMNS].tolist() == [0, 0, 0]


def test_no_sale_is_lost_or_duplicated(example):
    transactions, _ = example
    assert transactions["transaction_id"].tolist() == [1, 2, 3, 4, 5]


def test_full_run_writes_files_and_refuses_an_edited_input(tmp_path):
    cleaned = tmp_path / "cleaned.csv"
    make_sales(EXAMPLE_SALES).to_csv(cleaned, index=False, date_format=config.OUTPUT_DATETIME_FORMAT)
    outputs = [tmp_path / "tx.csv", tmp_path / "hourly.csv", tmp_path / "report.json"]

    report = consumption.run_consumption(cleaned, None, *outputs,
                                         period_start="2024-03-02 00:00:00",
                                         period_end_exclusive="2024-03-03 00:00:00")
    assert report["all_checks_passed"]
    assert report["hourly_file"]["rows"] == 24
    assert report["total_grams"] == {"coffee_g": 21.0, "milk_g": 36.0, "chocolate_g": 40.0}

    # A cleaning report whose fingerprint does not match the file must be refused.
    fake_report = tmp_path / "cleaning_report.json"
    fake_report.write_text(json.dumps({"cleaned_file": {"sha256": "0" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        consumption.run_consumption(cleaned, fake_report, *outputs)


# =============================================================================
# Part B. The real output files (skipped until `python consumption.py` has run)
# =============================================================================

real_outputs_exist = (config.TRANSACTION_CONSUMPTION_PATH.exists()
                      and config.HOURLY_CONSUMPTION_PATH.exists()
                      and config.CONSUMPTION_REPORT_PATH.exists())
needs_real_outputs = pytest.mark.skipif(not real_outputs_exist,
                                        reason="Run `python consumption.py` first.")


@pytest.fixture(scope="module")
def real():
    with open(config.CONSUMPTION_REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)
    cleaned = pd.read_csv(config.CLEANED_DATA_PATH)
    transactions = pd.read_csv(config.TRANSACTION_CONSUMPTION_PATH)
    hourly = pd.read_csv(config.HOURLY_CONSUMPTION_PATH)
    return report, cleaned, transactions, hourly


@needs_real_outputs
def test_real_report_hashes_match_the_files(real):
    report = real[0]
    for key, path in [("transaction_file", config.TRANSACTION_CONSUMPTION_PATH),
                      ("hourly_file", config.HOURLY_CONSUMPTION_PATH)]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report[key]["sha256"]


@needs_real_outputs
def test_real_one_row_per_sale_in_the_same_order(real):
    _, cleaned, transactions, _ = real
    assert len(transactions) == len(cleaned)
    assert transactions["transaction_id"].tolist() == list(range(1, len(cleaned) + 1))
    assert (transactions["datetime"] == cleaned["datetime"]).all()
    assert (transactions["coffee_name"] == cleaned["coffee_name"]).all()


@needs_real_outputs
def test_real_grams_match_a_plain_python_lookup(real):
    # Recompute every sale with an ordinary loop, independent of consumption.py.
    _, cleaned, transactions, _ = real
    for column, hopper in zip(["coffee_g", "milk_g", "chocolate_g"], config.HOPPERS):
        expected = [config.RECIPES_G[name][hopper] for name in cleaned["coffee_name"]]
        assert transactions[column].tolist() == expected


@needs_real_outputs
def test_real_hourly_totals_match_transactions(real):
    report, _, transactions, hourly = real
    assert hourly["transactions"].sum() == len(transactions)
    for column in ["coffee_g", "milk_g", "chocolate_g"]:
        assert hourly[column].sum() == transactions[column].sum() == report["total_grams"][column]


@needs_real_outputs
def test_real_hourly_series_is_complete_and_zero_filled(real):
    _, _, transactions, hourly = real
    hours = pd.to_datetime(hourly["hour_start"])
    expected = pd.date_range(config.STUDY_PERIOD_START,
                             pd.Timestamp(config.STUDY_PERIOD_END_EXCLUSIVE) - pd.Timedelta(hours=1),
                             freq="h")
    assert hours.tolist() == expected.tolist()
    assert not hourly.isna().any().any()
    empty = hourly[hourly["transactions"] == 0]
    assert (empty[["coffee_g", "milk_g", "chocolate_g"]] == 0).all().all()
    # Hours with sales = the number of different clock hours that appear in the transactions.
    sale_hours = pd.to_datetime(transactions["datetime"]).dt.floor("h").nunique()
    assert (hourly["transactions"] > 0).sum() == sale_hours
