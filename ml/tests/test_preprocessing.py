"""
Tests for preprocessing.py

Part A uses a tiny hand-made CSV with every problem planted on purpose, so we
know exactly what should happen to each row.

Part B checks the real output files made by `python preprocessing.py` against
the raw CSV. It re-reads everything with plain pandas, independently of the
cleaning code, so the two have to agree.

Run from the project folder:
    python -m pytest -v
"""

import hashlib
import json

import pandas as pd
import pytest

import config
import preprocessing


# =============================================================================
# Part A. Hand-made sample with known problems
# =============================================================================

# The line number of each row in this text is its source_row (header = line 1).
SAMPLE_CSV = (
    "date,datetime,cash_type,card,money,coffee_name\n"
    "2024-03-02,2024-03-02 10:00:00.000,card,ANON-0001,38.7,Latte\n"                 # 2 valid
    "2024-03-02,2024-03-02 09:00:00.000,cash,,40,Americano\n"                         # 3 cash, no card: keep
    "2024-03-02,2024-03-02 11:00:00.000,card,ANON-0002,38.7,  latte \n"               # 4 messy name: keep as Latte
    "2024-03-02,2024-03-02 12:00:00.000,card,ANON-0003,38.7,Matcha\n"                 # 5 not on the menu
    "2024-03-02,not a time,card,ANON-0004,38.7,Latte\n"                               # 6 unreadable datetime
    "2024-03-02,,card,ANON-0005,38.7,Latte\n"                                         # 7 empty datetime
    "2024-03-05,2024-03-06 08:00:00.000,card,ANON-0006,38.7,Latte\n"                  # 8 date vs datetime differ
    "2025-04-01,2025-04-01 08:00:00.000,card,ANON-0007,38.7,Latte\n"                  # 9 after the study period
    "2024-03-02,2024-03-02 10:00:00.000,card,ANON-0001,38.7,Latte\n"                  # 10 exact copy of line 2
    "2024-03-02,2024-03-02 10:00:00.000,card,ANON-0008,38.7,Cocoa\n"                  # 11 same time, other sale: keep
    "2024-03-02,2024-03-02 13:00:00.000,card,ANON-0009,33.8,Americano With  Milk\n"   # 12 case/spaces: keep
    "2025-03-23,2025-03-23 23:59:59.999,card,ANON-0010,28.9,Cortado\n"                # 13 last moment inside
    "2024-03-02,2024-03-02 14:00:00.000,card,ANON-0011,38.7,\n"                       # 14 empty beverage
    "2025-03-24,2025-03-24 00:00:00.000,card,ANON-0012,38.7,Latte\n"                  # 15 first moment outside
    "2024-03-01,2024-03-01 00:00:00.000,card,ANON-0013,24,Espresso\n"                 # 16 first moment inside
    ",2024-03-03 09:30:00.000,card,ANON-0014,38.7,Cappuccino\n"                       # 17 empty date only: keep
)

EXPECTED_EXCLUSIONS = {
    5: "unknown_coffee_name",
    6: "unparseable_datetime",
    7: "missing_datetime",
    8: "date_does_not_match_datetime",
    9: "outside_study_period",
    10: "exact_duplicate",
    14: "missing_coffee_name",
    15: "outside_study_period",
}

# The kept rows, in the order they should appear in the cleaned file.
EXPECTED_CLEANED_NAMES = [
    "Espresso",             # line 16, 2024-03-01 00:00
    "Americano",            # line 3,  2024-03-02 09:00
    "Latte",                # line 2,  10:00 (comes before line 11: same time, earlier in file)
    "Cocoa",                # line 11, 10:00
    "Latte",                # line 4,  11:00
    "Americano with Milk",  # line 12, 13:00
    "Cappuccino",           # line 17, 2024-03-03
    "Cortado",              # line 13, 2025-03-23 23:59:59.999
]


@pytest.fixture
def sample_run(tmp_path):
    """Write the sample CSV to a temporary folder and run the cleaning on it."""
    raw_path = tmp_path / "raw.csv"
    raw_path.write_bytes(SAMPLE_CSV.encode("utf-8"))
    raw_bytes_before = raw_path.read_bytes()

    paths = {
        "raw": raw_path,
        "cleaned": tmp_path / "processed" / "cleaned.csv",
        "excluded": tmp_path / "processed" / "excluded.csv",
        "report": tmp_path / "processed" / "report.json",
    }
    report = preprocessing.run_preprocessing(paths["raw"], paths["cleaned"],
                                             paths["excluded"], paths["report"])
    return {
        "paths": paths,
        "report": report,
        "raw_bytes_before": raw_bytes_before,
        "cleaned": pd.read_csv(paths["cleaned"]),
        "excluded": pd.read_csv(paths["excluded"]),
    }


def test_each_problem_row_gets_the_expected_reason(sample_run):
    excluded = sample_run["excluded"]
    actual = dict(zip(excluded["source_row"], excluded["exclusion_reason"]))
    assert actual == EXPECTED_EXCLUSIONS


def test_valid_rows_are_kept_sorted_and_standardized(sample_run):
    assert sample_run["cleaned"]["coffee_name"].tolist() == EXPECTED_CLEANED_NAMES


def test_cash_sale_without_card_is_kept(sample_run):
    cleaned = sample_run["cleaned"]
    assert (cleaned["cash_type"] == "cash").sum() == 1


def test_cleaned_file_has_only_the_approved_columns(sample_run):
    assert sample_run["cleaned"].columns.tolist() == config.CLEANED_COLUMNS


def test_timestamps_survive_the_round_trip(sample_run):
    # The last kept sale must still be exactly 23:59:59.999 after writing and reading back.
    times = pd.to_datetime(sample_run["cleaned"]["datetime"], format=config.OUTPUT_DATETIME_FORMAT)
    assert times.iloc[-1] == pd.Timestamp("2025-03-23 23:59:59.999")
    assert times.is_monotonic_increasing


def test_raw_file_is_not_modified(sample_run):
    assert sample_run["paths"]["raw"].read_bytes() == sample_run["raw_bytes_before"]
    assert sample_run["report"]["checks"]["raw_file_unchanged"]


def test_report_counts_add_up(sample_run):
    cleaning = sample_run["report"]["cleaning"]
    assert cleaning["original_records"] == 16
    assert cleaning["retained_records"] == len(EXPECTED_CLEANED_NAMES)
    assert cleaning["excluded_records"] == len(EXPECTED_EXCLUSIONS)
    assert sum(cleaning["exclusions_by_reason"].values()) == cleaning["excluded_records"]
    assert sample_run["report"]["all_checks_passed"]


def test_inspection_notices_planted_issues(sample_run):
    inspection = sample_run["report"]["inspection"]
    assert inspection["unknown_beverage_names"] == ["Matcha"]
    assert inspection["exact_duplicate_rows_as_written_in_file"] == 1
    assert inspection["date_differs_from_datetime_day"] == 1
    assert inspection["missing_card_by_payment_type"] == {"card": 0, "cash": 1}


# =============================================================================
# Part B. The real output files (skipped until `python preprocessing.py` has run)
# =============================================================================

real_outputs_exist = config.CLEANED_DATA_PATH.exists() and config.CLEANING_REPORT_PATH.exists()
needs_real_outputs = pytest.mark.skipif(not real_outputs_exist,
                                        reason="Run `python preprocessing.py` first.")


@pytest.fixture(scope="module")
def real():
    with open(config.CLEANING_REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)
    raw = pd.read_csv(config.RAW_DATA_PATH)  # plain read, independent of preprocessing.py
    raw["source_row"] = range(2, len(raw) + 2)
    cleaned = pd.read_csv(config.CLEANED_DATA_PATH)
    cleaned["datetime"] = pd.to_datetime(cleaned["datetime"], format=config.OUTPUT_DATETIME_FORMAT)
    excluded = pd.read_csv(config.EXCLUDED_RECORDS_PATH)
    return report, raw, cleaned, excluded


@needs_real_outputs
def test_report_hashes_match_the_files(real):
    report = real[0]
    assert hashlib.sha256(config.RAW_DATA_PATH.read_bytes()).hexdigest() == report["raw_file"]["sha256"]
    assert hashlib.sha256(config.CLEANED_DATA_PATH.read_bytes()).hexdigest() == report["cleaned_file"]["sha256"]


@needs_real_outputs
def test_report_counts_match_the_files(real):
    report, raw, cleaned, excluded = real
    cleaning = report["cleaning"]
    assert cleaning["original_records"] == len(raw)
    assert cleaning["retained_records"] == len(cleaned)
    assert cleaning["excluded_records"] == len(excluded)
    assert len(cleaned) + len(excluded) == len(raw)
    assert sum(cleaning["exclusions_by_reason"].values()) == len(excluded)
    assert report["all_checks_passed"]


@needs_real_outputs
def test_report_summary_matches_the_cleaned_file(real):
    report, _, cleaned, _ = real
    summary = report["cleaned_file"]
    assert summary["columns"] == config.CLEANED_COLUMNS == cleaned.columns.tolist()
    assert summary["first_datetime"] == str(cleaned["datetime"].min())
    assert summary["last_datetime"] == str(cleaned["datetime"].max())
    assert summary["beverage_counts"] == cleaned["coffee_name"].value_counts().to_dict()


@needs_real_outputs
def test_cleaned_values_are_valid(real):
    cleaned = real[2]
    assert cleaned["datetime"].notna().all()
    assert cleaned["datetime"].is_monotonic_increasing
    assert cleaned["coffee_name"].isin(config.MENU_ITEMS).all()
    assert cleaned["datetime"].min() >= pd.Timestamp(config.STUDY_PERIOD_START)
    assert cleaned["datetime"].max() < pd.Timestamp(config.STUDY_PERIOD_END_EXCLUSIVE)


@needs_real_outputs
def test_cleaned_rows_are_exactly_the_kept_raw_rows(real):
    # Take the raw rows that were not excluded, sort them the same way, and
    # compare them value by value with the cleaned file.
    _, raw, cleaned, excluded = real
    kept_raw = raw[~raw["source_row"].isin(excluded["source_row"])].copy()
    kept_raw["datetime"] = pd.to_datetime(kept_raw["datetime"], format="ISO8601")
    kept_raw = kept_raw.sort_values(["datetime", "source_row"]).reset_index(drop=True)

    assert (kept_raw["datetime"] == cleaned["datetime"]).all()
    assert (kept_raw["coffee_name"].str.strip().str.lower() == cleaned["coffee_name"].str.lower()).all()
    assert (kept_raw["cash_type"].str.strip().str.lower() == cleaned["cash_type"]).all()
    assert (kept_raw["money"] == cleaned["money"]).all()
