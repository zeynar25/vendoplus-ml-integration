"""
preprocessing.py - Phase 1 (raw dataset inspection) and Phase 2 (data cleaning)

What this script does:
  1. Reads the raw Coffee Sales CSV (data/raw/index_1.csv) without changing it.
  2. Inspects the raw data and records what it finds (Phase 1).
  3. Applies the approved cleaning rules (Phase 2).
  4. Writes three files to data/processed/:
       coffee_sales_cleaned.csv  - the sales kept for the next phases
       excluded_records.csv      - every removed row and why (may be empty)
       cleaning_report.json      - findings, counts, checks and the raw file's SHA-256

How to run (from the project folder):
    python preprocessing.py
"""

import hashlib
import json
import platform
from datetime import datetime

import numpy as np
import pandas as pd

import config


# =============================================================================
# Block 1. Rules and small helpers
# =============================================================================

# Exclusion rules, in the order they are checked. A row gets the FIRST reason
# that applies to it, so every excluded row has exactly one reason.
# Only problems that make a sale unusable for the simulation can exclude it.
EXCLUSION_RULES = {
    "missing_datetime": "The datetime cell is empty, so the sale cannot be placed in time.",
    "unparseable_datetime": "The datetime text is not a valid date and time.",
    "date_does_not_match_datetime": "The date column names a different day than the datetime "
                                    "column, so we cannot tell when the sale happened.",
    "missing_coffee_name": "The beverage cell is empty, so ingredient use cannot be calculated.",
    "unknown_coffee_name": "The beverage is not one of the eight menu items, so it has no recipe.",
    "outside_study_period": "The sale falls outside 1 March 2024 - 23 March 2025.",
    "exact_duplicate": "All six fields are identical to an earlier row, so the same sale "
                       "was recorded twice. The first copy is kept.",
}

# Situations that look unusual but are NOT reasons to exclude a sale.
KEPT_ON_PURPOSE = [
    "Cash sales with an empty card field (cash payments have no card identifier).",
    "Sales that share a timestamp with another sale but differ in another field.",
    "Sales made seconds apart, including repeat purchases with the same card.",
    "Days with zero sales (the data does not show whether the machine was closed).",
    "Rows whose date cell is empty or unreadable while datetime is valid (datetime is the "
    "primary timestamp; date is only a cross-check). These are counted in the inspection.",
    "Unexpected cash_type, card or money values (audit fields only). These are reported.",
]

# Fields that must all be identical for two rows to count as an exact duplicate.
DUPLICATE_KEY = ["datetime_parsed", "date_parsed", "cash_type_clean", "card",
                 "money_clean", "coffee_name_clean"]


def file_sha256(path):
    """Return the SHA-256 fingerprint of a file. Changing even one byte changes it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()  # read_bytes() only reads


def relative_path(path):
    """Path relative to the project folder, written with / so it looks the same on every OS."""
    try:
        return path.resolve().relative_to(config.PROJECT_DIR).as_posix()
    except ValueError:  # file lives outside the project folder (happens in the tests)
        return path.resolve().as_posix()


def counts_as_dict(series):
    """Turn value counts into a plain {text: whole number} dict that JSON can store."""
    return {str(key): int(count) for key, count in series.value_counts().items()}


# =============================================================================
# Block 2. Load the raw CSV exactly as it is
# =============================================================================

def load_raw_csv(raw_path):
    """
    Read every cell as plain text, so pandas does not silently guess types or
    turn words like "NA" into missing values. We convert each column ourselves
    in Block 3 and count anything that fails.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}\n"
                                f"Put index_1.csv in the data/raw folder first.")

    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)

    missing_columns = [c for c in config.EXPECTED_RAW_COLUMNS if c not in raw.columns]
    if missing_columns:
        raise ValueError(f"The raw file is missing expected columns: {missing_columns}")

    # source_row = the line number in the CSV file (the header is line 1), so any
    # excluded record can be found again by opening the raw file in a text editor.
    raw.insert(0, "source_row", range(2, len(raw) + 2))
    return raw


# =============================================================================
# Block 3. Make a working copy with parsed and standardized values
#          (no rows are removed here)
# =============================================================================

def prepare_working_copy(raw):
    work = raw.copy()  # the raw table itself is never changed

    # 3a. Remove spaces around every value and treat blank cells as missing.
    for column in config.EXPECTED_RAW_COLUMNS:
        text = work[column].str.strip()
        work[column] = text.mask(text == "")

    # 3b. Turn the timestamp text into real date-time values. "ISO8601" accepts the
    #     standard layout (2024-03-01 10:15:50.520), with or without fractions of a
    #     second. Anything else becomes NaT ("not a time") instead of crashing.
    work["datetime_parsed"] = pd.to_datetime(work["datetime"], format="ISO8601", errors="coerce")
    work["date_parsed"] = pd.to_datetime(work["date"], format="ISO8601", errors="coerce")

    # 3c. Standardize beverage names: squeeze repeated spaces, ignore upper/lower
    #     case, then map to the official menu spelling. Unknown names become NaN.
    menu_lookup = {name.lower(): name for name in config.MENU_ITEMS}
    name_key = work["coffee_name"].str.replace(r"\s+", " ", regex=True).str.lower()
    work["coffee_name_clean"] = name_key.map(menu_lookup)

    # 3d. Payment type in lower case, and the amount paid as a number.
    work["cash_type_clean"] = work["cash_type"].str.lower()
    work["money_clean"] = pd.to_numeric(work["money"], errors="coerce")
    return work


# =============================================================================
# Block 4. Phase 1: inspect the data and record what we find
#          (this block only reports; it never removes anything)
# =============================================================================

def inspect_data(raw, work):
    columns = config.EXPECTED_RAW_COLUMNS
    findings = {}

    # 4a. Structure and missing values
    findings["records"] = len(raw)
    findings["columns"] = [c for c in raw.columns if c != "source_row"]
    findings["unexpected_extra_columns"] = [c for c in findings["columns"] if c not in columns]
    findings["missing_values_per_column"] = {c: int(work[c].isna().sum()) for c in columns}
    findings["values_with_leading_or_trailing_spaces"] = {
        c: int((raw[c] != raw[c].str.strip()).sum()) for c in columns}
    findings["exact_duplicate_rows_as_written_in_file"] = int(raw.duplicated(subset=columns).sum())

    # 4b. Categories
    findings["beverage_counts_as_written"] = counts_as_dict(raw["coffee_name"])
    findings["unknown_beverage_names"] = sorted(
        work.loc[work["coffee_name"].notna() & work["coffee_name_clean"].isna(), "coffee_name"].unique())
    findings["payment_type_counts_as_written"] = counts_as_dict(raw["cash_type"])
    findings["unexpected_payment_types"] = sorted(
        work.loc[work["cash_type_clean"].notna()
                 & ~work["cash_type_clean"].isin(config.VALID_PAYMENT_TYPES), "cash_type"].unique())
    findings["missing_card_by_payment_type"] = {
        payment: int(group["card"].isna().sum())
        for payment, group in work.groupby("cash_type_clean")}

    # 4c. Timestamps
    dt = work["datetime_parsed"]
    both_dates_valid = dt.notna() & work["date_parsed"].notna()
    findings["datetime_missing_or_unparseable"] = int(dt.isna().sum())
    findings["date_missing_or_unparseable"] = int(work["date_parsed"].isna().sum())
    findings["date_differs_from_datetime_day"] = int(
        (both_dates_valid & (work["date_parsed"] != dt.dt.normalize())).sum())
    findings["first_datetime"] = str(dt.min())
    findings["last_datetime"] = str(dt.max())
    valid_times = dt.dropna()
    findings["rows_sharing_a_timestamp_with_another_row"] = int(valid_times.duplicated(keep=False).sum())
    findings["file_already_in_chronological_order"] = bool(valid_times.is_monotonic_increasing)
    findings["rows_earlier_than_the_row_above"] = int((valid_times.diff() < pd.Timedelta(0)).sum())

    # 4d. How sales are spread over time (rows with a valid timestamp only)
    valid = work[dt.notna()].sort_values("datetime_parsed")
    times = valid["datetime_parsed"]
    per_day = valid.set_index("datetime_parsed").resample("D").size()  # includes zero days
    per_month = valid.set_index("datetime_parsed").resample("MS").size()
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    findings["calendar_days_covered"] = len(per_day)
    findings["mean_sales_per_day"] = round(float(per_day.mean()), 2)
    findings["max_sales_in_one_day"] = int(per_day.max())
    findings["days_with_zero_sales"] = [day.strftime("%Y-%m-%d") for day in per_day.index[per_day == 0]]
    findings["sales_per_month"] = {m.strftime("%Y-%m"): int(n) for m, n in per_month.items()}
    findings["sales_per_hour_of_day"] = {
        int(h): int(n) for h, n in times.dt.hour.value_counts().sort_index().items()}
    findings["sales_per_weekday"] = {
        weekday_names[d]: int(n) for d, n in times.dt.dayofweek.value_counts().sort_index().items()}

    # 4e. Gaps between sales and purchases made close together
    gaps = times.diff()
    long_gap = gaps > pd.Timedelta(hours=config.LONG_GAP_HOURS)
    findings["gaps_longer_than_hours"] = config.LONG_GAP_HOURS
    findings["long_gaps_between_sales"] = [
        {"from": str(start), "to": str(end), "hours": round(gap.total_seconds() / 3600, 1)}
        for start, end, gap in zip(times.shift()[long_gap], times[long_gap], gaps[long_gap])]
    close = pd.Timedelta(seconds=config.CLOSE_PURCHASE_SECONDS)
    card_sales = valid[valid["card"].notna()]
    same_card_gap = card_sales.groupby("card")["datetime_parsed"].diff()
    findings["close_purchases"] = {
        "threshold_seconds": config.CLOSE_PURCHASE_SECONDS,
        "consecutive_sales_closer_than_threshold": int((gaps < close).sum()),
        "same_card_buying_again_within_threshold": int((same_card_gap < close).sum()),
        "shortest_gap_seconds": round(gaps.min().total_seconds(), 3),
    }

    # 4f. Amount paid (audit field). The card price history lets us see whether
    #     each beverage's price changes in a consistent way, which supports the
    #     correctness of coffee_name.
    money = work["money_clean"]
    findings["money"] = {
        "missing_or_not_a_number": int(money.isna().sum()),
        "zero_or_negative": int((money <= 0).sum()),
        "min": float(money.min()),
        "max": float(money.max()),
        "distinct_values": int(money.nunique()),
    }
    card_paid = valid[(valid["cash_type_clean"] == "card") & valid["coffee_name_clean"].notna()]
    price_history = {}
    for name, sales in card_paid.groupby("coffee_name_clean"):
        changed = sales[sales["money_clean"] != sales["money_clean"].shift()]
        price_history[name] = [
            {"first_seen": when.strftime("%Y-%m-%d"), "price": float(price)}
            for when, price in zip(changed["datetime_parsed"], changed["money_clean"])]
    findings["card_price_history_by_beverage"] = price_history
    cash_paid = valid[valid["cash_type_clean"] == "cash"]["datetime_parsed"]
    findings["cash_sales_period"] = (
        {"first": str(cash_paid.min()), "last": str(cash_paid.max())} if len(cash_paid) else None)
    return findings


# =============================================================================
# Block 5. Phase 2: decide which rows to exclude, and why
# =============================================================================

def assign_exclusion_reasons(work):
    work = work.copy()
    work["exclusion_reason"] = ""  # empty text = the row is kept

    def exclude(condition, reason):
        # Give the reason only to rows that are still kept (first matching rule wins).
        still_kept = work["exclusion_reason"] == ""
        work.loc[condition & still_kept, "exclusion_reason"] = reason

    dt = work["datetime_parsed"]
    exclude(work["datetime"].isna(), "missing_datetime")
    exclude(dt.isna(), "unparseable_datetime")

    # Only a real contradiction excludes a row: both values readable but different days.
    both_dates_valid = dt.notna() & work["date_parsed"].notna()
    exclude(both_dates_valid & (work["date_parsed"] != dt.dt.normalize()), "date_does_not_match_datetime")

    exclude(work["coffee_name"].isna(), "missing_coffee_name")
    exclude(work["coffee_name_clean"].isna(), "unknown_coffee_name")

    start = pd.Timestamp(config.STUDY_PERIOD_START)
    end = pd.Timestamp(config.STUDY_PERIOD_END_EXCLUSIVE)
    exclude((dt < start) | (dt >= end), "outside_study_period")

    # Exact duplicates: compared after standardization, so "Latte" and " latte"
    # count as the same. Rows are still in file order, so the first copy is kept.
    still_kept = work["exclusion_reason"] == ""
    is_repeat = work[still_kept].duplicated(subset=DUPLICATE_KEY, keep="first")
    exclude(is_repeat.reindex(work.index, fill_value=False), "exact_duplicate")
    return work


# =============================================================================
# Block 6. Build the cleaned table and the excluded-records table
# =============================================================================

def build_outputs(raw, work):
    kept = work[work["exclusion_reason"] == ""]

    # Sort by time. If two sales share a timestamp, keep their file order (source_row),
    # so the result is identical every time the script runs.
    cleaned = (
        kept.sort_values(["datetime_parsed", "source_row"])
        [["datetime_parsed", "coffee_name_clean", "cash_type_clean", "money_clean"]]
        .rename(columns={"datetime_parsed": "datetime", "coffee_name_clean": "coffee_name",
                         "cash_type_clean": "cash_type", "money_clean": "money"})
        .reset_index(drop=True)
    )
    cleaned = cleaned[config.CLEANED_COLUMNS]

    # Excluded rows keep their ORIGINAL text (from raw, before trimming) so they can be audited.
    is_excluded = work["exclusion_reason"] != ""
    excluded = raw.loc[is_excluded, ["source_row"] + config.EXPECTED_RAW_COLUMNS].copy()
    excluded["exclusion_reason"] = work.loc[is_excluded, "exclusion_reason"]
    return cleaned, excluded.reset_index(drop=True)


# =============================================================================
# Block 7. Final checks: every check must be True, or the script stops with an error
# =============================================================================

def run_final_checks(raw, work, cleaned, excluded):
    start = pd.Timestamp(config.STUDY_PERIOD_START)
    end = pd.Timestamp(config.STUDY_PERIOD_END_EXCLUSIVE)
    kept = work[work["exclusion_reason"] == ""]
    checks = {
        "every_raw_row_is_either_kept_or_excluded": len(cleaned) + len(excluded) == len(raw),
        "every_excluded_row_has_one_known_reason": excluded["exclusion_reason"].isin(EXCLUSION_RULES).all(),
        "cleaned_has_the_approved_columns": list(cleaned.columns) == config.CLEANED_COLUMNS,
        "no_missing_datetime_in_cleaned": cleaned["datetime"].notna().all(),
        "every_coffee_name_is_a_menu_item": cleaned["coffee_name"].isin(config.MENU_ITEMS).all(),
        "all_sales_inside_study_period": ((cleaned["datetime"] >= start) & (cleaned["datetime"] < end)).all(),
        "cleaned_is_in_chronological_order": cleaned["datetime"].is_monotonic_increasing,
        "no_exact_duplicates_left": not kept.duplicated(subset=DUPLICATE_KEY).any(),
    }
    return {name: bool(passed) for name, passed in checks.items()}


# =============================================================================
# Block 8. Run everything in order and write the three output files
# =============================================================================

def run_preprocessing(raw_path, cleaned_path, excluded_path, report_path):
    # Safety: refuse to write any output on top of the raw file.
    for output in (cleaned_path, excluded_path, report_path):
        if output.resolve() == raw_path.resolve():
            raise ValueError(f"Output path {output} would overwrite the raw file.")

    raw_hash_before = file_sha256(raw_path)

    raw = load_raw_csv(raw_path)                    # Block 2
    work = prepare_working_copy(raw)                # Block 3
    findings = inspect_data(raw, work)              # Block 4 (Phase 1)
    work = assign_exclusion_reasons(work)           # Block 5 (Phase 2)
    cleaned, excluded = build_outputs(raw, work)    # Block 6
    checks = run_final_checks(raw, work, cleaned, excluded)  # Block 7

    # Write CSVs with "\n" line endings and UTF-8 so the files are byte-identical
    # on Windows and Linux.
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False, date_format=config.OUTPUT_DATETIME_FORMAT,
                   lineterminator="\n", encoding="utf-8")
    excluded_path.parent.mkdir(parents=True, exist_ok=True)
    excluded.to_csv(excluded_path, index=False, lineterminator="\n", encoding="utf-8")

    # Prove the raw file was not touched while the script ran.
    checks["raw_file_unchanged"] = file_sha256(raw_path) == raw_hash_before

    reason_counts = excluded["exclusion_reason"].value_counts()
    # Beverage names whose written form (original text) differs from the menu spelling.
    was_renamed = work["coffee_name_clean"].notna() & (raw["coffee_name"] != work["coffee_name_clean"])
    renamed = pd.DataFrame({"old": raw.loc[was_renamed, "coffee_name"],
                            "new": work.loc[was_renamed, "coffee_name_clean"]})
    report = {
        "report_created_at": datetime.now().isoformat(timespec="seconds"),
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__},
        "raw_file": {
            "path": relative_path(raw_path),
            "sha256": raw_hash_before,
            "size_bytes": raw_path.stat().st_size,
            "download_date": config.RAW_DATA_DOWNLOAD_DATE,
            "records": len(raw),
            "columns": findings["columns"],
        },
        "inspection": findings,
        "cleaning": {
            "original_records": len(raw),
            "retained_records": len(cleaned),
            "excluded_records": len(excluded),
            "exclusions_by_reason": {r: int(reason_counts.get(r, 0)) for r in EXCLUSION_RULES},
            "exclusion_rules_in_order": EXCLUSION_RULES,
            "kept_on_purpose": KEPT_ON_PURPOSE,
            "beverage_names_changed_by_standardization": {
                f"{old!r} -> {new}": int(n)
                for (old, new), n in renamed.groupby(["old", "new"]).size().items()},
            "columns_dropped": {
                "date": "Redundant: it matches the day of datetime, which is the primary timestamp.",
                "card": "Anonymized customer identifier; not a predictive feature.",
            },
        },
        "cleaned_file": {
            "path": relative_path(cleaned_path),
            "sha256": file_sha256(cleaned_path),
            "records": len(cleaned),
            "columns": list(cleaned.columns),
            "column_meaning": {
                "datetime": "sale timestamp (YYYY-MM-DD HH:MM:SS.ffffff); used by the ML pipeline",
                "coffee_name": "one of the eight menu items; used by the ML pipeline",
                "cash_type": "card or cash; audit only",
                "money": "amount paid, currency not stated; audit only",
            },
            "pandas_dtypes": {c: str(t) for c, t in cleaned.dtypes.items()},
            "first_datetime": str(cleaned["datetime"].min()),
            "last_datetime": str(cleaned["datetime"].max()),
            "beverage_counts": counts_as_dict(cleaned["coffee_name"]),
            "payment_type_counts": counts_as_dict(cleaned["cash_type"]),
        },
        "excluded_file": {
            "path": relative_path(excluded_path),
            "sha256": file_sha256(excluded_path),
            "records": len(excluded),
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Cleaning checks failed: {failed}. See {report_path}")
    return report


def main():
    report = run_preprocessing(config.RAW_DATA_PATH, config.CLEANED_DATA_PATH,
                               config.EXCLUDED_RECORDS_PATH, config.CLEANING_REPORT_PATH)
    cleaning = report["cleaning"]
    print("Raw file        :", report["raw_file"]["path"])
    print("Raw SHA-256     :", report["raw_file"]["sha256"])
    print("Original records:", cleaning["original_records"])
    print("Retained records:", cleaning["retained_records"])
    print("Excluded records:", cleaning["excluded_records"])
    for reason, count in cleaning["exclusions_by_reason"].items():
        print(f"    {reason:30s} {count}")
    print("Checks passed   :", sum(report["checks"].values()), "of", len(report["checks"]))
    print("Written         :", report["cleaned_file"]["path"])
    print("                 ", report["excluded_file"]["path"])
    print("                 ", relative_path(config.CLEANING_REPORT_PATH))


if __name__ == "__main__":
    main()
