"""
config.py

Fixed settings for the Vendo+ ML pipeline, kept in one place so that every
assumption used in the thesis can be traced to a single file.

So far this file holds what cleaning (preprocessing.py) and ingredient
consumption (consumption.py) need. Later phases will add hopper capacity,
stock levels, label horizons, noise settings, split dates and
hyperparameter grids here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# File locations (built from this file's folder, so the scripts work no matter
# which folder you run them from)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "index_1.csv"

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "coffee_sales_cleaned.csv"
EXCLUDED_RECORDS_PATH = PROCESSED_DIR / "excluded_records.csv"
CLEANING_REPORT_PATH = PROCESSED_DIR / "cleaning_report.json"

TRANSACTION_CONSUMPTION_PATH = PROCESSED_DIR / "transaction_consumption.csv"
HOURLY_CONSUMPTION_PATH = PROCESSED_DIR / "hourly_consumption.csv"
CONSUMPTION_REPORT_PATH = PROCESSED_DIR / "consumption_report.json"

# The date you downloaded index_1.csv from Kaggle, written as "YYYY-MM-DD".
# The source dataset is still being updated, so this date (together with the
# SHA-256 hash in the cleaning report) identifies the exact version used.
# Leave it as None if you do not know it; the report will show null.
RAW_DATA_DOWNLOAD_DATE = None

# ---------------------------------------------------------------------------
# What the raw file is expected to contain
# ---------------------------------------------------------------------------
EXPECTED_RAW_COLUMNS = ["date", "datetime", "cash_type", "card", "money", "coffee_name"]

# ---------------------------------------------------------------------------
# Recipe table: grams taken from each hopper for ONE sale of each beverage.
# This is the ONLY place in the code where recipe quantities are defined.
#
# These are simulation inputs, NOT measurements of the Kaggle machine.
# Ingredient forms:
#   coffee    = ground coffee, ground from whole beans in the coffee hopper
#   milk      = vending milk powder
#   chocolate = vending cocoa / chocolate drink powder
# Values assume a cup of about 150 ml.
# Sources, confidence levels and the reason for every value are in
# docs/decision_log.md (decision D-24).
#   LD = literature-derived, RD = researcher-defined assumption
# ---------------------------------------------------------------------------
RECIPE_VERSION = "R2 (2026-11-17)"

HOPPERS = ["coffee", "milk", "chocolate"]

RECIPES_G = {
    #                       coffee         milk           chocolate
    "Espresso":            {"coffee": 7, "milk": 0,  "chocolate": 0},   # coffee LD
    "Americano":           {"coffee": 7, "milk": 0,  "chocolate": 0},   # coffee LD
    "Americano with Milk": {"coffee": 7, "milk": 6,  "chocolate": 0},   # coffee LD, milk LD
    "Cortado":             {"coffee": 7, "milk": 6,  "chocolate": 0},   # coffee LD, milk RD
    "Cappuccino":          {"coffee": 7, "milk": 10, "chocolate": 0},   # coffee LD, milk LD
    "Latte":               {"coffee": 7, "milk": 12, "chocolate": 0},   # coffee LD, milk LD
    "Cocoa":               {"coffee": 0, "milk": 6,  "chocolate": 20},  # milk RD, chocolate LD
    "Hot Chocolate":       {"coffee": 0, "milk": 8,  "chocolate": 20},  # milk RD, chocolate LD
}

# The eight beverages on the machine's menu (canonical spelling).
# Taken from the recipe table so the menu and the recipes can never disagree.
MENU_ITEMS = list(RECIPES_G)

# Payment types we expect. cash_type is an audit field only: an unexpected
# value is reported, but the sale is NOT excluded because of it.
VALID_PAYMENT_TYPES = ["card", "cash"]

# Study period: 1 March 2024 00:00 up to (but not including) 24 March 2025 00:00,
# i.e. everything on 23 March 2025 is still inside the period.
STUDY_PERIOD_START = "2024-03-01 00:00:00"
STUDY_PERIOD_END_EXCLUSIVE = "2025-03-24 00:00:00"

# ---------------------------------------------------------------------------
# Cleaned output
# ---------------------------------------------------------------------------
# Only datetime and coffee_name feed the ML data generation.
# cash_type and money are kept for auditing and consistency checks.
CLEANED_COLUMNS = ["datetime", "coffee_name", "cash_type", "money"]

# How timestamps are written to the cleaned CSV (microseconds, so nothing is lost).
OUTPUT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# ---------------------------------------------------------------------------
# Inspection thresholds (used only for reporting, never for removing rows)
# ---------------------------------------------------------------------------
LONG_GAP_HOURS = 36          # report gaps between sales longer than this
CLOSE_PURCHASE_SECONDS = 60  # report sales made less than this many seconds apart

# ===========================================================================
# Steps 3-4: hopper stock simulation and urgency labels
# ===========================================================================
SCENARIOS_PATH = PROCESSED_DIR / "stock_scenarios.csv"
SCENARIOS_REPORT_PATH = PROCESSED_DIR / "stock_simulation_report.json"
LABELED_SCENARIOS_PATH = PROCESSED_DIR / "labeled_scenarios.csv"
LABELS_REPORT_PATH = PROCESSED_DIR / "labels_report.json"

HOPPER_CAPACITY_G = 1000                # A-03
STOCK_LEVELS_G = [10, 50, 100, 200, 800]  # D-08 (reviewed with training data, D-33)

# Hourly decision times, both ends included.
DECISION_START = "2024-03-03 00:00:00"
DECISION_END = "2025-03-21 23:00:00"

# Label rules (A-04, D-30). Future sales in (t, t + 24 h] are replayed with no refill.
HIGH_HORIZON_HOURS = 4    # shortage at or before t + 4 h      -> High
LABEL_HORIZON_HOURS = 24  # shortage after 4 h, up to t + 24 h -> Medium; none -> Low
LABELS = ["Low", "Medium", "High"]

# ===========================================================================
# Step 5: features
# ===========================================================================
FEATURES_PATH = PROCESSED_DIR / "features.csv"
FEATURES_REPORT_PATH = PROCESSED_DIR / "features_report.json"

NOISE_SD_G = 2.0       # A-06, applied to every partition (D-09)
RANDOM_SEED = 42
RATE_WINDOWS_HOURS = [1, 6, 24]  # consumption over (t - window, t]

FEATURE_COLUMNS = [
    "stock_ratio",
    "rate_1h", "rate_6h", "rate_24h",
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "hopper_coffee", "hopper_milk", "hopper_chocolate",
]

# ===========================================================================
# Step 6: chronological partitions (dates of the decision time, both ends included)
# ===========================================================================
ML_DATASET_PATH = PROCESSED_DIR / "ml_dataset.csv"
SPLIT_REPORT_PATH = PROCESSED_DIR / "split_report.json"

PARTITIONS = {
    "train":      ("2024-03-03", "2024-10-30"),
    "validation": ("2024-11-01", "2024-12-30"),
    "test":       ("2025-01-01", "2025-03-21"),
}
# 31 October and 31 December are left out so no label window crosses into the next partition.

# ===========================================================================
# Step 7: models and validation-based selection
# ===========================================================================
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
VALIDATION_RESULTS_PATH = RESULTS_DIR / "validation_results.csv"
SELECTION_PATH = RESULTS_DIR / "selection.json"

LOGISTIC_REGRESSION_GRID = [{"C": c} for c in [0.1, 1, 10]]
RANDOM_FOREST_GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
    for n in [100, 200] for d in [8, 16] for leaf in [5, 20]
]
HIST_GRADIENT_BOOSTING_GRID = [{"max_leaf_nodes": m} for m in [15, 31]]

TARGET_MACRO_F1 = 0.80
TARGET_HIGH_RECALL = 0.90

# ===========================================================================
# Step 8: baselines and the single test evaluation
# ===========================================================================
TEST_RESULTS_PATH = RESULTS_DIR / "test_results.csv"
EVALUATION_REPORT_PATH = RESULTS_DIR / "evaluation_report.json"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"

STOCK_BASELINE_HIGH_MAX_G = 50     # High at or below 5% of capacity
STOCK_BASELINE_MEDIUM_MAX_G = 200  # Medium above 5% and at or below 20%
