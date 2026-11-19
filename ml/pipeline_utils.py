"""
pipeline_utils.py - small helpers shared by the pipeline steps from step 3 on.

Every step reads the output of the step before it. To make sure it uses
exactly the file that step produced (not an edited copy), each step checks
the file's SHA-256 fingerprint against the one recorded in that step's report.
"""

import hashlib
import json

import numpy as np
import pandas as pd

import config


def file_sha256(path):
    """Return the SHA-256 fingerprint of a file (reading only)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(path):
    """Path relative to the project folder, written with / on every OS."""
    try:
        return path.resolve().relative_to(config.PROJECT_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def verify_input(data_path, report_path, report_key):
    """
    Stop if data_path is not the file recorded in report_path[report_key]["sha256"].
    report_path=None skips the check (used by the tests with small made-up files).
    """
    if report_path is None:
        return
    with open(report_path, encoding="utf-8") as f:
        expected = json.load(f)[report_key]["sha256"]
    if file_sha256(data_path) != expected:
        raise ValueError(f"{data_path} does not match the SHA-256 recorded in {report_path}. "
                         f"Re-run the earlier steps instead of editing files by hand.")


def save_csv(table, path, datetime_format="%Y-%m-%d %H:%M:%S"):
    """Write a CSV the same way on every OS (UTF-8, \\n line endings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n", encoding="utf-8",
                 date_format=datetime_format)


def _to_plain_python(value):
    """Let json.dump store numpy numbers and timestamps."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    raise TypeError(f"Cannot store {type(value)} in JSON")


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_to_plain_python)


def stop_if_any_check_failed(checks, report_path):
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Checks failed: {failed}. See {report_path}")


def load_transactions(path=None, report_path=None):
    """Read transaction_consumption.csv (step 2 output) with parsed timestamps."""
    path = path or config.TRANSACTION_CONSUMPTION_PATH
    verify_input(path, report_path, "transaction_file")
    transactions = pd.read_csv(path)
    transactions["datetime"] = pd.to_datetime(transactions["datetime"],
                                              format=config.OUTPUT_DATETIME_FORMAT)
    return transactions
