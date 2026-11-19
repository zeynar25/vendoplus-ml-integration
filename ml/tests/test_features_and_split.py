"""
Tests for features.py (step 5) and split.py (step 6).

Run from the project folder:
    python -m pytest -v
"""

import numpy as np
import pandas as pd
import pytest

import config
import features
import split

T = pd.Timestamp("2024-03-11 12:00:00")  # a Monday


def latte_sales(texts):
    """Sales of Latte (12 g of milk each) at the given times."""
    times = pd.to_datetime(texts, format="ISO8601")
    return pd.DataFrame({"datetime": times, "coffee_name": "Latte",
                         "coffee_g": 7, "milk_g": 12, "chocolate_g": 0})


# ---- Step 5: features -----------------------------------------------------------

def test_rate_windows_include_t_and_exclude_the_window_start_and_the_future():
    sales = latte_sales([
        "2024-03-11 11:00:00",      # exactly t - 1 h: outside the 1 h window, inside 6 h and 24 h
        "2024-03-11 11:00:00.001",  # inside all windows
        "2024-03-11 12:00:00",      # exactly t: inside all windows
        "2024-03-11 12:00:00.001",  # after t: never counted
    ])
    rates = features.consumption_rates(sales["datetime"], sales["milk_g"], [T]).iloc[0]
    assert rates["rate_1h"] == pytest.approx(24 / 1000 / 1)
    assert rates["rate_6h"] == pytest.approx(36 / 1000 / 6)
    assert rates["rate_24h"] == pytest.approx(36 / 1000 / 24)


def small_scenarios():
    rows = []
    for t in [pd.Timestamp("2024-03-11 00:00"), pd.Timestamp("2024-03-11 06:00")]:
        for hopper in config.HOPPERS:
            for stock_g in config.STOCK_LEVELS_G:
                rows.append({"decision_time": t, "hopper": hopper, "stock_g": stock_g,
                             "hours_to_shortage": np.nan, "label": "Low"})
    return pd.DataFrame(rows)


def test_time_and_hopper_features():
    table = features.add_features(small_scenarios(), latte_sales(["2024-03-10 09:00"]))
    midnight = table[table["decision_time"].dt.hour == 0].iloc[0]
    six = table[table["decision_time"].dt.hour == 6].iloc[0]
    assert (midnight["hour_sin"], midnight["hour_cos"]) == pytest.approx((0.0, 1.0))
    assert (six["hour_sin"], six["hour_cos"]) == pytest.approx((1.0, 0.0), abs=1e-12)
    assert (midnight["weekday_sin"], midnight["weekday_cos"]) == pytest.approx((0.0, 1.0))  # Monday
    flags = table[[f"hopper_{h}" for h in config.HOPPERS]]
    assert (flags.sum(axis=1) == 1).all()
    assert (table.loc[table["hopper"] == "milk", "hopper_milk"] == 1).all()
    assert list(table.columns[-11:]) == config.FEATURE_COLUMNS


def test_noise_is_reproducible_and_only_touches_the_measured_stock():
    scenarios = small_scenarios()
    sales = latte_sales(["2024-03-10 09:00"])
    first = features.add_features(scenarios, sales)
    again = features.add_features(scenarios, sales)
    other_seed = features.add_features(scenarios, sales, seed=7)
    assert first["measured_stock_g"].tolist() == again["measured_stock_g"].tolist()
    assert first["measured_stock_g"].tolist() != other_seed["measured_stock_g"].tolist()
    assert (first["stock_g"] == scenarios["stock_g"]).all()  # true stock unchanged
    assert first["stock_ratio"].tolist() == pytest.approx((first["measured_stock_g"] / 1000).tolist())


def test_future_sales_do_not_change_features():
    scenarios = small_scenarios()
    past = latte_sales(["2024-03-10 09:00", "2024-03-10 23:30"])
    with_future = pd.concat([past, latte_sales(["2024-03-11 07:00", "2024-03-12 10:00"])], ignore_index=True)
    a = features.add_features(scenarios[scenarios["decision_time"].dt.hour == 0], past)
    b = features.add_features(scenarios[scenarios["decision_time"].dt.hour == 0], with_future)
    pd.testing.assert_frame_equal(a, b)


# ---- Step 6: partitions ----------------------------------------------------------

def test_partition_boundaries():
    times = pd.to_datetime([
        "2024-03-02 23:00", "2024-03-03 00:00", "2024-10-30 23:00", "2024-10-31 00:00",
        "2024-11-01 00:00", "2024-12-30 23:00", "2024-12-31 12:00", "2025-01-01 00:00",
        "2025-03-21 23:00",
    ])
    assert split.assign_partitions(times).tolist() == [
        "excluded", "train", "train", "excluded",
        "validation", "validation", "excluded", "test",
        "test",
    ]


needs_real = pytest.mark.skipif(not config.ML_DATASET_PATH.exists(), reason="Run split.py first.")


@needs_real
def test_real_dataset_partitions():
    data = pd.read_csv(config.ML_DATASET_PATH, parse_dates=["decision_time"])
    counts = data["partition"].value_counts().to_dict()
    assert counts == {"train": 87120, "test": 28800, "validation": 21600}
    assert data.groupby("decision_time")["partition"].nunique().max() == 1
    assert not data[config.FEATURE_COLUMNS].isna().any().any()
    last_train = data.loc[data["partition"] == "train", "decision_time"].max()
    first_val = data.loc[data["partition"] == "validation", "decision_time"].min()
    assert last_train + pd.Timedelta(hours=24) < first_val
