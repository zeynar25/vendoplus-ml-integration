"""
Tests for stock_simulation.py (step 3) and labels.py (step 4).

The boundary cases use one decision time, t = 2024-03-10 00:00, a coffee
hopper holding 10 g, and espressos that need 7 g each. After the first
espresso 3 g are left, so the SECOND espresso is the shortage.

Run from the project folder:
    python -m pytest -v
"""

import numpy as np
import pandas as pd
import pytest

import config
import labels
import stock_simulation as sim

T = pd.Timestamp("2024-03-10 00:00:00")


def shortage(sale_texts, grams, stock_g=10, t=T):
    """Run the fast replay for one decision time and return the shortage time (or NaT)."""
    sale_times = pd.to_datetime(sale_texts, format="ISO8601")
    return sim.first_shortage_times(sale_times, grams, pd.DatetimeIndex([t]), stock_g)[0]


def hours_after_t(when):
    return (when - T) / pd.Timedelta(hours=1)


# ---- Step 3: the replay -------------------------------------------------------

def test_shortage_exactly_at_4_hours_is_found():
    when = shortage(["2024-03-10 01:00", "2024-03-10 04:00"], [7, 7])
    assert hours_after_t(when) == 4


def test_shortage_exactly_at_24_hours_counts():
    when = shortage(["2024-03-10 01:00", "2024-03-11 00:00"], [7, 7])
    assert hours_after_t(when) == 24


def test_shortage_just_after_24_hours_does_not_count():
    when = shortage(["2024-03-10 01:00", "2024-03-11 00:00:00.001"], [7, 7])
    assert pd.isna(when)


def test_sale_exactly_at_t_belongs_to_the_past():
    # The 00:00 sale is before the decision, so only the 01:00 sale is replayed: 10 - 7 = 3 g left.
    when = shortage(["2024-03-10 00:00", "2024-03-10 01:00"], [7, 7])
    assert pd.isna(when)


def test_stock_equal_to_the_need_is_enough():
    # 7 g in the hopper serves exactly one 7 g espresso; the next one is short.
    when = shortage(["2024-03-10 01:00", "2024-03-10 02:00"], [7, 7], stock_g=7)
    assert hours_after_t(when) == 2


def test_drinks_without_this_ingredient_are_ignored():
    # 0 g sales (e.g. Cocoa for the coffee hopper) never cause a coffee shortage.
    when = shortage(["2024-03-10 01:00", "2024-03-10 02:00", "2024-03-10 03:00"], [0, 7, 0])
    assert pd.isna(when)


def test_fast_replay_agrees_with_slow_replay_on_random_data():
    rng = np.random.default_rng(0)
    sale_times = pd.Timestamp("2024-03-10") + pd.to_timedelta(np.sort(rng.uniform(0, 72, 150)), unit="h")
    grams = rng.choice([0, 6, 7, 8, 10, 12, 20], size=150)
    times = pd.date_range("2024-03-10", "2024-03-11 23:00", freq="h")
    for stock_g in [5, 10, 50, 100]:
        fast = sim.first_shortage_times(sale_times, grams, times, stock_g)
        for k, t in enumerate(times):
            slow = sim.slow_replay(sale_times, grams, t, stock_g)
            assert (pd.isna(slow) and pd.isna(fast[k])) or slow == fast[k]


def test_complete_windows_need_24_hours_before_and_after():
    times = pd.DatetimeIndex(["2024-03-01 23:00", "2024-03-02 00:00",
                              "2025-03-23 00:00", "2025-03-23 01:00"])
    assert sim.window_is_complete(times).tolist() == [False, True, True, False]


# ---- Step 4: the labels -------------------------------------------------------

def test_label_boundaries():
    hours = [np.nan, 0.5, 4.0, 4.0001, 23.99, 24.0]
    assert labels.assign_labels(hours).tolist() == ["Low", "High", "High", "Medium", "Medium", "Medium"]


def test_replay_and_labels_together():
    four_h = hours_after_t(shortage(["2024-03-10 01:00", "2024-03-10 04:00"], [7, 7]))
    just_after_four = hours_after_t(shortage(["2024-03-10 01:00", "2024-03-10 04:00:00.001"], [7, 7]))
    none = shortage(["2024-03-10 01:00"], [7])
    assert labels.assign_labels([four_h, just_after_four, np.nan if pd.isna(none) else 0]).tolist() == \
        ["High", "Medium", "Low"]


# ---- Real outputs (skipped until the steps have run) ---------------------------

needs_real = pytest.mark.skipif(not config.LABELED_SCENARIOS_PATH.exists(),
                                reason="Run stock_simulation.py and labels.py first.")


@needs_real
def test_real_labeled_scenarios():
    table = pd.read_csv(config.LABELED_SCENARIOS_PATH)
    assert len(table) == 9216 * 3 * 5
    assert table["label"].isin(config.LABELS).all()
    high = table[table["label"] == "High"]
    assert (high["hours_to_shortage"] <= 4).all()
    # 800 g can never run short within 24 h in this data.
    assert (table.loc[table["stock_g"] == 800, "label"] == "Low").all()
