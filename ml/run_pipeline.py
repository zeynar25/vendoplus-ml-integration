"""
run_pipeline.py - run every step of the Vendo+ ML pipeline in order.

    1. preprocessing.py     raw CSV -> cleaned sales
    2. consumption.py       sales -> grams per hopper (per sale and per hour)
    3. stock_simulation.py  no-refill replay -> time to first shortage
    4. labels.py            time to shortage -> Low / Medium / High
    5. features.py          the 11 features (noise seed 42)
    6. split.py             train / validation / test by date
    7. train.py             13 fits, selection on validation only
    8. evaluate.py          kept models + 2 baselines, scored once on test

Each step checks the fingerprint of the file the previous step wrote, so the
steps must run in this order. Running the pipeline again reproduces the same
files and the same results.

How to run (from the project folder):
    python run_pipeline.py
"""

import consumption
import evaluate
import features
import labels
import preprocessing
import split
import stock_simulation
import train

STEPS = [
    ("1. Cleaning", preprocessing.main),
    ("2. Ingredient consumption", consumption.main),
    ("3. Hopper stock simulation", stock_simulation.main),
    ("4. Urgency labels", labels.main),
    ("5. Features", features.main),
    ("6. Chronological split", split.main),
    ("7. Training and validation-based selection", train.main),
    ("8. Baselines and test evaluation", evaluate.main),
]

if __name__ == "__main__":
    for title, run_step in STEPS:
        print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
        run_step()
    print("\nPipeline finished.")
