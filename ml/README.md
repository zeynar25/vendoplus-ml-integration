# Vendo+ ML pipeline

Predicts the restock urgency (Low / Medium / High) of a coffee vending
machine's three ingredient hoppers, using the Kaggle Coffee Sales dataset
(`index_1.csv`). Every step is a separate, tested script.

| Step | Script | Output (data/processed/ or results/) |
|---|---|---|
| 1 Cleaning | `preprocessing.py` | `coffee_sales_cleaned.csv` |
| 2 Ingredient consumption | `consumption.py` | `transaction_consumption.csv`, `hourly_consumption.csv` |
| 3 Hopper stock simulation | `stock_simulation.py` | `stock_scenarios.csv` |
| 4 Urgency labels | `labels.py` | `labeled_scenarios.csv` |
| 5 Features | `features.py` | `features.csv` |
| 6 Chronological split | `split.py` | `ml_dataset.csv` |
| 7 Training + selection (validation only) | `train.py` | `validation_results.csv`, `selection.json`, `models/` |
| 8 Baselines + single test evaluation | `evaluate.py` | `test_results.csv`, figures, `evaluation_report.json` |

Other files:

- `config.py`: all fixed settings, including the one recipe table.
- `pipeline_utils.py`: shared helpers.
- `docs/decision_log.md`: every assumption, decision and verified finding.

## Run on Windows

```bat
cd C:\path\to\vendoplus-ml-integration\ml
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
copy "C:\path\to\index_1.csv" data\raw\index_1.csv
python run_pipeline.py
python -m pytest -v
```

The full pipeline takes about 1-2 minutes. The raw CSV is never modified.
