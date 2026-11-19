# Vendo+ ML Component: Decision Log

This log records how the machine learning component was actually built. It is
the source material for the thesis chapters on the dataset, preprocessing,
labels, features, partitioning, models and evaluation.

Every entry has one tag:

| Tag | Meaning |
|---|---|
| **A**: Assumption | Taken as given for the simulation. Not measured and not verified from the data. |
| **D**: Decision | A choice made by the researchers, with its reason. |
| **F**: Verified finding | A fact checked directly against the data or by running the code. |
| **P**: Pending | Open item that must be resolved later. |

Rules for this log:
- Entries are never deleted. If something changes, add a new entry that says
  which entry it replaces (for example, "replaces D-10").
- A finding (F) states where and how it was verified.
- Results are only written here after the experiment that produced them has run.

**Raw data file:** `data/raw/index_1.csv`, 279,369 bytes,
SHA-256 `2a600445a1836da436a3e890e1e1cce0ecf279d369d3a9e280db3ea0d06b594e`.

---

## 1. Assumptions (simulation inputs)

- **A-01 Recipe quantities.** Each beverage uses fixed grams from the coffee,
  milk and chocolate hoppers (methodology Table 2). These values are
  simulation assumptions based on a vending manufacturer's description, not
  measurements of the source machine. *Superseded on 17 November 2026 by
  D-24, which replaces the values with source-based ones where evidence
  exists.*
- **A-02 One row is one drink.** Each transaction in the sales file is assumed
  to be one dispensed beverage, so each row consumes one recipe's worth of
  ingredients.
- **A-03 Hopper capacity.** Each hopper holds 1,000 g.
- **A-04 Urgency horizons.** High = shortage within 4 hours, Medium = after 4
  hours and within 24 hours, Low = no shortage within 24 hours. These horizons
  are design assumptions, not measured operator response times.
- **A-05 No refill during the replay.** When labels are generated, the next 24
  hours of sales are replayed with no refill.
- **A-06 Measurement noise.** Sensor readings have zero-mean Gaussian noise
  with a standard deviation of 2 g (random seed 42).
- **A-07 Local time.** Timestamps are the machine's local time. No time-zone
  conversion is applied.
- **A-08 Labels are simulated.** Urgency labels come from replaying recorded
  sales against assumed stock levels and recipes. They are not historical
  restock records.

## 2. Decisions approved on 17 November 2026 (planning stage)

- **D-01 Start from the raw CSV.** The whole pipeline is rebuilt from the
  original `index_1.csv`, not from an earlier processed file, so every step is
  documented and reproducible.
- **D-02 The raw file is never modified.** Scripts only read it. Its SHA-256
  hash is recorded before and after cleaning, and the run fails if they differ.
- **D-03 `datetime` is the primary timestamp.** `date` is used only to
  cross-check `datetime`, then dropped as redundant.
- **D-04 Cleaned dataset columns.** The cleaned CSV keeps `datetime`,
  `coffee_name`, `cash_type` and `money`. Only `datetime` and `coffee_name`
  feed the ML data generation; `cash_type` and `money` are kept for auditing
  and consistency checks.
- **D-05 `card` is not used.** It is an anonymized customer identifier, not a
  predictive feature. A missing `card` value on a cash sale is normal and is
  never a reason to exclude the sale.
- **D-06 `money` is not a model feature.** The dataset does not state its
  currency, and price says nothing about ingredient use beyond what
  `coffee_name` already says.
- **D-07 Zero-sale days are kept and documented.** The data does not show
  whether the machine was closed, so removing these days would be an
  unsupported assumption.
- **D-08 Stock levels.** Start with 10, 50, 100, 200 and 800 g. After the
  labelled dataset is generated, inspect the class distribution per stock
  level in the **training period only**. If the levels need to change, decide
  using training data only and regenerate the complete dataset. Validation and
  test results are never used to choose stock levels.
- **D-09 Noise in all partitions.** The 2 g noise (A-06) is applied to the stock
  readings in training, validation and test. Labels are always computed from the
  true simulated stock. All models and both baselines receive the same noisy
  readings. *(Clarifies the methodology document, which mentions only "training
  readings".)*
- **D-10 Random Forest class weights.** `class_weight="balanced_subsample"`,
  as stated in the methodology's ML Development section.
- **D-11 Hyperparameter tie-breakers.** Within each algorithm, keep the
  configuration with the highest validation macro F1. Ties are broken by higher
  High-class recall, then higher High-class precision, then the smaller model:
  - Logistic Regression: smaller C.
  - Random Forest: fewer trees, then smaller maximum depth, then larger minimum
    leaf size.
  - Histogram Gradient Boosting: fewer maximum leaf nodes.
- **D-12 Final model selection uses validation only.** Models that meet both
  targets (macro F1 >= 0.80 and High recall >= 0.90) on validation are
  preferred. If none meets both, choose the highest validation macro F1, using
  the D-11 tie-breakers. Test results are never used to select or tune.
- **D-13 Baseline comparison.** Macro F1 is the primary comparison metric, with
  High-class recall reported beside it. The selected models and both baselines
  are evaluated on the test set exactly once. If a baseline performs as well as
  or better than a model, this is reported.
- **D-14 Environment.** The project runs locally on Windows with Python 3.12 or
  3.13 and the exact library versions in `requirements.txt` (pandas 3.0.5,
  NumPy 2.5.3, scikit-learn 1.9.1).
- **D-15 Project structure.** Fixed settings live in `config.py`. Each phase has
  its own script, tests live in `tests/`, and this log lives in `docs/`.

## 3. Phase 1 and 2: inspection and cleaning (implemented, awaiting your run)

### Implementation decisions

- **D-16 Read every cell as text first.** The raw CSV is loaded with all
  columns as text and blank cells kept as blanks. Each column is then converted
  explicitly, so no value is silently changed and every failed conversion can be
  counted.
- **D-17 Timestamp parsing.** `datetime` and `date` are parsed with the
  standard ISO 8601 layout. Text that cannot be read becomes "not a time" and is
  counted, instead of stopping the script.
- **D-18 Beverage name standardization.** Spaces are trimmed, repeated spaces
  are reduced to one, and upper/lower case is ignored. The result is then
  matched to the eight menu names. Any name that still does not match is
  "unknown".
- **D-19 Exclusion rules.** Only problems that make a sale unusable for the
  simulation exclude it. Rules are applied in this order, and each excluded row
  gets the first reason that applies:
  1. `missing_datetime`
  2. `unparseable_datetime`
  3. `date_does_not_match_datetime`: only when both values are readable and
     name different days.
  4. `missing_coffee_name`
  5. `unknown_coffee_name`
  6. `outside_study_period`: outside 1 March 2024 00:00 to 23 March 2025
     23:59:59.999999.
  7. `exact_duplicate`: all six fields identical after standardization. The
     first copy is kept.
- **D-20 Kept on purpose.** The following are reported but never excluded:
  - cash sales without a card;
  - shared timestamps;
  - purchases seconds apart, including repeat purchases with the same card;
  - zero-sale days;
  - an empty or unreadable `date` when `datetime` is valid;
  - unexpected values in `cash_type`, `card` or `money`.
- **D-21 Sort order.** Sales are sorted by `datetime`. Sales with the same
  timestamp keep their file order, so the output is identical on every run.
- **D-22 Output format.** CSVs are written in UTF-8 with `\n` line endings, and
  timestamps include microseconds (`YYYY-MM-DD HH:MM:SS.ffffff`). The files
  should therefore be byte-identical on Windows and Linux (to be confirmed in
  P-01), and no time precision is lost.
- **D-23 Traceability.** Excluded rows keep their original text and their line
  number in the raw file (`source_row`, with the header as line 1).

### Verified findings from the raw data

Verified on 17 November 2026 by running `preprocessing.py` on the file with the
hash above, in Claude's workspace. The same output was produced with Python
3.12/3.13 and with both pandas 2.2.3 and 3.0.5. All 13 tests passed. To be
re-confirmed on your computer (P-01).

- **F-01 Size.** 3,636 records and 6 columns: `date`, `datetime`, `cash_type`,
  `card`, `money`, `coffee_name`.
- **F-02 Timestamps.**
  - Every `datetime` and `date` value parses.
  - The range is 2024-03-01 10:15:50.520 to 2025-03-23 18:11:38.635, covering 388
    calendar days.
  - `date` equals the day of `datetime` in every row.
  - The file is already in time order, and no two rows share a timestamp.
- **F-03 Missing values.** Only `card` has missing values: 89 of them, exactly
  the 89 cash sales. No card payment is missing its card value.
- **F-04 Duplicates.** There are 0 exact duplicate rows.
- **F-05 Beverages.** Exactly the eight menu names, with no spelling, spacing
  or case variants:

  | Beverage | Sales |
  |---|---:|
  | Americano with Milk | 824 |
  | Latte | 782 |
  | Americano | 578 |
  | Cappuccino | 501 |
  | Cortado | 292 |
  | Hot Chocolate | 282 |
  | Cocoa | 243 |
  | Espresso | 134 |
- **F-06 Payment.** 3,547 card and 89 cash sales. All cash sales fall between
  2024-03-02 and 2024-06-03.
- **F-07 Amount paid.**
  - 20 distinct values from 18.12 to 40.00, none zero or negative.
  - Each beverage's card price changed exactly three times and never reverted.
    The first card sale at each new price falls between 20 April and 15 May,
    6 and 18 July, and 28 September and 6 October 2024. Espresso, the
    least-sold drink, is always the last to show the new price.
  - This consistent pattern supports the correctness of `coffee_name`.
- **F-08 Activity.**
  - 9.37 sales per day on average; the busiest day had 26.
  - No sales occur between 23:00 and 06:00.
  - Seven days have zero sales: 2024-05-01, 2024-05-04, 2024-05-05, 2024-11-27,
    2025-01-01, 2025-01-19 and 2025-01-23.
  - Six gaps between sales are longer than 36 hours; the longest is 65.0 hours,
    from 2024-05-03 17:03 to 2024-05-06 10:05.
- **F-09 Close purchases.** 293 consecutive sales are less than 60 seconds
  apart (the shortest is 24.2 s). In 224 cases the same card bought again
  within 60 seconds. These are treated as separate purchases (D-20).
- **F-10 Cleaning result.**
  - 3,636 records retained, 0 excluded, and all 9 checks passed.
  - Cleaned file SHA-256:
    `1354f42c7072069332f1a49c8e344dc66ac8d7976965bc21de383d7b5d69c4bc`.

### Findings from the planning review that affect later phases

These were checked on the raw data on 17 November 2026 and will be
re-verified in code in the phase noted.

- **F-11 Maximum ingredient use (Phase 4).** With the *original* Table 2
  recipes (revised figures are in F-14), the most used in any 24-hour window is:

  | Hopper | Most used in 24 h | Most used in 4 h |
  |---|---:|---:|
  | Coffee | 208 g | 105 to 112 g (all hoppers) |
  | Milk | 202 g | |
  | Chocolate | 200 g | |

  So 800 g can never run short within 24 hours, and 200 g almost never does
  (relevant to D-08).
- **F-12 Seasonal differences between partitions (Phase 6).**

  | Partition | Chocolate drinks (share of sales) | Sales per day |
  |---|---:|---:|
  | Training | 10.8% | 8.9 |
  | Validation | 23.3% | 8.4 |
  | Test | 18.0% | 11.6 |
- **F-13 Expected example count (Phase 4).**
  - 137,520 = 382 decision days x 24 hours x 3 hoppers x 5 stock levels.
  - Every 24-hour window in the proposed decision grid lies inside the data,
    so no incomplete-window exclusions are expected.

## 4. Recipe revision (17 November 2026)

The full evidence is in the doc "Vendo+ Recipe Evidence Review". This section
records what was decided.

- **D-24 Revised recipe table (version R2).** Replaces A-01. The only code
  copy is `config.RECIPES_G` in `config.py`.

  **Rule applied (set by the researchers):**
  - Replace a value when credible, relevant sources document a different
    quantity.
  - Keep a value when the evidence is insufficient, and label it
    researcher-defined.
  - Never invent a value or a citation.

  **Ingredient forms:**
  - Coffee = ground coffee, ground from whole beans in the coffee hopper.
  - Milk = vending milk powder.
  - Chocolate = vending cocoa or chocolate drink powder.
  - All values assume a cup of about 150 ml.

  **Revised table** (grams per sale):

  | Beverage | Original (coffee / milk / chocolate) | Revised (coffee / milk / chocolate) | Basis per value | Confidence |
  |---|---|---|---|---|
  | Espresso | 8 / 0 / 0 | **7** / 0 / 0 | Coffee: literature-derived [1][2][4] | Coffee: High |
  | Americano | 8 / 0 / 0 | **7** / 0 / 0 | Coffee: literature-derived [1][2][7] | Coffee: Medium |
  | Americano with Milk | 8 / 6 / 0 | **7** / 6 / 0 | Coffee: literature-derived. Milk: literature-derived, top of 4-6 g [8] | Coffee: Medium. Milk: Medium |
  | Cortado | 8 / 6 / 0 | **7** / 6 / 0 | Coffee: literature-derived. Milk: **researcher-defined** (no cortado source) | Coffee: Medium. Milk: Low |
  | Cappuccino | 8 / 10 / 0 | **7** / 10 / 0 | Coffee: literature-derived [2]. Milk: literature-derived, top of 8-10 g [8] | Coffee: High. Milk: High |
  | Latte | 8 / 12 / 0 | **7** / 12 / 0 | Coffee: literature-derived. Milk: literature-derived, top of 10-12 g [8] | Coffee: Medium. Milk: High |
  | Cocoa | 0 / 6 / 15 | 0 / 6 / **20** | Chocolate: literature-derived, 20-22 g per 150 ml [9][10][11]. Milk: **researcher-defined** | Chocolate: Medium. Milk: Low |
  | Hot Chocolate | 0 / 8 / 20 | 0 / 8 / 20 | Chocolate: literature-derived [9][10][11]. Milk: **researcher-defined** | Chocolate: Medium. Milk: Low |

  Zeros are by definition of the drink; for example, an espresso contains no
  milk.

  **Why each value changed or stayed:**
  - **Coffee 8 g to 7 g (all six coffee drinks).** A vending manufacturer's
    factory default is 7.0 g [1]. The Italian espresso standard is
    7 ± 0.5 g [2], and a peer-reviewed study preferred 6.5-7.5 g [4].
    - 8 g lies inside the Necta (5.5-8.5 g) [5] and TESDA (7-9 g) [3] ranges,
      but above every stated default.
    - Milk drinks use the same single dose, because INEI's cappuccino is built
      on one 7 g espresso [2].
  - **Cocoa chocolate 15 g to 20 g.** 15 g is below every documented vending
    chocolate dose. Those doses are 20-22 g per 150 ml [9], and about 21 g per
    150 ml when scaled from 25 g per 175-177 ml [10][11].
  - **Retained as literature-supported:**
    - the milk in Americano with Milk (6 g), Cappuccino (10 g) and Latte (12 g),
      each the top of the manufacturer's range [8];
    - the chocolate in Hot Chocolate (20 g).
  - **Retained as researcher-defined (insufficient evidence to change):**
    - Cortado milk (6 g): no source documents a cortado dose.
    - Cocoa milk (6 g) and Hot Chocolate milk (8 g): no source documents milk
      powder added to a vending chocolate drink. Necta's Kikko Max menu does
      list a "Chocomilk" drink [6], so such drinks exist, but no dose is given.
  - **Not used:**
    - Liquid-milk recipes (INEI's 100 ml of milk [2]). Converting to powder
      would need the powder's composition, which varies by product.
    - Animo's derived grams, which are not stated by the manufacturer.

  **Citation change:** the N&W Krea manual (2014), cited for the original
  Table 2, shows that doses are adjustable (pp. 20-21, 33). It gives no grams
  for any drink, so it supports the machine design but not the recipe values.

  **Supporting references** (full APA list in the evidence review):
  1. Saeco Vending, *Phedra* manual, sec. 7.6, p. 25 (default 7.0 g; settings
     6-9 g). https://www.segafredo.com.au/wp-content/uploads/2015/05/PHEDRA-TECHNICAL-SHEET-USER-MANUAL.pdf
  2. Istituto Nazionale Espresso Italiano (2006), *The certified Italian
     espresso and cappuccino*:
     - p. 7: espresso of 7 ± 0.5 g;
     - pp. 8-9: cappuccino of 25 ml espresso + 100 ml milk.

     https://www.nespressoguide.com/files/File/istituzionale_inei_hq_en.pdf
  3. TESDA (2013), *Training regulations: Barista NC II*, pp. 31, 37 (7-9 g
     single shot). https://tesda.gov.ph/Downloadables/TR%20Barista%20NC%20II.pdf
  4. Andueza et al. (2007), *J. Sci. Food Agric.* 87(4), 586-592 (6.5-7.5 g per
     40 ml preferred). https://dadun.unav.edu/handle/10171/35076
  5. Necta (2003), *Colibrì* handbook, p. 12 (coffee doses 5.5-8.5 g).
     https://www.manualslib.com/manual/1597619/Necta-Colibri.html
  6. Evoca (2010), *Kikko Max* product sheet (canisters; "Chocomilk"
     selection). https://newebcdn-necta.evocagroup.com/sites/necta/files/2019-07/KikkoMax_UK.pdf
  7. Merriam-Webster, "Americano" (espresso diluted with hot water).
     https://www.merriam-webster.com/dictionary/americano
  8. Barry Callebaut, Caprimo Semi-Skimmed Milk Powder (per 150 ml: coffee
     4-6 g, cappuccino 8-10 g, latte 10-12 g).
     https://www.barry-callebaut.com/en/vending/products/Caprimo-Semi-Skimmed-Milk-Powder/VM-21002-V39
  9. Barry Callebaut, Van Houten Organic Fairtrade (150 ml: 20-22 g).
     https://www.barry-callebaut.com/en-JO/vending/products/Van-Houten%E2%80%93Organic-Fairtrade-Chocolate/VM-75978-V17
  10. Nestlé Professional, Hot Chocolate Complete Mix Vending (25 g per
      175 ml). https://www.nestleprofessional.co.nz/node/473
  11. Conagra, Swiss Miss Vending Mix Cocoa (25 g per 6 oz cup).
      https://smartlabel.labelinsight.com/product/6358905/other/instructions

- **A-09 Remaining recipe assumptions** (still researcher-defined after D-24):
  - The Kaggle machine's model, recipe settings, cup size and ingredient
    brands are unknown. The table is a simulation input, not a measurement of
    that machine.
  - Cup size is about 150 ml.
  - Cortado milk is 6 g; Cocoa milk 6 g; Hot Chocolate milk 8 g.
  - Cocoa and Hot Chocolate use the same chocolate dose (no documented
    difference).
  - Doses are fixed, nominal values. Real machines vary (INEI allows ±0.5 g),
    and the same drink can take 7.5 g or 15 g of milk powder depending on the
    product.
  - The bean hopper loses the same mass that is ground for each drink.

- **F-14 Effect of D-24 on consumption** (verified by `consumption.py` on the
  cleaned data, 17 November 2026).

  | Hopper | Total, original to revised | Most in 24 h | Most in 4 h | Days a full 1,000 g lasts |
  |---|---|---:|---:|---:|
  | Coffee | 24,888 g to 21,777 g (-12.5%) | 182 g | 98 g | 17.8 |
  | Milk | 24,804 g to 24,804 g (unchanged) | 202 g | 112 g | 15.6 |
  | Chocolate | 9,285 g to 10,500 g (+13.1%) | 220 g | 120 g | 37.0 |

  Preview of how often each stock level runs short. This replays the next
  24 hours after every hourly decision time from 3 March 2024 to 21 March 2025
  (9,216 hours). The real labels come in Phase 4.

  | Stock level | Short within 24 h | Short within 4 h (High) |
  |---|---|---|
  | 800 g | 0% | 0% |
  | 200 g | 0 hours (coffee); 1 hour each (milk, chocolate) | 0% |
  | 100 g | 3.0-16.1% | 0-3 hours |
  | 50 g | 20.1-60.6% | 1.1-2.5% |
  | 10 g | 58.1-95.6% | 15.1-37.5% |

  The High class will come almost entirely from the 10 g level, which is
  relevant to P-03.

## 5. Pipeline step 2: ingredient consumption (implemented, awaiting your run)

### Implementation decisions

- **D-25 One recipe definition.** `consumption.py` reads only
  `config.RECIPES_G`. `config.MENU_ITEMS` is now built from the recipe table,
  so the menu and the recipes cannot disagree.
  - `tests/test_consumption.py` holds a deliberate copy of the approved table.
    It fails if `config.py` is edited without updating this log.
- **D-26 Unknown beverages stop the run.** A beverage with no recipe raises an
  error that lists it. It is never given zero consumption.
- **D-27 Input check.** The cleaned CSV's SHA-256 must match the one in
  `cleaning_report.json`, and must be unchanged after the run.
- **D-28 Hourly series.**
  - Each hour H holds the sales from H:00:00 up to, but not including,
    (H+1):00:00.
  - The series covers every hour of the study period: 2024-03-01 00:00 to
    2025-03-23 23:00, 9,312 hours.
  - Hours with no sales are kept with zero transactions and zero grams. They
    are not treated as downtime (D-07).
- **D-29 Output format.**
  - Grams are whole numbers.
  - Every sale gets a `transaction_id` (1, 2, 3, ...) in time order.
  - Files are written in UTF-8 with `\n` line endings.
- No stock levels, noise or labels are produced in this step.

### Verified findings (Claude's workspace, 17 November 2026)

- **F-15 Consumption result.**
  - 3,636 transaction rows (one per cleaned sale) and 9,312 hourly rows.
  - 2,084 hours have at least one sale and 7,228 have none.
  - The busiest hour had 8 sales.
  - The most used in one hour was 49 g of coffee, 58 g of milk and 80 g of
    chocolate.
  - All 13 checks and all 28 tests passed. The output was identical with
    Python 3.12/3.13 and pandas 2.2.3/3.0.5.
  - SHA-256 hashes:
    - `transaction_consumption.csv`:
      `f6507e577de820e6037b8bd04bf888a08bcd8d300d4c149856b9bd8e21ba6f30`
    - `hourly_consumption.csv`:
      `755c319bc107e50b579e306cace56de81cd2154f8a208243dd9646728e58d32c`

## 6. Pipeline steps 3-8: simulation, labels, features, split, models, evaluation

Implemented and run on 17 November 2026, at your request to complete all the
remaining phases in one go. Decisions that would normally have waited for your
review were taken with the rules already approved, and are recorded here.

### Implementation decisions

- **D-30 Time windows (resolves P-04).**
  - Labels replay sales in (t, t + 24 h]. A sale exactly at t is in the past.
  - High: shortage at or before t + 4 h.
  - Medium: shortage after t + 4 h and at or before t + 24 h.
  - Low: no shortage within 24 h.
  - The shortage is the first sale that needs more than what is left. A stock
    equal to the need is enough.
- **D-31 Complete windows.** A decision time is used only if both of these lie
  inside the study period:
  - the 24 hours before it (for the features);
  - the 24 hours after it (for the label).
- **D-32 Features (resolves P-05).**
  - Consumption rates use exact sale times, counting sales in (t - w, t]. The
    hourly table is not used.
  - Noise is drawn from `numpy.random.default_rng(42)`, one draw per row, in a
    fixed row order (decision time, hopper, stock level).
  - Noise is not clipped.
  - Models and baselines see the same measured stock. Labels use the true stock.
- **D-33 Stock-level review (resolves P-03).** Done with training data only
  (D-08). The five levels are kept.
  - Every class has thousands of training examples (High: 5,092).
  - 95.2% of training High labels come from the 10 g level.
  - The 200 g and 800 g levels are almost entirely Low. They are kept, but
    their effect is reported per stock level.
- **D-34 No refit before testing.** The kept models are trained on the
  training partition only and are tested exactly as selected. They are not
  retrained on training plus validation, matching the methodology's "kept
  models are evaluated once on the test period".
- **D-35 Choosing among the three kept models.** D-12 is applied with
  validation macro F1, then High recall, then High precision. "Smaller model"
  is not used across algorithms, because sizes are not comparable between
  algorithm types.
- **D-36 Baselines.**
  - The stock-only baseline compares the measured stock in grams
    (≤ 50 g = High, ≤ 200 g = Medium).
  - The consumption baseline divides the measured stock by the average use over
    the last 24 h, in g per hour.
  - Hours left ≤ 4 gives High, ≤ 24 gives Medium, otherwise Low. No use in the
    last 24 h gives Low.
- **D-37 One test evaluation.** The order of work was:
  1. The model selection was frozen in `results/selection.json`, SHA-256
     `fe447b07ed4db62c00c9a6097b179cbae89729c6b235a28038dd8d109281014d`.
  2. The evaluation code was dry-run on the validation partition only.
  3. The test partition was then scored once, on 17 November 2026.
  4. No setting was changed afterwards.

  Re-running `run_pipeline.py` reproduces the same numbers. That is a
  reproduction, not a second evaluation.
- **D-38 Shared helpers.** `pipeline_utils.py` holds the fingerprint, CSV and
  JSON helpers used by steps 3-8. Each step checks the SHA-256 of the file the
  previous step wrote before using it.

### Verified findings

- **F-16 Scenarios and labels.**
  - 9,216 decision times (3 March 2024 00:00 to 21 March 2025 23:00), with
    0 incomplete windows.
  - 138,240 scenarios: Low 100,702, Medium 29,008, High 8,530.
  - 94% of High labels are at 10 g, and 800 g is always Low.
  - The fast replay matched a slow, step-by-step replay on 500 random
    scenarios.
- **F-17 Features.**
  - The observed noise had a mean of -0.0065 g and a standard deviation of
    2.0076 g.
  - The rates matched an independent recomputation on 200 random scenarios.
  - A unit test confirms that future sales do not change any feature.
- **F-18 Partitions.**

  | Partition | Rows | Low | Medium | High |
  |---|---:|---:|---:|---:|
  | Training | 87,120 | 64,815 | 17,213 | 5,092 |
  | Validation | 21,600 | 15,608 | 4,637 | 1,355 |
  | Test | 28,800 | 19,730 | 7,037 | 2,033 |

  The 720 rows from 31 October and 31 December were excluded.
- **F-19 Validation results** (all 13 configurations converged). The best
  configuration of each algorithm:

  | Algorithm | Kept configuration | Validation macro F1 | Validation High recall |
  |---|---|---:|---:|
  | Logistic Regression | C = 1 | 0.6970 | 0.8133 |
  | Random Forest | 200 trees, depth 16, leaf 20 | 0.6994 | 0.8480 |
  | Hist. Gradient Boosting | 31 leaf nodes | 0.6953 | 0.8399 |

  - All 13 configurations scored between 0.690 and 0.699 macro F1.
  - No configuration met either target.
  - The final model is Random Forest: no model met both targets, and it had the
    highest validation macro F1.
  - Baselines on validation (context only): stock-only macro F1 0.363;
    consumption 0.569.
- **F-20 Test results** (single evaluation):

  | Method | Accuracy | Macro F1 | High precision | High recall |
  |---|---:|---:|---:|---:|
  | Logistic Regression | 0.806 | 0.724 | 0.518 | 0.830 |
  | **Random Forest (final)** | 0.806 | 0.714 | 0.490 | 0.831 |
  | Hist. Gradient Boosting | 0.805 | 0.716 | 0.495 | 0.831 |
  | Stock-only baseline | 0.451 | 0.410 | 0.224 | 0.952 |
  | Consumption baseline | 0.761 | 0.617 | 0.398 | 0.527 |

  - Neither target is met by the final model (0.714 < 0.80; 0.831 < 0.90) or
    by any other model.
  - All three models beat both baselines on macro F1, so they count as useful
    under D-13.
  - The stock-only baseline has a higher High recall (0.952). It reaches it by
    calling most low-stock rows High (High precision 0.224, macro F1 0.410).
  - Logistic Regression had the highest test macro F1 (0.724). The final model
    stays Random Forest, because selection was made on validation (D-12, D-37).
- **F-21 Results by stock level (test, Random Forest).**

  | Stock level | Accuracy | High recall |
  |---|---:|---:|
  | 10 g | 0.609 | 0.915 |
  | 50 g | 0.622 | 0.011 (187 High cases) |
  | 100 g | 0.798 | 0.000 (2 High cases) |
  | 200 g and 800 g | 1.000 | no High cases (all Low) |

  So the overall accuracy is inflated by the trivial 200 g and 800 g levels.
- **F-22 Reproducibility.**
  - The full pipeline runs in about 75 seconds.
  - A clean run on Python 3.12 produced byte-identical data files, models and
    test results. Only timestamps and fit times differed.
  - 53 tests passed.
  - `test_results.csv` SHA-256:
    `c4d85688a2414af93e78bc05018a8898fd57a0ecbbd66ff154e4fff76cc2dc80`.

## 7. Changes from the original methodology document

For the thesis text. Items 1-5 clarify or extend the methodology. Item 6
revises its recipe values.

1. **Noise (D-09).** Applied to all partitions, not only "training readings".
2. **Tie-breakers (D-11).** The full chain is applied, and "smaller model" is
   now defined.
3. **Final selection (D-12).** Adds the rule for when no model meets both
   targets.
4. **Baseline comparison (D-13).** Names macro F1 as the metric for
   "outperforms".
5. **Stock levels (D-08).** May be revised after Phase 4, using training data
   only.
6. **Recipe table (D-24).**
   - Coffee is 7 g, not 8 g, and Cocoa chocolate is 20 g, not 15 g.
   - Cortado milk and the milk in both chocolate drinks are labelled
     researcher-defined.
   - Table 2 should cite the sources in D-24, not only the Krea manual.
7. **Evaluation details (D-30 to D-37).** The time-window conventions, the
   no-refit rule and the baseline formulas are now stated explicitly.
8. **Results vs targets.** The proposed targets (macro F1 ≥ 0.80, High
   recall ≥ 0.90) were not met. The thesis should report this together with
   the fact that every model beat both baselines.

## 8. Pending items

*Resolved on 17 November 2026:*
- P-03, by D-33.
- P-04, by D-30.
- P-05, by D-32.


- **P-01** Run `preprocessing.py` on your computer and record whether the raw
  and cleaned SHA-256 hashes match F-10 and the header of this log.
- **P-02** Fill in `RAW_DATA_DOWNLOAD_DATE` in `config.py` with the date
  `index_1.csv` was downloaded from Kaggle.
- **P-03 (resolved, D-33)** Review the stock levels after Phase 4 (D-08).
- **P-04 (resolved, D-30)** Confirm the time-window conventions before Phase 4. Proposed:
  - Features use sales in (t - window, t].
  - Labels use sales in (t, t + 24 h].
  - High means a shortage at or before t + 4 h.
  - Medium means a shortage after t + 4 h and at or before t + 24 h.
- **P-05 (resolved, D-32)** Note for Phases 4-5: hourly buckets include the start of the hour
  and exclude its end, [H, H+1), while the proposed feature and label windows
  exclude their start and include their end, (t - window, t] and (t, t + 24 h].
  - The two agree for this data, because no sale falls exactly on the hour
    (F-02).
  - Later phases should still compute windows from transaction times, or
    state which convention they use.
- **P-06** Update thesis Table 2 and the recipe paragraph using D-24. A draft
  paragraph is in the evidence review doc.
- **P-07** Run `consumption.py` on your computer and confirm the two output
  hashes in F-15.
- **P-08** Run `python run_pipeline.py` on your computer and confirm that
  `results/test_results.csv` has the SHA-256 in F-22.
- **P-09** Update the thesis methodology and results chapters using the
  thesis draft doc and F-16 to F-22.
