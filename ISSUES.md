# Known Issues

Project review findings, recorded 2026-11-24. Items marked *verified* were confirmed by running code or querying `data/database.db`.

---

## 1. Dashboard not updating in real time

> **Status: fixed in `app/dashboard.py`.**
> - The dashboard warns when ingestion has stopped (1.1/1.2).
> - All times are shown in local time (1.3).
> - Transactions without a prediction are counted and reported (1.4).
> - The window start is floored to 10s so the cache works (1.5).
>
> The dashboard still needs `python -m src.ingestion` running to get new data.

### 1.1 Ingestion service not running — *verified*
The dashboard auto-refresh works (`st.fragment(run_every=...)`), but it only re-reads the database. When the review ran, the newest row in `fraud_predictions` was ~5 hours old, so nothing was writing new data.

**Fix:** keep both processes running in separate terminals:
```bash
python -m src.ingestion
streamlit run app/dashboard.py
```

### 1.2 Default "Last hour" window looks empty
When ingestion is stopped, the default window has no rows, so the dashboard looks frozen even though it refreshes.

### 1.3 UTC vs local time mismatch
- Tables show `predicted_at` in UTC.
- "Last updated" uses local time via `datetime.now()` ([app/dashboard.py:386](app/dashboard.py#L386)).

As a result, live data looks 5.5 hours old in IST.

**Fix:** convert to one timezone, e.g. `df["predicted_at"].dt.tz_convert("Asia/Kolkata")`, and use the same zone for "Last updated".

### 1.4 Most transactions are hidden — *verified*
`transactions` has 14,603 rows but `fraud_predictions` has only 195. The dashboard uses an INNER JOIN on predictions ([app/dashboard.py:84](app/dashboard.py#L84)), so rows ingested before the model existed never appear.

**Fix:** use a LEFT JOIN, or backfill predictions for old rows, or clear out the legacy data.

### 1.5 Cache never hits
`since_iso` changes every microsecond, so `@st.cache_data(ttl=2)` never reuses a result.

**Fix:** round `since` down to the refresh interval, or drop the cache.

---

## 2. Validator is broken (critical) — *verified*

File: [src/validator.py](src/validator.py)

> **Status: fixed.** All fields are now checked, and amounts must be finite and greater than 0.

### 2.1 Indentation bug
The type checks, amount check, timestamp check and the `return` are indented inside the `for field in REQUIRED_FIELDS` loop. The function returns after checking only the first field, `transaction_id`.
- A transaction with `customer_id` deleted returns `(True, [])`.
- Ingestion then crashes with a `KeyError` when it reads the missing field.

### 2.2 Amount condition logic
Line 34 has:
```python
or not math.isfinite(amount) and amount < 0
```
`and` should be `or`, and the check should probably be `amount <= 0`. Right now `amount = -100` is accepted.

### 2.3 Consequence
`quarantine_transactions` has 0 rows. The data-quality layer never rejects anything.

---

## 3. Training data does not match live data (critical) — *verified*

> **Status: fixed.**
> - A new `TransactionSimulator` covers 200 customer profiles and injects labelled fraud episodes (amount spikes, new cities, bursts).
> - `scripts/generate_training_data.py` replays that simulator through the same `calculate_features` code.
> - The model is split by time, and the threshold is tuned for F1 on the validation set.
> - Ground-truth labels are stored in `transaction_labels`, and the dashboard shows live precision and recall.
> - There is now a single model artifact.
>
> Model v2.0 test set: ROC-AUC 0.97, precision 0.94, recall 0.67, 3.6% flagged against a 5% true fraud rate.

| Feature | Training data | Live data (from DB) |
|---|---|---|
| `txn_count_5m` | Poisson(5) | mean 22, max 42 |
| `amount_sum_1h` | 0 – 120,000 | mean ~6.1 lakh, max ~14.5 lakh |

- The live generator uses only 4 customers with one transaction every 2s, so each customer transacts about every 8s.
- Live features are far outside the training range, and the model is extrapolating.
- **51% of transactions get flagged as suspicious**, so the scores carry little real information.

**Fix:**
- Make the generator realistic: more customers, random gaps between transactions, occasional injected fraud patterns.
- Build training data with the same `calculate_features` code the pipeline uses.

### Related model issues
- `amount_ratio` has no fraud signal tied to how live transactions are generated.
- Live transactions have no fraud labels, so real-world accuracy can't be measured.
- The 0.65 threshold is hard-coded rather than tuned on validation data.
- `scripts/train_model.py` saves the same artifact twice (`.pkl` and `.joblib`).

---

## 4. Database and performance

> **Status: fixed.**
> - Indexes added, and timestamps normalised to a fixed UTC format, so feature queries compare strings and use the index.
> - WAL mode enabled.
> - A schema mismatch now raises `SchemaMismatchError` instead of dropping the table.
> - Transaction IDs are full 128-bit UUIDs.

- **No indexes.** `transactions(customer_id, timestamp)` and `fraud_predictions(predicted_at)` are not indexed.
  - Each ingested transaction runs 3 full scans with `julianday()` on every row.
  - Each dashboard refresh scans all predictions.
  - Both get slower as the DB grows.
- **Journal mode is `delete`.** Switch to `PRAGMA journal_mode=WAL` so the dashboard can read while ingestion writes, without "database is locked" errors.
- **Silent table drop.** `create_database()` drops `transaction_features` whenever its columns change ([src/database.py:44](src/database.py#L44)). That loses data with no migration.
- **Short IDs.** Transaction IDs are `uuid4()[:8]`, only 32 bits, so duplicates become likely at scale.

---

## 5. Tests

> **Status: fixed.**
> - 47 pytest tests across the validator, features, ingestion, schema and simulator.
> - Every test uses a temporary database.
> - Run them with `python -m pytest`.

- Files in `tests/` are demo scripts. They have no asserts and pytest does not collect them.
- They **write into the real `data/database.db`**. That is where the `DEMO_*` customers with back-dated timestamps come from.
- `test_validator` case 3 sets `customer_id = "None"` (a string), so it does not test a missing field.
- Real tests with asserts would have caught the bugs in section 2.

**Fix:** rewrite as pytest tests with asserts. Point `DB_PATH` at a temporary database, e.g. using `tmp_path` and `monkeypatch`.

---

## 6. Minor

> **Status: fixed.**
> - Requirements are pinned to the `mlenv` versions.
> - The simulator's `__main__` prints sample transactions.
> - Merchant and city names are consistent.

- `requirements.txt` has no pinned versions. `st.fragment(run_every=...)` and `width="stretch"` need a recent Streamlit.
- `transaction_generator.py` `__main__` generates a transaction, sleeps, and never prints it.
- Naming is inconsistent: `"amazon"` vs `"Flipkart"`, and `"Bangalore"` vs `"Bengaluru"` in the tests.

---

## Suggested fix order

1. Start ingestion alongside the dashboard (section 1.1).
2. Fix the timezone display (section 1.3).
3. Fix the validator and rewrite the tests as real pytest tests (sections 2 and 5).
4. Align training data with live features, then retrain (section 3).
5. Add indexes and enable WAL mode (section 4).
