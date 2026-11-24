# Fraud-Detection-MLOps

Real-time banking fraud detection pipeline: synthetic transactions are validated,
enriched with behavioural features, scored by a trained model and stored in SQLite,
with a Streamlit dashboard for live monitoring.

## How it works

1. **Simulator** (`src/transaction_generator.py`) produces a stream of transactions
   from 200 customers, each with a home city, usual device and typical spend.
   About 5% of transactions belong to labelled fraud episodes: amount spikes,
   new cities and rapid bursts.
2. **Validation** (`src/validator.py`) rejects malformed transactions into
   `quarantine_transactions`.
3. **Feature engineering** (`src/feature_engineering.py`) computes
   `txn_count_5m`, `amount_sum_1h`, `amount_ratio` and `city_changed`
   from the customer's history.
4. **Prediction** (`src/fraud_predictor.py`) scores the transaction and
   applies the tuned threshold.
5. **Dashboard** (`app/dashboard.py`) shows live activity, alerts, data quality,
   and live precision / recall against the simulator's ground-truth labels.

Training data is produced by replaying the same simulator through the same
feature code, so training and serving features match.

## Project structure

```
├── src/                     # core pipeline package
│   ├── config.py            # project paths (data, models, database)
│   ├── database.py          # SQLite schema, indexes and insert helpers
│   ├── transaction_generator.py
│   ├── validator.py
│   ├── feature_engineering.py
│   ├── fraud_predictor.py
│   └── ingestion.py         # live ingestion service
├── scripts/                 # one-off jobs
│   ├── generate_training_data.py
│   ├── train_model.py
│   └── view_transaction.py
├── app/
│   └── dashboard.py         # Streamlit monitoring dashboard
├── tests/                   # pytest suite (uses temporary databases)
├── data/                    # training_data.csv, database.db (local only)
└── models/                  # trained model artifact
```

## Usage

Run every command from the project root.

```bash
pip install -r requirements.txt

python -m scripts.generate_training_data   # build training data
python -m scripts.train_model              # train model + tune threshold

# In two separate terminals:
python -m src.ingestion                    # start the live pipeline (run ONE instance)
streamlit run app/dashboard.py             # open the dashboard

python -m pytest                           # run tests
```

The dashboard only shows new data while `src.ingestion` is running. It
shows a warning when no prediction has arrived for 30 seconds.
