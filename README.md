# Fraud-Detection-MLOps

Real-time banking fraud detection pipeline: synthetic transactions are validated,
enriched with behavioural features, scored by a trained model and stored in SQLite,
with a Streamlit dashboard for live monitoring.

## Project structure

```
├── src/                     # core pipeline package
│   ├── config.py            # project paths (data, models, database)
│   ├── database.py          # SQLite schema & DB_PATH
│   ├── transaction_generator.py
│   ├── validator.py
│   ├── feature_engineering.py
│   ├── fraud_predictor.py
│   └── ingestion.py         # live ingestion service
├── scripts/                 # one-off jobs
│   ├── generate_transaction.py   # build training CSV
│   ├── train_model.py
│   └── view_transaction.py
├── app/
│   └── dashboard.py         # Streamlit monitoring dashboard
├── tests/                   # demo / check scripts
├── data/                    # transactions.csv, database.db
└── models/                  # trained model artifacts
```

## Usage

Run every command from the project root.

```bash
pip install -r requirements.txt

python -m scripts.generate_transaction   # generate training data
python -m scripts.train_model            # train & save the model
python -m src.ingestion                  # start the live pipeline
streamlit run app/dashboard.py           # open the dashboard

python -m tests.test_validator
python -m tests.test_features
```
