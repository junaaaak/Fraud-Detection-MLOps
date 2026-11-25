# Fraud-Detection-MLOps

Real-time banking fraud detection pipeline: synthetic transactions are validated,
enriched with behavioural features, scored by a trained model and stored in SQLite,
with a Streamlit dashboard for live monitoring.

## How it works

1. **Simulator** (`src/transaction_generator.py`) produces a stream of transactions
   from 200 customers, each with a home city, usual device and typical spend.
   2% of events start a labelled fraud episode (amount spike, new city and/or
   rapid burst); with bursts, about 5% of transactions are fraud.
   Customer profiles are fixed (seeded), and on startup ingestion seeds ~20
   past transactions per customer so behavioural features work from the
   first live event. About 4% of live events are deliberately malformed or
   duplicate records, to exercise validation and quarantine.
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

## Model and formulas

### Simulated data

Each customer $c$ gets a fixed profile: a home city, a usual device and a
typical spend

$$
\text{typical}_c = \operatorname{clip}\big(\text{LogNormal}(\ln 2000,\ 0.7),\ 200,\ 20000\big)
$$

| Event | Amount | City | Device |
|---|---|---|---|
| Normal | $\text{typical}_c \times \text{LogNormal}(0,\ 0.5)$ | home city (95%), else random | usual device (90%), else random |
| Fraud, spike (p = 0.7) | $\text{typical}_c \times U(4,\ 15)$ | | random |
| Fraud, no spike | $\text{typical}_c \times U(0.8,\ 2)$ | | random |
| Fraud, new city (p = 0.6) | | a city other than home | |
| Fraud, burst (p = 0.5) | | | 2–5 more fraud transactions follow back-to-back |

Each event is fraud with probability `fraud_rate = 0.02`; every fraud episode
has at least one of spike, new city or burst. Live events are 2 s apart on
average.

### Features

For a transaction by customer $c$ at time $t$ with amount $a$, using only
the customer's **earlier** transactions $H_c(t) = \{\,j : c_j = c,\ t_j < t\,\}$:

$$
\begin{aligned}
\texttt{txn\_count\_5m} &= \big|\{\, j \in H_c(t) : t_j \ge t - 5\,\text{min} \,\}\big| \\
\texttt{amount\_sum\_1h} &= \sum_{j \in H_c(t),\ t_j \ge t - 1\,\text{h}} a_j \\
\texttt{avg\_prior\_amount} &= \frac{1}{|H_c(t)|} \sum_{j \in H_c(t)} a_j \\
\texttt{amount\_ratio} &= \frac{a}{\texttt{avg\_prior\_amount}} \\
\texttt{city\_changed} &= \mathbb{1}\big[\,\text{city} \ne \text{city of the latest } j \in H_c(t)\,\big]
\end{aligned}
$$

With no history, `amount_ratio` is missing (filled with the training median
by the imputer) and `city_changed` is 0. That is why ingestion seeds ~20
past transactions per customer before going live.

The model's inputs are `amount`, `txn_count_5m`, `amount_sum_1h`,
`amount_ratio` and `city_changed`.

### Model

A scikit-learn pipeline (`scripts/train_model.py`):

1. `SimpleImputer(strategy="median")`, which fills missing `amount_ratio`.
2. `RandomForestClassifier` with 200 trees, `max_depth=10`,
   `min_samples_leaf=5` and `class_weight="balanced"`. Balanced weights
   up-weight the rare fraud class:

$$
w_k = \frac{n}{2\, n_k} \qquad (n_k = \text{number of training rows in class } k)
$$

The fraud score is the average of the trees' fraud probabilities, and a
transaction is flagged when it reaches the decision threshold $\tau$:

$$
p(x) = \frac{1}{200} \sum_{m=1}^{200} p_m(\text{fraud} \mid x),
\qquad
\texttt{is\_suspicious} = \mathbb{1}\big[\,p(x) \ge \tau\,\big]
$$

### Training and threshold

The data (30,000 replayed transactions) is split **by time**, never shuffled,
because in production the model always scores future transactions:
60% train, then 20% validation, then 20% test.

$\tau$ is the threshold that maximises F1 on the validation set:

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad
\text{Recall} = \frac{TP}{TP + FN}, \qquad
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

$$
\tau = \arg\max_{\tau}\ F_1(\tau) \quad \text{on validation}
$$

The final metrics are measured once on the untouched test set. The
dashboard computes live precision and recall with the same formulas,
against the simulator's ground-truth labels.

### Current model (v2.1)

| Metric (test set) | Value |
|---|---|
| Threshold $\tau$ | 0.763 |
| Precision (fraud) | 0.856 |
| Recall (fraud) | 0.796 |
| F1 (fraud) | 0.825 |
| ROC AUC | 0.973 |
| PR AUC (average precision) | 0.881 |
| Fraud rate / flag rate | 5.5% / 5.1% |

ROC AUC is the chance that a random fraud scores higher than a random normal
transaction. PR AUC summarises precision against recall, and is the more
informative of the two when fraud is rare.

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
│   ├── export_parquet.py    # snapshot features to data/snapshots/*.parquet
│   └── view_transaction.py
├── app/
│   └── dashboard.py         # Streamlit monitoring dashboard
├── tests/                   # pytest suite (uses temporary databases)
├── data/                    # training_data.csv, database.db (local only), snapshots/
└── models/                  # trained model artifact
```

## Usage

Run every command from the project root.

```bash
pip install -r requirements.txt

python -m scripts.generate_training_data   # build training data
python -m scripts.train_model              # train model + tune threshold
python -m scripts.export_parquet           # export feature snapshot to Parquet

# In two separate terminals:
python -m src.ingestion                    # start the live pipeline (run ONE instance)
streamlit run app/dashboard.py             # open the dashboard

python -m pytest                           # run tests
```

The dashboard only shows new data while `src.ingestion` is running. It
shows a warning when no prediction has arrived for 30 seconds.

## Data versioning (DVC + DagsHub)

`data/training_data.csv` and `data/snapshots/fraud_features.parquet` are
tracked by DVC, not Git. Git stores only the small `.dvc` pointer files;
the data lives on the DagsHub remote
`https://dagshub.com/junaaaak/Fraud-Detection-MLOps.dvc`.

One-time auth setup (stored in the git-ignored `.dvc/config.local`):

```bash
dagshub login
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user <dagshub-username>
dvc remote modify dagshub --local ask_password true
```

Then `dvc pull` to download the data, or `dvc push` after `dvc add`.

Dataset versions are Git tags (`dataset-v1`, `stage-7-data-v1`):

```bash
git switch --detach stage-7-data-v1   # go to a data version
dvc checkout
git switch -                          # back to your branch
dvc checkout
```
