"""
Live fraud monitoring dashboard.

Run with:
    streamlit run app/dashboard.py

Reads from the same SQLite database that ingestion.py writes to,
using a read-only connection so it never blocks the pipeline.
"""

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import DB_PATH, to_utc_iso  # noqa: E402

# All timestamps are stored in UTC; display them in the machine's
# local timezone so tables, charts and "Last updated" agree.
LOCAL_TZ = datetime.now().astimezone().tzinfo

# If no prediction has been written for this long, the ingestion
# service is probably not running.
STALE_AFTER = timedelta(seconds=30)

# Window start is floored to this many seconds so the cache key stays
# stable between reruns instead of changing every microsecond.
WINDOW_GRANULARITY_SECONDS = 10


# ----------------------------------------
# PAGE SETUP
# ----------------------------------------

st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="🛡️",
    layout="wide",
)


# ----------------------------------------
# 1. DATA ACCESS
# ----------------------------------------

def get_connection():
    # Read-only URI mode: the dashboard can never lock or modify
    # the database while ingestion.py is writing to it.
    return sqlite3.connect(
        f"file:{DB_PATH.as_posix()}?mode=ro",
        uri=True,
        timeout=5,
    )


def table_exists(conn, name):
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


@st.cache_data(ttl=10, show_spinner=False)
def load_model_versions():
    """Model versions with predictions, most recently used first."""

    if not DB_PATH.exists():
        return []
    with get_connection() as conn:
        if not table_exists(conn, "fraud_predictions"):
            return []
        rows = conn.execute("""
            SELECT model_version
            FROM fraud_predictions
            GROUP BY model_version
            ORDER BY MAX(predicted_at) DESC
        """).fetchall()
    return [row[0] for row in rows]


@st.cache_data(ttl=2, show_spinner=False)
def load_transactions(since_iso, model_version):
    """Accepted transactions joined with their features, predictions and
    (when available) simulator ground-truth labels."""

    with get_connection() as conn:
        if not table_exists(conn, "fraud_predictions"):
            return pd.DataFrame()

        if table_exists(conn, "transaction_labels"):
            label_select = "l.is_fraud"
            label_join = (
                "LEFT JOIN transaction_labels l "
                "ON l.transaction_id = t.transaction_id"
            )
        else:
            label_select, label_join = "NULL AS is_fraud", ""

        df = pd.read_sql_query(f"""
            SELECT
                t.transaction_id,
                t.customer_id,
                t.amount,
                t.merchant,
                t.city,
                t.device,
                t.timestamp,
                f.txn_count_5m,
                f.amount_sum_1h,
                f.avg_prior_amount,
                f.amount_ratio,
                f.city_changed,
                p.fraud_score,
                p.is_suspicious,
                p.model_version,
                p.predicted_at,
                {label_select}
            FROM transactions t
            JOIN fraud_predictions p
                ON p.transaction_id = t.transaction_id
            LEFT JOIN transaction_features f
                ON f.transaction_id = t.transaction_id
            {label_join}
            WHERE p.predicted_at >= ?
              AND (? IS NULL OR p.model_version = ?)
            ORDER BY p.predicted_at DESC
        """, conn, params=(since_iso, model_version, model_version))

    if not df.empty:
        df["predicted_at"] = pd.to_datetime(
            df["predicted_at"], utc=True, format="ISO8601"
        ).dt.tz_convert(LOCAL_TZ)
    return df


@st.cache_data(ttl=2, show_spinner=False)
def load_pipeline_status(since_iso):
    """Latest prediction time overall, and accepted transactions in the
    window that have no prediction (e.g. ingested before the model)."""

    with get_connection() as conn:
        if not table_exists(conn, "fraud_predictions"):
            return None, 0

        last_prediction = conn.execute(
            "SELECT MAX(predicted_at) FROM fraud_predictions"
        ).fetchone()[0]

        unscored = conn.execute("""
            SELECT COUNT(*)
            FROM transactions t
            LEFT JOIN fraud_predictions p
                ON p.transaction_id = t.transaction_id
            WHERE p.transaction_id IS NULL
              AND t.timestamp >= ?
        """, (since_iso,)).fetchone()[0]

    if last_prediction is not None:
        last_prediction = pd.to_datetime(
            last_prediction, utc=True, format="ISO8601"
        ).tz_convert(LOCAL_TZ)
    return last_prediction, unscored


@st.cache_data(ttl=2, show_spinner=False)
def load_quarantine(since_iso):
    with get_connection() as conn:
        if not table_exists(conn, "quarantine_transactions"):
            return pd.DataFrame()

        df = pd.read_sql_query("""
            SELECT transaction_id, rejection_reason, rejected_at, raw_data
            FROM quarantine_transactions
            WHERE rejected_at >= ?
            ORDER BY rejected_at DESC
        """, conn, params=(since_iso,))

    if not df.empty:
        df["rejected_at"] = pd.to_datetime(
            df["rejected_at"], utc=True, format="ISO8601"
        ).dt.tz_convert(LOCAL_TZ)
    return df


# ----------------------------------------
# 2. SIDEBAR CONTROLS
# ----------------------------------------

TIME_WINDOWS = {
    "Last 15 minutes": timedelta(minutes=15),
    "Last hour": timedelta(hours=1),
    "Last 24 hours": timedelta(days=1),
    "Last 7 days": timedelta(days=7),
    "All time": None,
}

with st.sidebar:
    st.header("Controls")

    window_label = st.selectbox(
        "Time window", list(TIME_WINDOWS), index=1
    )

    ALL_VERSIONS = "All versions"
    versions = load_model_versions()
    version_choice = st.selectbox(
        "Model version",
        versions + [ALL_VERSIONS],
        index=0,
        help="Defaults to the most recently used model, so older "
             "models' predictions don't skew the numbers.",
    )
    model_version = None if version_choice in (ALL_VERSIONS, None) else version_choice

    live = st.toggle("Live refresh", value=True)
    refresh_seconds = st.slider(
        "Refresh every (seconds)", 2, 30, 5, disabled=not live
    )

    st.divider()
    st.caption(f"Database: `{DB_PATH}`")


def window_start_iso():
    delta = TIME_WINDOWS[window_label]
    if delta is None:
        return "0000"  # sorts before every ISO timestamp
    start = datetime.now(timezone.utc) - delta
    start -= timedelta(
        seconds=start.second % WINDOW_GRANULARITY_SECONDS,
        microseconds=start.microsecond,
    )
    return to_utc_iso(start)


def render_pipeline_status(last_prediction):
    now = datetime.now(LOCAL_TZ)

    if last_prediction is None:
        st.warning(
            "No predictions have been written yet. Start the pipeline "
            "with `python -m src.ingestion`."
        )
        return

    age = now - last_prediction
    if age > STALE_AFTER:
        minutes = int(age.total_seconds() // 60)
        ago = f"{minutes} min ago" if minutes else f"{int(age.total_seconds())}s ago"
        st.warning(
            f"⏸️ Ingestion appears to be stopped: the last prediction was at "
            f"{last_prediction:%Y-%m-%d %H:%M:%S} ({ago}). The dashboard is "
            f"refreshing, but no new data is arriving. Start it with "
            f"`python -m src.ingestion`."
        )
    else:
        st.caption(
            f"🟢 Pipeline live · last prediction "
            f"{last_prediction:%H:%M:%S}"
        )


# ----------------------------------------
# 3. DASHBOARD SECTIONS
# ----------------------------------------

def render_kpis(txns, quarantine):
    total = len(txns)
    suspicious = int(txns["is_suspicious"].sum()) if total else 0
    fraud_rate = suspicious / total if total else 0.0
    avg_score = txns["fraud_score"].mean() if total else 0.0
    flagged_amount = (
        txns.loc[txns["is_suspicious"] == 1, "amount"].sum()
        if total else 0.0
    )

    cols = st.columns(5)
    cols[0].metric("Transactions scored", f"{total:,}")
    cols[1].metric("Flagged suspicious", f"{suspicious:,}")
    cols[2].metric("Fraud rate", f"{fraud_rate:.1%}")
    cols[3].metric("Amount at risk", f"₹{flagged_amount:,.0f}")
    cols[4].metric("Quarantined", f"{len(quarantine):,}")

    st.caption(f"Average fraud score: {avg_score:.3f}")

    render_model_quality(txns)


def render_model_quality(txns):
    """Live precision / recall against the simulator's ground truth."""

    if txns.empty or "is_fraud" not in txns:
        return
    labelled = txns.dropna(subset=["is_fraud"])
    if labelled.empty:
        return

    actual = labelled["is_fraud"].astype(int)
    flagged = labelled["is_suspicious"].astype(int)
    true_pos = int(((actual == 1) & (flagged == 1)).sum())
    precision = true_pos / flagged.sum() if flagged.sum() else None
    recall = true_pos / actual.sum() if actual.sum() else None

    def pct(value):
        return "–" if value is None else f"{value:.1%}"

    st.markdown("**Model quality** (vs. simulator ground truth)")
    cols = st.columns(4)
    cols[0].metric("Labelled transactions", f"{len(labelled):,}")
    cols[1].metric("Actual fraud rate", pct(actual.mean()))
    cols[2].metric(
        "Precision", pct(precision),
        help="Of the transactions flagged, how many were really fraud.",
    )
    cols[3].metric(
        "Recall", pct(recall),
        help="Of the real frauds, how many were flagged.",
    )


def render_trends(txns):
    st.subheader("Activity over time")

    # Pick a bucket size that suits the selected window.
    span = txns["predicted_at"].max() - txns["predicted_at"].min()
    if span <= timedelta(hours=1):
        freq = "1min"
    elif span <= timedelta(days=1):
        freq = "15min"
    else:
        freq = "1h"

    by_time = (
        txns.set_index("predicted_at")
        .resample(freq)
        .agg(
            transactions=("transaction_id", "count"),
            suspicious=("is_suspicious", "sum"),
            avg_fraud_score=("fraud_score", "mean"),
        )
    )

    left, right = st.columns(2)
    with left:
        st.caption(f"Transactions vs. flagged (per {freq})")
        st.line_chart(by_time[["transactions", "suspicious"]])
    with right:
        st.caption(f"Average fraud score (per {freq})")
        st.line_chart(by_time[["avg_fraud_score"]])


def render_breakdowns(txns):
    st.subheader("Where suspicious activity happens")

    cols = st.columns(3)
    for col, field in zip(cols, ["merchant", "city", "device"]):
        breakdown = (
            txns.groupby(field)
            .agg(
                total=("transaction_id", "count"),
                suspicious=("is_suspicious", "sum"),
            )
            .assign(rate=lambda d: d["suspicious"] / d["total"])
            .sort_values("rate", ascending=False)
        )
        with col:
            st.caption(f"Fraud rate by {field}")
            st.bar_chart(breakdown["rate"], horizontal=True)

    st.caption("Fraud score distribution")
    bins = pd.cut(
        txns["fraud_score"],
        bins=[i / 10 for i in range(11)],
        include_lowest=True,
    )
    histogram = bins.value_counts().sort_index()
    histogram.index = histogram.index.astype(str)
    st.bar_chart(histogram)


def render_alerts(txns):
    st.subheader("Suspicious transactions")

    alerts = txns[txns["is_suspicious"] == 1]
    if alerts.empty:
        st.success("No suspicious transactions in this window.")
        return

    st.dataframe(
        alerts[[
            "predicted_at", "transaction_id", "customer_id", "amount",
            "merchant", "city", "device", "txn_count_5m",
            "amount_ratio", "city_changed", "fraud_score",
        ]],
        hide_index=True,
        width="stretch",
        column_config={
            "predicted_at": st.column_config.DatetimeColumn(
                "Time", format="YYYY-MM-DD HH:mm:ss"
            ),
            "amount": st.column_config.NumberColumn(
                "Amount", format="₹%.2f"
            ),
            "amount_ratio": st.column_config.NumberColumn(
                "Amount ratio", format="%.1fx"
            ),
            "city_changed": st.column_config.CheckboxColumn(
                "City changed"
            ),
            "fraud_score": st.column_config.ProgressColumn(
                "Fraud score", min_value=0.0, max_value=1.0,
                format="%.3f",
            ),
        },
    )


def render_recent(txns):
    st.subheader("Latest transactions")
    recent = txns.head(50).copy()
    recent["status"] = recent["is_suspicious"].map(
        {1: "🚨 Suspicious", 0: "✅ Normal"}
    )
    st.dataframe(
        recent[[
            "predicted_at", "status", "transaction_id", "customer_id",
            "amount", "merchant", "city", "device", "fraud_score",
            "model_version",
        ]],
        hide_index=True,
        width="stretch",
        column_config={
            "predicted_at": st.column_config.DatetimeColumn(
                "Time", format="HH:mm:ss"
            ),
            "amount": st.column_config.NumberColumn(
                "Amount", format="₹%.2f"
            ),
            "fraud_score": st.column_config.ProgressColumn(
                "Fraud score", min_value=0.0, max_value=1.0,
                format="%.3f",
            ),
        },
    )


def render_quarantine(quarantine):
    st.subheader("Data quality: quarantined transactions")

    if quarantine.empty:
        st.success("No transactions were quarantined in this window.")
        return

    # A single rejection can carry several reasons joined by "; ".
    reasons = (
        quarantine["rejection_reason"]
        .str.split("; ")
        .explode()
        .value_counts()
    )

    left, right = st.columns([1, 2])
    with left:
        st.caption("Rejection reasons")
        st.bar_chart(reasons, horizontal=True)
    with right:
        st.caption("Most recent rejections")
        st.dataframe(
            quarantine.head(50),
            hide_index=True,
            width="stretch",
            column_config={
                "rejected_at": st.column_config.DatetimeColumn(
                    "Time", format="YYYY-MM-DD HH:mm:ss"
                ),
            },
        )


def render_customer_lookup(txns):
    st.subheader("Customer lookup")

    customers = sorted(txns["customer_id"].dropna().unique())
    customer = st.selectbox(
        "Customer", customers, index=None,
        placeholder="Pick a customer to see their history",
    )
    if customer is None:
        return

    history = txns[txns["customer_id"] == customer]
    cols = st.columns(3)
    cols[0].metric("Transactions", len(history))
    cols[1].metric("Flagged", int(history["is_suspicious"].sum()))
    cols[2].metric("Total spent", f"₹{history['amount'].sum():,.0f}")

    st.line_chart(
        history.set_index("predicted_at")[["fraud_score"]]
    )


# ----------------------------------------
# 4. MAIN LAYOUT
# ----------------------------------------

st.title("🛡️ Banking Fraud Detection Monitor")


@st.fragment(run_every=refresh_seconds if live else None)
def live_dashboard():
    if not DB_PATH.exists():
        st.warning(
            "No database found yet. Start the pipeline with "
            "`python -m src.ingestion`."
        )
        return

    since = window_start_iso()

    try:
        txns = load_transactions(since, model_version)
        quarantine = load_quarantine(since)
        last_prediction, unscored = load_pipeline_status(since)
    except sqlite3.OperationalError as error:
        st.error(f"Could not read the database: {error}")
        return

    render_pipeline_status(last_prediction)

    st.caption(
        f"Window: {window_label} · Last updated "
        f"{datetime.now(LOCAL_TZ):%H:%M:%S} · Times shown in "
        f"{datetime.now(LOCAL_TZ):%Z}"
    )

    render_kpis(txns, quarantine)

    if unscored:
        st.caption(
            f"ℹ️ {unscored:,} accepted transactions in this window have no "
            f"fraud prediction (ingested before the model was added) and "
            f"are not included above."
        )

    if txns.empty:
        st.info(
            "No scored transactions in this window yet. "
            "Make sure `ingestion.py` is running."
        )
        render_quarantine(quarantine)
        return

    tab_overview, tab_alerts, tab_quality, tab_lookup = st.tabs(
        ["Overview", "Alerts", "Data quality", "Customer lookup"]
    )

    with tab_overview:
        render_trends(txns)
        render_breakdowns(txns)
        render_recent(txns)

    with tab_alerts:
        render_alerts(txns)

    with tab_quality:
        render_quarantine(quarantine)

    with tab_lookup:
        render_customer_lookup(txns)


live_dashboard()
