"""
Live fraud monitoring dashboard.

Run with:
    streamlit run dashboard.py

Reads from the same SQLite database that ingestion.py writes to,
using a read-only connection so it never blocks the pipeline.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from database import DB_PATH


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


@st.cache_data(ttl=2, show_spinner=False)
def load_transactions(since_iso):
    """Accepted transactions joined with their features and predictions."""

    with get_connection() as conn:
        if not table_exists(conn, "fraud_predictions"):
            return pd.DataFrame()

        df = pd.read_sql_query("""
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
                p.predicted_at
            FROM transactions t
            JOIN fraud_predictions p
                ON p.transaction_id = t.transaction_id
            LEFT JOIN transaction_features f
                ON f.transaction_id = t.transaction_id
            WHERE p.predicted_at >= ?
            ORDER BY p.predicted_at DESC
        """, conn, params=(since_iso,))

    if not df.empty:
        df["predicted_at"] = pd.to_datetime(
            df["predicted_at"], utc=True, format="ISO8601"
        )
    return df


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
        )
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
    return (datetime.now(timezone.utc) - delta).isoformat()


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
            "`python ingestion.py`."
        )
        return

    since = window_start_iso()

    try:
        txns = load_transactions(since)
        quarantine = load_quarantine(since)
    except sqlite3.OperationalError as error:
        st.error(f"Could not read the database: {error}")
        return

    st.caption(
        f"Window: {window_label} · Last updated "
        f"{datetime.now().strftime('%H:%M:%S')}"
    )

    render_kpis(txns, quarantine)

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
