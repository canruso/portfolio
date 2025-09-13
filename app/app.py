from __future__ import annotations

import io
from datetime import datetime
from typing import Optional, Iterable, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Dashboard — Step 1", layout="wide")

# ---------------- Utilities ----------------

@st.cache_data(show_spinner=False)
def get_stock_last_price(symbol: str) -> Optional[float]:
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1d")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        return None
    return None

@st.cache_data(show_spinner=False)
def get_option_chain(underlying: str, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = yf.Ticker(underlying)
    # Choose the exact expiration if available, else nearest
    try:
        options = t.options
    except Exception:
        options = []

    selected = expiration
    if options and expiration not in options:
        # pick nearest by absolute date difference
        try:
            ex_target = pd.Timestamp(expiration)
            selected = min(options, key=lambda x: abs(pd.Timestamp(x) - ex_target))
        except Exception:
            selected = options[0] if options else expiration

    try:
        chain = t.option_chain(selected)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None

def get_option_price(underlying: str, expiration: str, strike: float, right: str) -> Optional[float]:
    calls, puts = get_option_chain(underlying, expiration)
    if calls.empty and puts.empty:
        return None
    df = calls if str(right).upper().startswith("C") else puts
    # match by strike with tolerance
    strike = float(strike)
    row = df.loc[(df["strike"] - strike).abs() < 1e-6]
    if row.empty:
        # try nearest strike
        if "strike" in df.columns and not df.empty:
            idx = (df["strike"] - strike).abs().idxmin()
            row = df.loc[[idx]]
        else:
            return None
    last = _safe_float(row["lastPrice"].iloc[0]) if "lastPrice" in row.columns else None
    bid  = _safe_float(row["bid"].iloc[0]) if "bid" in row.columns else None
    ask  = _safe_float(row["ask"].iloc[0]) if "ask" in row.columns else None
    if last and last > 0:
        return last
    if bid is not None and ask is not None and (bid > 0 or ask > 0):
        return ( (bid or 0.0) + (ask or 0.0) ) / ( (1 if bid is not None else 0) + (1 if ask is not None else 0) or 1 )
    return None

def load_positions(default_path: str = "data/positions.csv", fallback: str = "data/positions.example.csv") -> pd.DataFrame:
    # Try to load default file; else fallback to example; else empty
    try:
        return pd.read_csv(default_path)
    except Exception:
        try:
            return pd.read_csv(fallback)
        except Exception:
            return pd.DataFrame(columns=[
                "type","symbol","quantity","avg_price","underlying","expiration","strike","right","multiplier"
            ])

def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize columns
    for col in ["type","symbol","quantity","avg_price","underlying","expiration","strike","right","multiplier"]:
        if col not in df.columns:
            df[col] = np.nan
    df["type"] = df["type"].str.lower().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["underlying"] = df["underlying"].astype(str).str.upper().str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce").fillna(0.0)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce").fillna(100.0)  # default for equity options
    # Normalize expiration to ISO
    if "expiration" in df.columns:
        df["expiration"] = df["expiration"].apply(lambda x: pd.to_datetime(x).date().isoformat() if pd.notna(x) and str(x).strip() != "" else "")
    return df

def compute_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        typ = str(r.get("type","")).strip().lower()
        qty = float(r.get("quantity") or 0.0)
        avg = float(r.get("avg_price") or 0.0)

        if typ == "stock":
            sym = r.get("symbol","").strip().upper()
            px = get_stock_last_price(sym) if sym else None
            mv = qty * (px or 0.0)
            cb = qty * avg
            row = {
                "type": "stock",
                "symbol": sym,
                "quantity": qty,
                "avg_price": avg,
                "current_price": px,
                "market_value": mv,
                "cost_basis": cb,
                "unrealized_pl": mv - cb,
                "pct_return": (mv / cb - 1.0) if cb else np.nan,
            }
            rows.append(row)

        elif typ == "option":
            und = r.get("underlying","").strip().upper()
            exp = r.get("expiration","").strip()
            strike = r.get("strike")
            right = r.get("right","").strip().upper() or "C"
            mult = float(r.get("multiplier") or 100.0)

            px = None
            if und and exp and strike is not None:
                px = get_option_price(und, exp, float(strike), right)
            mv = qty * mult * (px or 0.0)
            cb = qty * mult * avg
            sym = f"{und} {exp} {strike}{right}"
            row = {
                "type": "option",
                "symbol": sym,
                "quantity": qty,
                "avg_price": avg,
                "current_price": px,
                "market_value": mv,
                "cost_basis": cb,
                "unrealized_pl": mv - cb,
                "pct_return": (mv / cb - 1.0) if cb else np.nan,
                "underlying": und,
                "expiration": exp,
                "strike": strike,
                "right": right,
                "multiplier": mult,
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    return out

def format_currency(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${x:,.2f}"

# ---------------- UI ----------------

st.title("💼 Portfolio Dashboard — Step 1")
st.caption("Simple snapshot of positions, market value, and P/L.")

# Sidebar controls
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload positions CSV", type=["csv"], help="If omitted, the app will try data/positions.csv, else use the example.")
    st.button("Refresh prices", on_click=lambda: (get_stock_last_price.clear(), get_option_chain.clear()), help="Clear cache and refetch quotes.")

# Load positions
if uploaded is not None:
    try:
        positions = pd.read_csv(uploaded)
        st.success("Loaded positions from uploaded file.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        positions = load_positions()
else:
    positions = load_positions()

positions = normalize_positions(positions)

if positions.empty:
    st.info("No positions found. Create `data/positions.csv` or upload a CSV. See the example file for the required columns.")
    st.stop()

# Snapshot computation
with st.spinner("Fetching quotes and computing snapshot..."):
    snapshot = compute_snapshot(positions)

if snapshot.empty:
    st.warning("No valid rows to display.")
    st.stop()

# Summary metrics
total_mv = float(snapshot["market_value"].sum())
total_cb = float(snapshot["cost_basis"].sum())
total_pl = float(snapshot["unrealized_pl"].sum())
pct = (total_mv / total_cb - 1.0) if total_cb else np.nan

col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Value", format_currency(total_mv))
col2.metric("Cost Basis", format_currency(total_cb))
col3.metric("Unrealized P/L", format_currency(total_pl))
col4.metric("Return", f"{pct*100:,.2f}%" if pd.notna(pct) else "—")

st.divider()

# Table
st.subheader("Positions")
display_cols = ["type","symbol","quantity","avg_price","current_price","market_value","cost_basis","unrealized_pl","pct_return"]
display = snapshot[display_cols].copy()
display["avg_price"] = display["avg_price"].apply(format_currency)
display["current_price"] = display["current_price"].apply(format_currency)
display["market_value"] = display["market_value"].apply(format_currency)
display["cost_basis"] = display["cost_basis"].apply(format_currency)
display["unrealized_pl"] = display["unrealized_pl"].apply(format_currency)
display["pct_return"] = display["pct_return"].apply(lambda x: f"{x*100:,.2f}%" if pd.notna(x) else "—")
st.dataframe(display, use_container_width=True)

# Pie chart of market value by symbol
st.subheader("Portfolio Distribution (by symbol)")
pie_df = snapshot.groupby("symbol", as_index=False)["market_value"].sum()
pie_df = pie_df[pie_df["market_value"] > 0]

if pie_df.empty:
    st.info("No positive market values to chart.")
else:
    fig, ax = plt.subplots()
    ax.pie(pie_df["market_value"], labels=pie_df["symbol"], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

st.caption("Prices use latest available data from Yahoo Finance. Option quotes rely on last or mid (bid/ask).")
