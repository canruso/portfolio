from __future__ import annotations

import io
from datetime import datetime
from typing import Optional, Iterable, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px

from accesors.crypto_accesor import CryptoAccessor

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

def compute_crypto_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    # Initialize the accessor
    accessor = CryptoAccessor()

    # Filter the crypto data that is going to be submitted in a batch
    crypto_df = df[df["type"] == "crypto"]
    if crypto_df.empty:
        return pd.DataFrame()
    
    # Extract crypto symbols for batch request
    crypto_symbols = crypto_df["symbol"].tolist()
    crypto_str = ' '.join(crypto_symbols)

    # Make the batch API call to get spot prices
    crypto_prices = accessor.get_spot_price(crypto_str)
    
    # If no prices returned, return empty DataFrame
    if not crypto_prices:
        return pd.DataFrame()

    # Build the crypto snapshot DataFrame
    rows = []
    for idx, (_, r) in enumerate(crypto_df.iterrows()):
        if idx < len(crypto_prices):  # Ensure prices are available for the position
            sym = r.get("symbol", "").strip().upper()
            qty = float(r.get("quantity") or 0.0)
            avg = float(r.get("avg_price") or 0.0)
            px = crypto_prices[idx]
            
            mv = qty * (px or 0.0)
            cb = qty * avg
            
            row = {
                "type": "crypto",
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
    
    return pd.DataFrame(rows)

def compute_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Compute snapshot for non-crypto positions (stocks, options, etc.)"""
    rows = []
    for _, r in df.iterrows():
        typ = str(r.get("type","")).strip().lower()
        qty = float(r.get("quantity") or 0.0)
        avg = float(r.get("avg_price") or 0.0)

        # Skip crypto positions - they are handled in compute_crypto_snapshot()
        if typ == "crypto":
            continue

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

def format_pct_change(x: float) -> str:
    if pd.isna(x):
        return "—"
    color = "green" if x > 0 else "red"
    # This is a bit of a hack for st.dataframe, which doesn't render HTML well.
    # It uses unicode characters to simulate color.
    return f'{"+" if x > 0 else ""}{x:.2%}'

# ---------------- UI ----------------

st.title("💼 Portfolio Dashboard — Step 1")
st.caption("Simple snapshot of positions, market value, and P/L.")

# Sidebar controls
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload positions CSV", type=["csv"], help="If omitted, the app will try data/positions.csv, else use the example.")

def clear_all_caches():
    get_stock_last_price.clear()
    get_option_chain.clear()
    
    st.cache_data.clear()  # This clears all cache_data (since we want to fetch fresh data)

st.button("Refresh prices", on_click=clear_all_caches, help="Clear cache and refetch quotes.")

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
    crypto_snapshot = compute_crypto_snapshot(positions)

    # Fetch and merge momentum data
    if not crypto_snapshot.empty:
        crypto_symbols = crypto_snapshot["symbol"].tolist()
        accessor = CryptoAccessor()
        momentum_df = accessor.get_crypto_momentum(crypto_symbols)
        if not momentum_df.empty:
            crypto_snapshot = pd.merge(crypto_snapshot, momentum_df, on="symbol", how="left")

if snapshot.empty and crypto_snapshot.empty:
    st.warning("No valid rows to display.")
    st.stop()

# Traditional Portfolio Section
if not snapshot.empty:
    st.header("📊 Traditional Portfolio")
    
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

# Crypto Portfolio Section
if not crypto_snapshot.empty:
    st.header("₿ Crypto Portfolio")
    
    # Crypto summary metrics
    crypto_mv = float(crypto_snapshot["market_value"].sum())
    crypto_cb = float(crypto_snapshot["cost_basis"].sum())
    crypto_pl = float(crypto_snapshot["unrealized_pl"].sum())
    crypto_pct = (crypto_mv / crypto_cb - 1.0) if crypto_cb else np.nan

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Crypto Value", format_currency(crypto_mv))
    col2.metric("Cost Basis", format_currency(crypto_cb))
    col3.metric("Unrealized P/L", format_currency(crypto_pl))
    col4.metric("Return", f"{crypto_pct*100:,.2f}%" if pd.notna(crypto_pct) else "—")

    st.divider()

    # Crypto table
    st.subheader("Crypto Positions")
    crypto_display_cols = [
        "symbol", "quantity", "avg_price", "current_price", 
        "market_value", "cost_basis", "unrealized_pl", "pct_return",
        "1D_change", "12D_change", "26D_change", "52D_change"
    ]
    # Filter for columns that actually exist in the dataframe
    existing_cols = [col for col in crypto_display_cols if col in crypto_snapshot.columns]
    crypto_display = crypto_snapshot[existing_cols].copy()
    
    # --- UI Enhancement: Apply Coloring ---
    def style_pct_change(val):
        if pd.isna(val) or val == 0:
            color = 'gray'
        else:
            color = 'green' if val > 0 else 'red'
        return f'color: {color}'

    format_dict = {
        "avg_price": format_currency,
        "current_price": format_currency,
        "market_value": format_currency,
        "cost_basis": format_currency,
        "unrealized_pl": format_currency,
        "pct_return": '{:+.2%}',
        "1D_change": '{:+.2%}',
        "12D_change": '{:+.2%}',
        "26D_change": '{:+.2%}',
        "52D_change": '{:+.2%}',
    }
    
    styler = crypto_display.style
    styler.format(format_dict)
    
    # Apply color styling only to percentage columns that exist
    pct_cols = [col for col in format_dict if '_change' in col or '_return' in col]
    existing_pct_cols = [col for col in pct_cols if col in crypto_display.columns]
    
    for col in existing_pct_cols:
        styler.map(style_pct_change, subset=[col])
        
    st.dataframe(styler, use_container_width=True)

    # Crypto pie chart (using Plotly for a cleaner look)
    st.subheader("Crypto Distribution")
    crypto_pie_df = crypto_snapshot.groupby("symbol", as_index=False)["market_value"].sum()
    crypto_pie_df = crypto_pie_df[crypto_pie_df["market_value"] > 0].sort_values("market_value", ascending=False)

    if crypto_pie_df.empty:
        st.info("No positive crypto values to chart.")
    else:
        fig = px.pie(
            crypto_pie_df,
            values='market_value',
            names='symbol',
            hole=.4, # Creates the donut chart effect
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Market Value: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>"
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

st.caption("Prices use latest available data from Yahoo Finance. Option quotes rely on last or mid (bid/ask).")
