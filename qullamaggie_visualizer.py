import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Qullamaggie HTF Scanner & Visualizer", layout="wide")
st.title("🚩 Qullamaggie High Tight Flag & Shelf Scanner")
st.caption("Scans watchlists in a single batch, filters for momentum surfing, and plots exact shelf entry triggers.")

# ============================================================
# PRE-BUILT WATCHLIST UNIVERSES
# ============================================================
UNIVERSES = {
    "Momentum & Tech Leaders": [
        "NVDA", "AMD", "TSLA", "AAPL", "PLTR", "MU", "SMCI", "AMZN", 
        "MSFT", "AVGO", "GOOGL", "META", "ARM", "GE", "CSX", "NFLX",
        "QCOM", "MRVL", "KLAC", "LRCX", "ASML", "PANW", "CRWD", "NOW"
    ],
    "Broad Liquid Watchlist": [
        "NVDA", "AMD", "TSLA", "AAPL", "PLTR", "MU", "AMZN", "MSFT", 
        "AVGO", "GOOGL", "META", "ARM", "GE", "CSX", "NFLX", "SLV", 
        "GLD", "XOM", "CVX", "FCX", "BA", "UNH", "JPM", "COST", "LLY"
    ],
    "Custom / Manual Input": []
}

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Universe Selection")
selected_universe = st.sidebar.selectbox("Choose Universe", list(UNIVERSES.keys()), index=0)

if selected_universe == "Custom / Manual Input":
    default_text = "NVDA, AMD, ARM, PLTR, TSLA, AAPL, GE, CSX"
else:
    default_text = ", ".join(UNIVERSES[selected_universe])

watchlist_input = st.sidebar.text_area(
    "Tickers (comma or newline separated)",
    value=default_text,
    height=120
)

st.sidebar.header("Scanner Criteria")
min_surge_pct = st.sidebar.slider("Min Prior Surge %", 15, 60, 30, help="Minimum run-up over the last 40 trading days.")
max_dist_ribbon_pct = st.sidebar.slider("Max Distance from 10/20 EMA %", 1.0, 6.0, 3.0, help="Must be within this % of 10 or 20 EMA (not extended).")

st.sidebar.header("Shelf Visualizer")
shelf_lookback = st.sidebar.slider("Shelf Lookback Window (Days)", 3, 12, 6, help="Recent bars defining the tight resting shelf.")
buffer_cents = st.sidebar.number_input("Entry Buffer ($)", value=0.05, step=0.01)

def parse_tickers(raw):
    return [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]

# ============================================================
# BATCH SCANNING & ANALYSIS
# ============================================================
@st.cache_data(ttl=1800)
def download_all_data(tickers):
    if not tickers:
        return pd.DataFrame()
    # Single batch HTTP download prevents API rate-limiting
    df = yf.download(tickers, period="1y", interval="1d", group_by="ticker", auto_adjust=True, progress=False)
    return df

def run_screener(tickers, batch_df, min_surge, max_dist, s_lookback, buffer):
    candidates = {}
    
    for t in tickers:
        try:
            if isinstance(batch_df.columns, pd.MultiIndex):
                if t not in batch_df.columns.levels[0]:
                    continue
                df = batch_df[t].dropna()
            else:
                df = batch_df.dropna()
                
            if len(df) < 50:
                continue

            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            vol = df["Volume"]

            # Technical Indicators
            df["EMA10"] = close.ewm(span=10, adjust=False).mean()
            df["EMA20"] = close.ewm(span=20, adjust=False).mean()
            df["SMA50"] = close.rolling(50).mean()
            df["VolSMA50"] = vol.rolling(50).mean()

            curr_c = close.iloc[-1]
            curr_10 = df["EMA10"].iloc[-1]
            curr_20 = df["EMA20"].iloc[-1]
            curr_50 = df["SMA50"].iloc[-1]

            # 1. Momentum Surge Check (past 40 bars)
            low_40 = low.tail(40).min()
            high_40 = high.tail(40).max()
            surge_pct = ((high_40 - low_40) / low_40) * 100
            had_surge = surge_pct >= min_surge

            # 2. Moving Average Alignment
            trend_ok = (curr_c > curr_20) and (curr_10 > curr_20 > curr_50)

            # 3. Surfing the Ribbon (Not Extended)
            dist_10 = abs(curr_c - curr_10) / curr_10 * 100
            dist_20 = abs(curr_c - curr_20) / curr_20 * 100
            near_ribbon = (dist_10 <= max_dist) or (dist_20 <= max_dist)

            # 4. Volume Contraction (3-day average < 50-day average)
            recent_vol = vol.tail(3).mean()
            vol_dry = recent_vol < df["VolSMA50"].iloc[-1]

            if had_surge and trend_ok and near_ribbon and vol_dry:
                # Shelf Calculations
                shelf_slice = df.tail(s_lookback)
                shelf_high = shelf_slice["High"].max()
                shelf_low = shelf_slice["Low"].min()
                entry_trigger = shelf_high + buffer
                stop_loss = shelf_low
                risk_pct = ((entry_trigger - stop_loss) / entry_trigger) * 100

                candidates[t] = {
                    "df": df,
                    "Price": round(curr_c, 2),
                    "Surge %": round(surge_pct, 1),
                    "% Off 10 EMA": round(dist_10, 1),
                    "Buy Stop": round(entry_trigger, 2),
                    "Stop Loss": round(stop_loss, 2),
                    "Risk %": round(risk_pct, 2),
                    "shelf_start": shelf_slice.index[0],
                    "shelf_end": shelf_slice.index[-1],
                    "shelf_high": shelf_high,
                    "shelf_low": shelf_low,
                    "entry_trigger": entry_trigger,
                    "stop_loss": stop_loss
                }
        except Exception:
            continue
            
    return candidates

# ============================================================
# EXECUTION & UI
# ============================================================
tickers = parse_tickers(watchlist_input)

if st.sidebar.button("🔍 Run Full Scan", type="primary"):
    with st.spinner(f"Downloading and scanning {len(tickers)} tickers in batch..."):
        batch_data = download_all_data(tickers)
        results = run_screener(tickers, batch_data, min_surge_pct, max_dist_ribbon_pct, shelf_lookback, buffer_cents)
        st.session_state["scan_results"] = results

if "scan_results" in st.session_state:
    results = st.session_state["scan_results"]
    
    if not results:
        st.warning("No stocks currently meet all 4 criteria. Try slightly lowering 'Min Prior Surge %' or raising 'Max Distance %'.")
    else:
        st.subheader(f"✅ Found {len(results)} Qualified Setup(s)")
        
        # Build Summary Table
        summary_rows = []
        for t, data in results.items():
            summary_rows.append({
                "Ticker": t,
                "Price": f"${data['Price']:.2f}",
                "40D Surge": f"+{data['Surge %']:.1f}%",
                "Dist to 10 EMA": f"{data['% Off 10 EMA']:.1f}%",
                "Buy Stop Trigger": f"${data['Buy Stop']:.2f}",
                "Stop Loss": f"${data['Stop Loss']:.2f}",
                "Risk %": f"{data['Risk %']:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Interactive Chart Dropdown
        selected_ticker = st.selectbox("Select a setup to inspect visually:", list(results.keys()))
        
        if selected_ticker:
            meta = results[selected_ticker]
            df = meta["df"].tail(120)  # Focus on the last 5-6 months
            
            # Key Setup Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${meta['Price']:.2f}")
            col2.metric("Prior Surge Run", f"+{meta['Surge %']:.1f}%")
            col3.metric("Buy Stop Order", f"${meta['entry_trigger']:.2f}")
            col4.metric("Risk Span", f"{meta['Risk %']:.2f}%")

            # Main Price Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Price"
            ))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA10"], name="10 EMA", line=dict(color="orange", width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="20 EMA", line=dict(color="cyan", width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="50 SMA", line=dict(color="magenta", width=1, dash="dot")))

            # Shaded Shelf Box
            fig.add_shape(
                type="rect",
                x0=meta["shelf_start"], y0=meta["shelf_low"],
                x1=meta["shelf_end"], y1=meta["shelf_high"],
                fillcolor="rgba(255, 215, 0, 0.18)",
                line=dict(color="rgba(255, 215, 0, 0.7)", width=1),
            )

            # Buy Stop Line
            fig.add_shape(
                type="line",
                x0=meta["shelf_start"], y0=meta["entry_trigger"],
                x1=df.index[-1], y1=meta["entry_trigger"],
                line=dict(color="lime", width=2, dash="dash")
            )
            fig.add_annotation(
                x=df.index[-1], y=meta["entry_trigger"],
                text=f"BUY STOP: ${meta['entry_trigger']:.2f}",
                showarrow=False, xanchor="left", font=dict(color="lime", size=12)
            )

            # Stop Loss Line
            fig.add_shape(
                type="line",
                x0=meta["shelf_start"], y0=meta["stop_loss"],
                x1=df.index[-1], y1=meta["stop_loss"],
                line=dict(color="red", width=2, dash="dot")
            )
            fig.add_annotation(
                x=df.index[-1], y=meta["stop_loss"],
                text=f"STOP: ${meta['stop_loss']:.2f}",
                showarrow=False, xanchor="left", font=dict(color="red", size=12)
            )

            fig.update_layout(
                height=550,
                title=f"{selected_ticker} — High Tight Flag Shelf on the 10/20 EMA",
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=80, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Volume Sub-pane
            vol_fig = go.Figure()
            vol_colors = np.where(df["Close"] >= df["Open"], "rgba(0,180,0,0.6)", "rgba(220,0,0,0.6)")
            vol_fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume"))
            vol_fig.add_trace(go.Scatter(x=df.index, y=df["VolSMA50"], line=dict(color="orange", width=1), name="50d Avg Vol"))
            vol_fig.update_layout(height=170, margin=dict(l=20, r=80, t=10, b=10), xaxis_rangeslider_visible=False)
            st.plotly_chart(vol_fig, use_container_width=True)
else:
    st.info("👈 Select your universe in the sidebar and click **Run Full Scan** to begin.")
