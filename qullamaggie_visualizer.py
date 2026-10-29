import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="High Tight Flag Visualizer", layout="wide")
st.title("🚩 High Tight Flag & Shelf Entry Visualizer")
st.caption("Visually inspects prior price runs, moving average surfing, and breakout pivot levels.")

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Candidates")
ticker_input = st.sidebar.text_area(
    "Paste Qualified Tickers",
    value="ARM, NVDA, CSX, AMD, PLTR, GE",
    height=100
)

st.sidebar.header("Visual Parameters")
shelf_lookback = st.sidebar.slider("Consolidation Shelf Window (Days)", 3, 15, 6, 
                                   help="Number of recent bars defining the resting shelf.")
surge_window = st.sidebar.slider("Prior Surge Lookback (Days)", 20, 60, 40, 
                                 help="Lookback period to identify the initial momentum run-up.")
buffer_cents = st.sidebar.number_input("Entry Buffer ($)", value=0.05, step=0.01)

def parse_tickers(raw):
    return [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]

tickers = parse_tickers(ticker_input)

# ============================================================
# DATA FETCHING & STRUCTURAL CALCULATIONS
# ============================================================
@st.cache_data(ttl=1800)
def load_chart_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    
    # Core EMAs & SMA
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["VolSMA50"] = df["Volume"].rolling(50).mean()
    return df

def analyze_setup(df, s_window, s_lookback, buffer):
    # 1. Identify Prior Surge Window
    surge_slice = df.tail(s_window)
    surge_low_val = surge_slice["Low"].min()
    surge_low_date = surge_slice["Low"].idxmin()
    surge_high_val = surge_slice["High"].max()
    surge_high_date = surge_slice["High"].idxmax()
    surge_pct = ((surge_high_val - surge_low_val) / surge_low_val) * 100

    # 2. Identify the Active Resting Shelf (last N bars)
    shelf_slice = df.tail(s_lookback)
    shelf_high = shelf_slice["High"].max()
    shelf_low = shelf_slice["Low"].min()
    shelf_start_date = shelf_slice.index[0]
    shelf_end_date = shelf_slice.index[-1]

    # 3. Execution Levels
    entry_trigger = shelf_high + buffer
    stop_loss = shelf_low
    risk_pct = ((entry_trigger - stop_loss) / entry_trigger) * 100

    return {
        "surge_low_val": surge_low_val,
        "surge_low_date": surge_low_date,
        "surge_high_val": surge_high_val,
        "surge_high_date": surge_high_date,
        "surge_pct": surge_pct,
        "shelf_high": shelf_high,
        "shelf_low": shelf_low,
        "shelf_start": shelf_start_date,
        "shelf_end": shelf_end_date,
        "entry_trigger": entry_trigger,
        "stop_loss": stop_loss,
        "risk_pct": risk_pct
    }

# ============================================================
# MAIN VIEW
# ============================================================
if tickers:
    selected_ticker = st.selectbox("Select Ticker to Inspect:", tickers)
    
    if selected_ticker:
        df = load_chart_data(selected_ticker)
        
        if len(df) < 60:
            st.error(f"Insufficient historical data for {selected_ticker}.")
        else:
            meta = analyze_setup(df, surge_window, shelf_lookback, buffer_cents)
            
            # Top Metrics Bar
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            m2.metric(f"{surge_window}D Surge Run", f"+{meta['surge_pct']:.1f}%")
            m3.metric("Buy Stop Trigger", f"${meta['entry_trigger']:.2f}")
            m4.metric("Risk Span (Stop-to-Pivot)", f"{meta['risk_pct']:.2f}%")

            # ----------------- PLOTLY CHART -----------------
            # Focus chart on the last 5-6 months
            chart_df = df.tail(120)

            fig = go.Figure()

            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open"],
                high=chart_df["High"],
                low=chart_df["Low"],
                close=chart_df["Close"],
                name="Price"
            ))

            # Moving Averages
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA10"], name="10 EMA",
                                     line=dict(color="orange", width=1.5)))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA20"], name="20 EMA",
                                     line=dict(color="cyan", width=1.5)))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA50"], name="50 SMA",
                                     line=dict(color="magenta", width=1, dash="dot")))

            # 1. Shaded Shelf Consolidation Box
            fig.add_shape(
                type="rect",
                x0=meta["shelf_start"], y0=meta["shelf_low"],
                x1=meta["shelf_end"], y1=meta["shelf_high"],
                fillcolor="rgba(255, 215, 0, 0.15)",
                line=dict(color="rgba(255, 215, 0, 0.6)", width=1),
            )

            # 2. Breakout Trigger Line
            fig.add_shape(
                type="line",
                x0=meta["shelf_start"], y0=meta["entry_trigger"],
                x1=chart_df.index[-1], y1=meta["entry_trigger"],
                line=dict(color="lime", width=2, dash="dash")
            )
            fig.add_annotation(
                x=chart_df.index[-1], y=meta["entry_trigger"],
                text=f"BUY STOP: ${meta['entry_trigger']:.2f}",
                showarrow=False, xanchor="left",
                font=dict(color="lime", size=11)
            )

            # 3. Stop-Loss Line
            fig.add_shape(
                type="line",
                x0=meta["shelf_start"], y0=meta["stop_loss"],
                x1=chart_df.index[-1], y1=meta["stop_loss"],
                line=dict(color="red", width=2, dash="dot")
            )
            fig.add_annotation(
                x=chart_df.index[-1], y=meta["stop_loss"],
                text=f"STOP: ${meta['stop_loss']:.2f}",
                showarrow=False, xanchor="left",
                font=dict(color="red", size=11)
            )

            fig.update_layout(
                height=550,
                title=f"{selected_ticker} — High Tight Flag Setup & Shelf Structure",
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=80, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Volume Pane
            vol_fig = go.Figure()
            vol_colors = np.where(chart_df["Close"] >= chart_df["Open"], "rgba(0,180,0,0.6)", "rgba(220,0,0,0.6)")
            vol_fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], marker_color=vol_colors, name="Volume"))
            vol_fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["VolSMA50"], line=dict(color="orange", width=1), name="50d Avg Vol"))
            vol_fig.update_layout(height=180, margin=dict(l=20, r=80, t=10, b=10), xaxis_rangeslider_visible=False)
            st.plotly_chart(vol_fig, use_container_width=True)
else:
    st.info("Enter at least one ticker in the sidebar.")
