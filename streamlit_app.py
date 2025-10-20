# streamlit run tom_live_forecast.py
import warnings, numpy as np, pandas as pd, yfinance as yf, pytz
import streamlit as st
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from streamlit_autorefresh import st_autorefresh
import torch
from transformers import pipeline

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Real-Time 10-Minute Stock Forecast", layout="centered")
st.title("⏱️ Real-Time Stock Forecast")
st.caption("Predicts next 10–15 minute return using intraday 1-min data + TOM(9). Not financial advice.")

# ---------- Persistent ticker ----------
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "AAPL"

col1, col2, col3 = st.columns(3)
ticker_input = col1.text_input("Symbol", st.session_state["ticker"]).upper().strip()
if ticker_input and ticker_input != st.session_state["ticker"]:
    st.session_state["ticker"] = ticker_input
ticker = st.session_state["ticker"]

train_days = col2.slider("Training window (days of 1-min bars)", 1, 5, 3)
target_horizon_min = col3.number_input("Forecast horizon (minutes)", min_value=5, max_value=30, value=10, step=5)

with st.expander("Advanced"):
    use_prepost = st.checkbox("Include pre/post market", value=False)
    use_sentiment = st.checkbox("Use FinBERT sentiment (slower)", value=False)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

# ---------- Helper functions ----------
def tom(series, window=9):
    """Typical Oscillator Momentum"""
    roll_min = series.rolling(window=window).min()
    roll_max = series.rolling(window=window).max()
    tom_val = 100 * (series - roll_min) / (roll_max - roll_min)
    return tom_val.fillna(50)

def make_minute_features(df):
    out = df.copy()
    out["ret_1"]  = out["Close"].pct_change(1)
    out["ret_5"]  = out["Close"].pct_change(5)
    out["ret_10"] = out["Close"].pct_change(10)
    out["hl_spread"] = (out["High"] - out["Low"]) / out["Close"]
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_30"] = out["ret_1"].rolling(30).std()
    out["tom_9"]  = tom(out["Close"], 9)
    for col in ["ret_1","ret_5","ret_10","vol_10","vol_30","tom_9","hl_spread"]:
        out[f"{col}_lag1"]  = out[col].shift(1)
        out[f"{col}_lag5"]  = out[col].shift(5)
        out[f"{col}_lag10"] = out[col].shift(10)
    idx = out.index
    out["minute"] = idx.minute
    out["hour"]   = idx.hour
    out["dow"]    = idx.dayofweek
    return out

def in_market_hours(ts, tz="America/New_York"):
    local = ts.tz_convert(tz)
    hour = local.hour + local.minute / 60
    return (local.weekday() < 5) and (hour >= 9.5) and (hour <= 16)

def market_open_now():
    return True
    
# ---------- Download data ----------
st.write(f"⬇️ Downloading 1-minute data for **{ticker}** (last 5 days)…")
try:
    data = yf.download(
        tickers=ticker,
        period="5d",
        interval="1m",
        auto_adjust=True,
        prepost=use_prepost,
        progress=False
    )
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

if data.empty:
    st.error("No intraday data returned.")
    st.stop()

data.index = data.index.tz_localize("UTC") if data.index.tz is None else data.index.tz_convert("UTC")
reg = data[data.index.map(in_market_hours)]
if len(reg) < 500:
    reg = data

# ---------- Feature engineering ----------
h = int(target_horizon_min)
reg["future_ret_h"] = reg["Close"].shift(-h) / reg["Close"] - 1
cut_time = reg.index.max() - pd.Timedelta(days=int(train_days))
fx = make_minute_features(reg)
fx["target"] = reg["future_ret_h"]
fx = fx.dropna()
fx_recent = fx[fx.index >= cut_time]
if fx_recent.empty:
    st.error("Not enough data for training.")
    st.stop()

fx_recent["sentiment"] = 0.0

# ---------- Train model ----------
split_idx = int(len(fx_recent) * 0.8)
X_cols = [c for c in fx_recent.columns if c not in ["Open","High","Low","Close","Adj Close","Volume","target","future_ret_h"]]
X_train = fx_recent[X_cols].iloc[:split_idx]
y_train = fx_recent["target"].iloc[:split_idx]
X_valid = fx_recent[X_cols].iloc[split_idx:]
y_valid = fx_recent["target"].iloc[split_idx:]

st.write("🤖 Training Gradient Boosting model…")
model = GradientBoostingRegressor(
    n_estimators=800,
    learning_rate=0.02,
    max_depth=3,
    min_samples_split=40,
    min_samples_leaf=15,
    subsample=0.85,
    max_features="sqrt",
    random_state=int(random_state),
    loss="huber",
)
model.fit(X_train, y_train)

if len(X_valid) >= 50:
    valid_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, valid_pred)
    st.metric("Validation MAE (10-min return)", f"{mae:.5f}")

# ---------- Live forecast (updates every 15 s) ----------
st_autorefresh(interval=15 * 1000, key="forecast_refresh")

def signal_from_ret(r):
    if r >= 0.004: return "🟩 Strong Buy"
    elif r >= 0.001: return "🟢 Buy"
    elif r > -0.001: return "⚪ Neutral"
    elif r > -0.004: return "🔻 Sell"
    else: return "🔴 Strong Sell"

st.subheader(f"Live Forecast • {ticker}")
c1, c2, c3 = st.columns(3)
signal_box = st.empty()

if not market_open_now():
    signal_box.info("⏸ Market closed — updates paused.")
else:
    try:
        tk = yf.Ticker(ticker)
        # --- use the actual current quote if available ---
        live_info = getattr(tk, "fast_info", None)
        if live_info and getattr(live_info, "last_price", None):
            live_price = float(live_info.last_price)
        else:
            live_data = tk.history(period="1d", interval="1m", auto_adjust=True)
            live_price = float(live_data["Close"].iloc[-1]) if not live_data.empty else np.nan

        latest_X = fx_recent.iloc[[-1]][X_cols]
        pred_ret = float(model.predict(latest_X)[0])
        pred_price = live_price * (1 + pred_ret)
        signal = signal_from_ret(pred_ret)
        color = "green" if "Buy" in signal else "red" if "Sell" in signal else "gray"

        c1.metric("Live Price", f"${live_price:.2f}")
        c2.metric("Predicted Return (≈10 min)", f"{pred_ret:+.3%}")
        c3.metric("Implied Price", f"${pred_price:.2f}")
        signal_box.markdown(f"<h3 style='color:{color}'>{signal}</h3>", unsafe_allow_html=True)
    except Exception as e:
        signal_box.error(f"⚠️ Live update error: {e}")

st.caption("⚡ Auto-updates every 15 seconds — TOM(9) feature + live price feed.")
