# intraday_10m_forecast.py
# Run with: streamlit run intraday_10m_forecast.py

import warnings, numpy as np, pandas as pd, yfinance as yf, time, pytz
import streamlit as st
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

try:
    from transformers import pipeline
except ImportError:
    from transformers.pipelines import pipeline
import torch

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Real-Time 10-Minute Stock Forecast", layout="centered")
st.title("⏱️ Real-Time 10-Minute Stock Forecast")
st.caption("Predicts next 10-minute return using intraday (1-min) data. Not financial advice.")

# ---------- Persistent Ticker ----------
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

# ---------- Helper Functions ----------
def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def make_minute_features(df):
    out = df.copy()
    out["ret_1"]  = out["Close"].pct_change(1)
    out["ret_5"]  = out["Close"].pct_change(5)
    out["ret_10"] = out["Close"].pct_change(10)
    out["hl_spread"] = (out["High"] - out["Low"]) / out["Close"]
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_30"] = out["ret_1"].rolling(30).std()
    out["rsi_14"] = rsi(out["Close"], 14)
    for col in ["ret_1","ret_5","ret_10","vol_10","vol_30","rsi_14","hl_spread"]:
        out[f"{col}_lag1"]  = out[col].shift(1)
        out[f"{col}_lag5"]  = out[col].shift(5)
        out[f"{col}_lag10"] = out[col].shift(10)
    idx = out.index
    out["minute"] = idx.minute
    out["hour"]   = idx.hour
    out["dow"]    = idx.dayofweek
    return out

def get_live_price(ticker):
    """Try multiple methods to get a true live price from Yahoo."""
    tk = yf.Ticker(ticker)
    # Try fast_info first
    try:
        fi = tk.fast_info
        if hasattr(fi, "last_price") and fi.last_price:
            return float(fi.last_price)
        if hasattr(fi, "regularMarketPrice") and fi.regularMarketPrice:
            return float(fi.regularMarketPrice)
    except Exception:
        pass

    # Try .info as fallback
    try:
        info = tk.info
        for key in ["regularMarketPrice", "currentPrice", "bid", "ask"]:
            if key in info and info[key] is not None and info[key] > 0:
                return float(info[key])
    except Exception:
        pass

    # Last resort: last 1m bar
    try:
        hist = tk.history(period="1d", interval="1m", auto_adjust=True)
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass

    return np.nan

@st.cache_resource(show_spinner=False)
def get_sentiment_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)

# ---------- Data Download ----------
st.write(f"⬇️ Downloading 1-minute data for **{ticker}** (last 7 days)…")
try:
    data = yf.download(
        tickers=ticker,
        period="7d",
        interval="1m",
        auto_adjust=True,
        prepost=use_prepost,
        progress=False
    )
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

if data.empty:
    st.error("No intraday data returned. Try a different symbol or later.")
    st.stop()

data = data.dropna().copy()
data.index = data.index.tz_localize("UTC") if data.index.tz is None else data.index.tz_convert("UTC")

reg = data.copy()
h = int(target_horizon_min)
reg["future_ret_h"] = reg["Close"].shift(-h) / reg["Close"] - 1.0
cut_time = reg.index.max() - pd.Timedelta(days=int(train_days))
fx = make_minute_features(reg).copy()
fx["target"] = reg["future_ret_h"]
fx = fx.dropna()
fx_recent = fx[fx.index >= cut_time].copy()
if fx_recent.empty:
    st.error("Not enough data for training.")
    st.stop()

if use_sentiment:
    st.write("🧠 Using FinBERT sentiment (placeholder score = 0)")
    fx_recent["sentiment"] = 0.0
else:
    fx_recent["sentiment"] = 0.0

# ---------- Model Training ----------
split_idx = int(len(fx_recent) * 0.8)
X_cols = [c for c in fx_recent.columns if c not in ["Open","High","Low","Close","Adj Close","Volume","target","future_ret_h"]]
X_train = fx_recent[X_cols].iloc[:split_idx]
y_train = fx_recent["target"].iloc[:split_idx]
X_valid = fx_recent[X_cols].iloc[split_idx:]
y_valid = fx_recent["target"].iloc[split_idx:]

st.write("🤖 Training Gradient Boosting model…")
model = GradientBoostingRegressor(
    n_estimators=600,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.9,
    random_state=int(random_state)
)
model.fit(X_train, y_train)

if len(X_valid) >= 50:
    valid_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, valid_pred)
    st.metric("Validation MAE (10-min return)", f"{mae:.5f}")
else:
    st.caption("Validation window too small.")

# ---------- Live Forecast (auto-updating every 15s) ----------
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=15 * 1000, key="forecast_refresh")

def signal_from_ret(r):
    if r >= 0.004:
        return "🟩 Strong Buy"
    elif r >= 0.001:
        return "🟢 Buy"
    elif r > -0.001:
        return "⚪ Neutral"
    elif r > -0.004:
        return "🔻 Sell"
    else:
        return "🔴 Strong Sell"

st.subheader(f"Live 10-Minute Forecast • {ticker}")
c1, c2, c3 = st.columns(3)
signal_box = st.empty()

try:
    # Always get the latest live quote, not last close
    live_close = get_live_price(ticker)
    if np.isnan(live_close):
        raise ValueError("Live price unavailable.")

    # Update last row’s Close in dataset to reflect real-time price
    latest_df = reg.copy()
    latest_df.iloc[-1, latest_df.columns.get_loc("Close")] = live_close

    # Recalculate features with this live price
    fx_live = make_minute_features(latest_df).iloc[[-1]]
    fx_live["sentiment"] = 0.0
    fx_live = fx_live[X_cols].fillna(method="ffill").fillna(0)

    pred_ret = float(model.predict(fx_live)[0])
    pred_price = live_close * (1 + pred_ret)

    signal = signal_from_ret(pred_ret)
    color = "green" if "Buy" in signal else "red" if "Sell" in signal else "gray"

    c1.metric("Live Price", f"${live_close:.2f}")
    c2.metric("Predicted 10-min Return", f"{pred_ret:+.3%}")
    c3.metric("Implied Price in ~10 min", f"${pred_price:.2f}")
    signal_box.markdown(f"<h3 style='color:{color}'>{signal}</h3>", unsafe_allow_html=True)

except Exception as e:
    signal_box.error(f"⚠️ Live update error: {e}")

st.caption("⚡ Uses *live quote* (not last close) for predictions — updates every 15s.")
