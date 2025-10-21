# Run it with:
# streamlit run intraday_live_signal.py

import warnings, numpy as np, pandas as pd, yfinance as yf
import streamlit as st
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas_ta as ta
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="📈 Live Intraday Signal", layout="centered")
st.title("⚡ Real-Time Intraday BUY / SELL Signal (LightGBM)")
st.caption("Predicts next-minute move from 1-min data. Not financial advice.")

# ---------------- SIDEBAR INPUTS ----------------
col1, col2 = st.columns(2)
ticker = col1.text_input("Symbol", "AAPL").upper().strip()
train_days = col2.slider("Training window (days)", 1, 5, 3)
refresh_secs = st.number_input("Auto-refresh (seconds)", 30, 300, 60, 10)

# Auto-refresh app
st_autorefresh(interval=refresh_secs * 1000, key="refresh_key")

# ---------------- FEATURE FUNCTION ----------------
def make_features(df):
    f = pd.DataFrame(index=df.index)
    f["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    f["FISHER"] = ta.fisher(df["High"].squeeze(), df["Low"].squeeze(), 9)["FISHERT_9_1"]
    f["RSI_3"] = ta.rsi(df["Close"].squeeze(), 3)
    f["SQUEEZE"] = ta.squeeze(
        df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze()
    )["SQZ_20_2.0_20_1.5"]
    f["ER"] = ta.er(df["Close"].squeeze(), 10)
    vol_mean = df["Volume"].rolling(10).mean()
    vol_std = df["Volume"].rolling(10).std()
    f["volume_zscore"] = (df["Volume"] - vol_mean) / (vol_std + 1e-6)
    f["vol_price_interaction"] = f["volume_zscore"] * f["log_return"]
    f["vol_jump"] = f["log_return"].rolling(3).std() / (
        f["log_return"].rolling(6).std() + 1e-6
    )
    for lag in [1, 2, 3]:
        f[f"log_return_lag{lag}"] = f["log_return"].shift(lag)
    ema_fast = df["Close"].ewm(span=5).mean()
    ema_slow = df["Close"].ewm(span=15).mean()
    f["ema_angle"] = np.arctan((ema_fast - ema_slow) / ema_slow)
    high_10 = df["High"].rolling(10).max()
    low_10 = df["Low"].rolling(10).min()
    f["price_pos_10"] = (df["Close"] - low_10) / (high_10 - low_10 + 1e-6)
    return f.dropna()

# ---------------- DATA DOWNLOAD ----------------
st.write(f"⬇️ Downloading 1-minute data for **{ticker}** …")
try:
    df = yf.download(
        tickers=ticker,
        period=f"{train_days}d",
        interval="1m",
        prepost=True,
        auto_adjust=True,
        progress=False,
    )
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

if df.empty:
    st.error("No intraday data returned. Try a different symbol or later.")
    st.stop()

# ---------------- FEATURE ENGINEERING ----------------
f = make_features(df)
future_close = df["Close"].shift(-3)  # predict ~3 bars ahead
y = (future_close.loc[f.index] > df["Close"].loc[f.index]).astype(int)
X = f.values
y = y.values.ravel()
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ---------------- TRAIN MODEL ----------------
model = LGBMClassifier(
    class_weight="balanced",
    n_estimators=3000,
    learning_rate=0.01,
    num_leaves=63,
    max_depth=7,
    min_child_samples=40,
    reg_lambda=3.0,
    reg_alpha=0.5,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(100), log_evaluation(period=0)],
)

val_acc = accuracy_score(y_val, model.predict(X_val)) * 100

# ---------------- LIVE PREDICTION ----------------
tk = yf.Ticker(ticker)
live_data = yf.download(
    ticker, period="1d", interval="1m", prepost=True, progress=False
)
latest_bar = live_data.iloc[-1]
df_live = pd.concat([df, latest_bar.to_frame().T]).drop_duplicates()
f_live = make_features(df_live)
latest_features = f_live.iloc[-1].values.reshape(1, -1)
p_up = model.predict_proba(latest_features)[0, 1]

# ---------------- SIGNAL LOGIC ----------------
if p_up >= 0.75:
    signal = "🟩 STRONG BUY"
    color = "limegreen"
elif p_up >= 0.65:
    signal = "🟢 BUY"
    color = "green"
elif p_up <= 0.25:
    signal = "🔴 STRONG SELL"
    color = "red"
elif p_up <= 0.35:
    signal = "🔻 SELL"
    color = "darkred"
else:
    signal = "⚪ HOLD"
    color = "gray"

# ---------------- PRICE ESTIMATION ----------------
live_price = getattr(tk, "fast_info", None)
latest_close = float(live_price.last_price)
recent_vol = f_live["log_return"].rolling(20).std().iloc[-1]
expected_return = (p_up - 0.5) * 2 * recent_vol
predicted_price = latest_close * (1 + expected_return)

# ---------------- DISPLAY ----------------
st.subheader(f"📊 {ticker} — Live Forecast")
c1, c2, c3 = st.columns(3)
c1.metric("Validation Accuracy", f"{val_acc:.2f}%")
c2.metric("Probability Up", f"{p_up:.3f}")
c3.metric("Current Price", f"${latest_close:.2f}")

st.markdown(f"<h2 style='color:{color}'>{signal}</h2>", unsafe_allow_html=True)
st.metric("Predicted Next Price (~2-3 min ahead)", f"${predicted_price:.2f}")

# ---------------- CONFIDENT-TRADE ACCURACY ----------------
y_prob = model.predict_proba(X_val)[:, 1]
mask = (y_prob > 0.65) | (y_prob < 0.35)
if mask.sum() > 0:
    acc_conf = accuracy_score(y_val[mask], (y_prob[mask] > 0.5).astype(int)) * 100
    st.caption(f"🎯 Confident-Trade Accuracy: **{acc_conf:.2f}%**")
else:
    st.caption("No confident trades found in validation.")

st.caption(f"Auto-refreshes every {refresh_secs} seconds.")
