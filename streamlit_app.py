# streamlit run realtime_forecast.py
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

st.set_page_config(page_title="10s Live Stock Forecast", layout="centered")
st.title("⚡ Real-Time 10s Stock Forecast (Intraday)")
st.caption("Updates every 10 seconds without reloading. Not financial advice.")

# ---------- Persistent Inputs ----------
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "AAPL"

ticker = st.text_input("Symbol", st.session_state["ticker"]).upper().strip()
if ticker and ticker != st.session_state["ticker"]:
    st.session_state["ticker"] = ticker

train_days = st.slider("Training window (days of 1-min bars)", 1, 5, 2)
target_horizon_min = st.number_input("Forecast horizon (minutes)", 5, 30, 10, step=5)
refresh_secs = 10  # <-- update every 10 seconds

# ---------- Helper Functions ----------
def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def make_features(df):
    out = df.copy()
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_10"] = out["Close"].pct_change(10)
    out["hl_spread"] = (out["High"] - out["Low"]) / out["Close"]
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["rsi_14"] = rsi(out["Close"], 14)
    for c in ["ret_1", "ret_5", "ret_10", "vol_10", "rsi_14", "hl_spread"]:
        out[f"{c}_lag1"] = out[c].shift(1)
        out[f"{c}_lag5"] = out[c].shift(5)
    out["minute"] = out.index.minute
    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek
    return out

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

# ---------- Main Loop ----------
st.write(f"📡 Live monitoring: {st.session_state['ticker']} — updates every {refresh_secs}s")

output_box = st.empty()  # placeholder for dynamic updates

while True:
    try:
        px = yf.download(
            st.session_state["ticker"],
            period="2d",
            interval="1m",
            auto_adjust=True,
            progress=False
        )
        if px.empty:
            output_box.error("⚠️ No data. Market closed or ticker invalid.")
        else:
            h = int(target_horizon_min)
            px["future_ret"] = px["Close"].shift(-h) / px["Close"] - 1.0
            fx = make_features(px).dropna()
            fx = fx.iloc[-int(390 * train_days):]  # limit training window (~390 min/day)
            X_cols = [c for c in fx.columns if c not in ["Open","High","Low","Close","Volume","future_ret"]]
            X_train = fx[X_cols].iloc[:-1]
            y_train = fx["future_ret"].iloc[:-1]
            model = GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42
            )
            model.fit(X_train, y_train)
            pred_ret = float(model.predict(fx[X_cols].iloc[[-1]])[0])
            live_price = float(px["Close"].iloc[-1])
            pred_price = live_price * (1 + pred_ret)
            signal = signal_from_ret(pred_ret)
            color = "green" if "Buy" in signal else "red" if "Sell" in signal else "gray"

            # --- Display in one shot (reused container) ---
            with output_box.container():
                st.subheader(f"🔹 {st.session_state['ticker']} | Updated: {pd.Timestamp.now().strftime('%H:%M:%S')}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Last Price", f"${live_price:.2f}")
                c2.metric("Predicted 10-min Return", f"{pred_ret:+.3%}")
                c3.metric("Target Price (≈10 min)", f"${pred_price:.2f}")
                st.markdown(f"<h3 style='color:{color}'>{signal}</h3>", unsafe_allow_html=True)

    except Exception as e:
        output_box.error(f"Error: {e}")

    time.sleep(refresh_secs)
    st.rerun()  # safely rerun app logic but preserve state
