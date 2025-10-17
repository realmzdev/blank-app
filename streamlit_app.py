import streamlit as st
import warnings, numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from GoogleNews import GoogleNews
from transformers import pipeline
import torch

warnings.filterwarnings("ignore")

# ===== Helper: Technical Indicators =====
def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def make_features(df):
    out = df.copy()
    out["ret_1d"]  = out["Close"].pct_change(1)
    out["ret_5d"]  = out["Close"].pct_change(5)
    out["ret_10d"] = out["Close"].pct_change(10)
    out["hl_spread"] = (out["High"] - out["Low"]) / out["Close"]
    out["vol_5d"]  = out["ret_1d"].rolling(5).std()
    out["vol_10d"] = out["ret_1d"].rolling(10).std()
    out["rsi_14"]  = rsi(out["Close"], 14)
    for col in ["ret_1d","ret_5d","vol_5d","vol_10d","rsi_14","hl_spread"]:
        out[f"{col}_lag1"] = out[col].shift(1)
        out[f"{col}_lag5"] = out[col].shift(5)
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    return out

# ===== Helper: Sentiment =====
@st.cache_resource(show_spinner=False)
def get_sentiment_model():
    """Load FinBERT model for finance-specific sentiment"""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)

@st.cache_data(show_spinner=False)
def fetch_latest_news(ticker, days=2, max_items=10):
    googlenews = GoogleNews(period=f"{days}d")
    googlenews.search(ticker)
    news = googlenews.result()[:max_items]
    googlenews.clear()
    return [n["title"] for n in news]

def analyze_news_sentiment(headlines):
    model = get_sentiment_model()
    results = model(headlines)
    score = 0
    for r in results:
        label = r["label"].upper()
        if "POS" in label:
            score += r["score"]
        elif "NEG" in label:
            score -= r["score"]
    return score / len(results) if results else 0

# ===== Helper: Trading Signal =====
def get_signal(pred_return):
    if pred_return >= 0.10:
        return "🟩 Strong Buy"
    elif pred_return >= 0.03:
        return "🟢 Buy"
    elif pred_return > -0.03:
        return "⚪ Hold"
    elif pred_return > -0.10:
        return "🔻 Sell"
    else:
        return "🔴 Strong Sell"

# ===== Streamlit UI =====
st.set_page_config(page_title="AI Stock Forecast & News Sentiment", layout="centered")
st.title("📈 AI Stock Forecast & News Sentiment")
st.caption("Combines FinBERT financial news sentiment with price-based forecasting.")

ticker = st.text_input("Enter stock symbol:", "AAPL").upper()
horizon = st.slider("Forecast horizon (days):", 1, 20, 1)
start_date = st.date_input("Training start date:", pd.to_datetime("2015-01-01"))

if st.button("Run Forecast"):
    try:
        st.write("⏳ Downloading market data...")
        px = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
        if px.empty:
            st.error("No data found for this ticker.")
        else:
            # === Sentiment from news ===
            st.write("📰 Fetching latest news and analyzing sentiment...")
            headlines = fetch_latest_news(ticker, days=2, max_items=10)
            if headlines:
                sentiment_score = analyze_news_sentiment(headlines)
                st.subheader("Recent Headlines")
                for h in headlines:
                    st.caption(f"• {h}")
                st.write(f"🧠 Average Sentiment Score: {sentiment_score:+.2f}")
            else:
                sentiment_score = 0
                st.info("No recent headlines found. Using neutral sentiment (0).")

            # === Build features & model ===
            future_max = px["High"].shift(-1).rolling(horizon, min_periods=1).max()
            px["future_max_pct"] = future_max / px["Close"] - 1.0
            if len(px) > horizon:
                px = px.iloc[:-horizon].copy()
            fx = make_features(px).dropna().copy()

            # ✅ Add sentiment column BEFORE selecting feature_cols
            fx = fx.assign(sentiment=float(sentiment_score))

            base_exclude = ["Open","High","Low","Close","Adj Close","Volume","future_max_pct"]
            feature_cols = [c for c in fx.columns if c not in base_exclude]

            # Avoid slicing away too much data
            train_size = max(1, len(fx) - 30)
            X_train = fx[feature_cols].iloc[:train_size]
            y_train = fx["future_max_pct"].iloc[:train_size]

            st.write("🤖 Training forecasting model...")
            model = GradientBoostingRegressor(
                n_estimators=800, max_depth=4, learning_rate=0.02,
                subsample=0.9, random_state=42)
            model.fit(X_train, y_train)

            # Latest market close (safe for weekends)
            hist = yf.Ticker(ticker).history(period="5d")
            latest_price = float(hist["Close"].dropna().iloc[-1])

            latest_X = fx.iloc[[-1]][feature_cols].values
            pred = float(model.predict(latest_X))
            pred_price = latest_price * (1 + pred)
            signal = get_signal(pred)

            # === Display results (no chart) ===
            st.subheader(f"Results for {ticker}")
            st.metric("Latest Close", f"${latest_price:.2f}")
            st.metric(f"Forecasted {horizon}-day Max", f"${pred_price:.2f}", f"{pred:+.2%}")

            color = "green" if "Buy" in signal else "red" if "Sell" in signal else "gray"
            st.markdown(f"<h3 style='color:{color}'>{signal}</h3>", unsafe_allow_html=True)

            st.success("✅ Forecast complete!")

    except Exception as e:
        st.error(f"Error: {e}")
