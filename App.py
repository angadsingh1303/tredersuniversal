# app.py - Merged: Pro Multi-Asset AI Trading + Session/Levels Detection
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import requests
import datetime
from streamlit_autorefresh import st_autorefresh
import re
import os

# ---------------------------
# Configurations
# ---------------------------

SYMBOLS = {
    "Indian Stocks": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
    "Forex": ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDINR=X"],
    "Cryptocurrencies": ["BTC-USD", "ETH-USD", "ADA-USD"],
    "Metals/Commodities": ["GC=F", "SI=F", "HG=F"],
}

TIMEFRAMES = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d",
}

# Put your Telegram credentials here (optional)
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

TRADE_FEE = 0.0005  # 0.05% per trade

# Auto-refresh every 5 seconds
st_autorefresh(interval=5 * 1000, limit=None, key="refresh")

# ---------------------------
# Utilities
# ---------------------------

def sanitize_symbol_for_filename(symbol: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', symbol)

# ---------------------------
# Fetching & Caching
# ---------------------------

@st.cache_data(ttl=60)
def fetch_data(symbol, interval, period="7d"):
    """
    Fetch historical data from yfinance and normalize column names to:
    Datetime, Open, High, Low, Close, Volume
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=interval, period=period)
        if df is None or df.empty:
            return None
        df = df.dropna()
        df = df.reset_index()
        # Some yfinance returns 'Date' column name, some 'Datetime'; unify to 'Datetime'
        if 'Date' in df.columns and 'Datetime' not in df.columns:
            df.rename(columns={"Date": "Datetime"}, inplace=True)
        if 'Datetime' not in df.columns and 'index' in df.columns:
            df.rename(columns={"index": "Datetime"}, inplace=True)
        # Clamp column names to capitalized style used by rest of script
        col_map = {}
        for c in df.columns:
            low = c.lower()
            if low == 'open': col_map[c] = 'Open'
            if low == 'high': col_map[c] = 'High'
            if low == 'low': col_map[c] = 'Low'
            if low == 'close': col_map[c] = 'Close'
            if low == 'volume': col_map[c] = 'Volume'
        if col_map:
            df.rename(columns=col_map, inplace=True)
        # Ensure Datetime is datetime dtype and timezone-aware if possible
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        # sort ascending
        df.sort_values('Datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return None

# ---------------------------
# Technical Indicators
# ---------------------------

def add_indicators(df):
    df = df.copy()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA30'] = df['Close'].ewm(span=30, adjust=False).mean()
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    df['ret_1'] = df['Close'].pct_change(1)
    df['ret_3'] = df['Close'].pct_change(3)
    df['close_lag_1'] = df['Close'].shift(1)
    df['close_lag_2'] = df['Close'].shift(2)
    df['close_lag_3'] = df['Close'].shift(3)
    df.dropna(inplace=True)
    return df

# ---------------------------
# SMC Detection
# ---------------------------

def detect_smc(df):
    df = df.copy()
    df['BOS_UP'] = False
    df['BOS_DOWN'] = False
    df['CHOCH'] = False
    highs = df['High']
    lows = df['Low']
    for i in range(2, len(df)-2):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2]:
            df.at[df.index[i], 'BOS_UP'] = True
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2]:
            df.at[df.index[i], 'BOS_DOWN'] = True
    df['EMA10_prev'] = df['EMA10'].shift(1)
    df['EMA30_prev'] = df['EMA30'].shift(1)
    df['CHOCH'] = ((df['EMA10'] > df['EMA30']) & (df['EMA10_prev'] <= df['EMA30_prev'])) | \
                  ((df['EMA10'] < df['EMA30']) & (df['EMA10_prev'] >= df['EMA30_prev']))
    df.drop(columns=['EMA10_prev', 'EMA30_prev'], inplace=True)
    return df

def detect_support_resistance(df, window=20):
    df = df.copy()
    df['Support'] = df['Low'].rolling(window, center=True).min()
    df['Resistance'] = df['High'].rolling(window, center=True).max()
    return df

def fibonacci_levels(df):
    max_p = df['High'].max()
    min_p = df['Low'].min()
    diff = max_p - min_p
    levels = {
        "Fib 0%": max_p,
        "Fib 23.6%": max_p - 0.236 * diff,
        "Fib 38.2%": max_p - 0.382 * diff,
        "Fib 50%": max_p - 0.5 * diff,
        "Fib 61.8%": max_p - 0.618 * diff,
        "Fib 78.6%": max_p - 0.786 * diff,
        "Fib 100%": min_p,
    }
    return levels

# ---------------------------
# ML Model
# ---------------------------

def prepare_features(df):
    features = df[['EMA10', 'EMA30', 'MACD', 'MACD_Signal', 'RSI', 'ret_1', 'ret_3', 'close_lag_1', 'close_lag_2', 'close_lag_3']].copy()
    features.dropna(inplace=True)
    return features

def prepare_labels(df):
    df = df.copy()
    df['Next_Close'] = df['Close'].shift(-1)
    df['Next_Open'] = df['Open'].shift(-1)
    df['Label'] = (df['Next_Close'] > df['Next_Open']).astype(int)
    df.dropna(inplace=True)
    return df['Label']

def align_features_labels(features, labels):
    common_idx = features.index.intersection(labels.index)
    return features.loc[common_idx], labels.loc[common_idx]

def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

def save_model(model, symbol):
    fname = f"model_{sanitize_symbol_for_filename(symbol)}.joblib"
    joblib.dump(model, fname)

def load_model(symbol):
    try:
        fname = f"model_{sanitize_symbol_for_filename(symbol)}.joblib"
        return joblib.load(fname)
    except:
        return None

# ---------------------------
# Prediction & Backtest
# ---------------------------

def predict_signal(model, features, threshold=0.7):
    if model is None or features.empty:
        return None
    last_feat = features.iloc[-1].values.reshape(1, -1)
    if not hasattr(model, "predict_proba"):
        pred = model.predict(last_feat)[0]
        return "BUY" if pred == 1 else "SELL"
    probs = model.predict_proba(last_feat)[0]  # [prob_down, prob_up]
    prob_up = probs[1]
    if prob_up >= threshold:
        return "BUY"
    elif prob_up <= (1 - threshold):
        return "SELL"
    else:
        return "HOLD"

def max_drawdown(returns):
    if len(returns) == 0:
        return 0
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return dd.min()

def backtest_strategy(df, signals, trade_fee=TRADE_FEE):
    df = df.copy()
    df['Signal'] = signals
    position = 0
    entry_price = 0
    returns = []

    for i in range(len(df)):
        signal = df['Signal'].iloc[i]
        close = df['Close'].iloc[i]

        if signal == "BUY" and position == 0:
            position = 1
            entry_price = close * (1 + trade_fee)
        elif signal == "SELL" and position == 1:
            exit_price = close * (1 - trade_fee)
            ret = (exit_price - entry_price) / entry_price
            returns.append(ret)
            position = 0

    if position == 1:
        exit_price = df['Close'].iloc[-1] * (1 - trade_fee)
        ret = (exit_price - entry_price) / entry_price
        returns.append(ret)

    returns = np.array(returns)
    total_return = np.prod(1 + returns) - 1 if len(returns) > 0 else 0
    annual_return = (1 + total_return) ** (252 / len(df)) - 1 if len(df) > 0 else 0
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
    sharpe = annual_return / volatility if volatility != 0 else 0
    max_dd = max_drawdown(returns)
    win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0

    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Trades": len(returns),
    }

# ---------------------------
# Alerts
# ---------------------------

def send_telegram(message):
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        # Do not spam errors; just return.
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        r = requests.post(url, data=data, timeout=5)
        if r.status_code != 200:
            st.error(f"Telegram error: {r.text}")
    except Exception as e:
        st.error(f"Telegram exception: {e}")

# ---------------------------
# Session / Monday / Prev Day / Week / Month Levels
# ---------------------------

def get_monday_high_low_open(df):
    monday_df = df[df['Datetime'].dt.dayofweek == 0]
    if monday_df.empty:
        return None, None, None
    high = monday_df['High'].max()
    low = monday_df['Low'].min()
    open_ = monday_df.iloc[0]['Open']
    return high, low, open_

def get_previous_day_high_low_open(df, current_datetime):
    prev_day = (current_datetime - pd.Timedelta(days=1)).date()
    # step back until a trading day is found
    while True:
        prev_day_dt = pd.Timestamp(prev_day)
        prev_day_df = df[df['Datetime'].dt.date == prev_day_dt.date()]
        if not prev_day_df.empty:
            high = prev_day_df['High'].max()
            low = prev_day_df['Low'].min()
            open_ = prev_day_df.iloc[0]['Open']
            return high, low, open_
        prev_day = (prev_day_dt - pd.Timedelta(days=1)).date()
        # safety break if loop would be infinite
        if (pd.Timestamp.now().date() - prev_day).days > 14:
            return None, None, None

def get_week_high_low_open(df, current_datetime):
    week_start = current_datetime - pd.Timedelta(days=current_datetime.weekday())
    week_df = df[(df['Datetime'].dt.date >= week_start.date()) & (df['Datetime'].dt.date <= current_datetime.date())]
    if week_df.empty:
        return None, None, None
    high = week_df['High'].max()
    low = week_df['Low'].min()
    open_ = week_df.iloc[0]['Open']
    return high, low, open_

def get_month_high_low_open(df, current_datetime):
    month_start = current_datetime.replace(day=1)
    month_df = df[(df['Datetime'].dt.date >= month_start.date()) & (df['Datetime'].dt.date <= current_datetime.date())]
    if month_df.empty:
        return None, None, None
    high = month_df['High'].max()
    low = month_df['Low'].min()
    open_ = month_df.iloc[0]['Open']
    return high, low, open_

def check_cross(close_series, level):
    if level is None or len(close_series) < 2:
        return False
    prev = close_series.iloc[-2]
    curr = close_series.iloc[-1]
    return (prev < level and curr >= level) or (prev > level and curr <= level)

# ---------------------------
# Plotting
# ---------------------------

def plot_chart(df, fib_levels, symbol, signals=None, show_levels=True, levels_dict=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Datetime'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA10'], mode='lines', name='EMA10'))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA30'], mode='lines', name='EMA30'))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(dash='dash')))
    if 'Support' in df.columns:
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Support'], mode='lines', name='Support', line=dict(dash='dot')))
    if 'Resistance' in df.columns:
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Resistance'], mode='lines', name='Resistance', line=dict(dash='dot')))

    # Fib lines
    for lvl_name, lvl_price in (fib_levels or {}).items():
        if lvl_price is None:
            continue
        fig.add_hline(y=lvl_price, line_dash="dot", annotation_text=lvl_name, annotation_position="right")

    # SMC markers
    bos_up = df[df['BOS_UP']]
    bos_down = df[df['BOS_DOWN']]
    choch = df[df['CHOCH']]
    if not bos_up.empty:
        fig.add_trace(go.Scatter(x=bos_up['Datetime'], y=bos_up['High'], mode='markers', name='BOS_UP', marker=dict(size=10, symbol='triangle-up')))
    if not bos_down.empty:
        fig.add_trace(go.Scatter(x=bos_down['Datetime'], y=bos_down['Low'], mode='markers', name='BOS_DOWN', marker=dict(size=10, symbol='triangle-down')))
    if not choch.empty:
        fig.add_trace(go.Scatter(x=choch['Datetime'], y=choch['Close'], mode='markers', name='CHOCH', marker=dict(size=10, symbol='star')))

    # Buy/Sell signals markers
    if signals is not None:
        signals_aligned = signals.reindex(df.index)
        buys = df[signals_aligned == "BUY"]
        sells = df[signals_aligned == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys['Datetime'], y=buys['Low'] * 0.995, mode='markers', name='Buy Signal', marker=dict(size=12, symbol='triangle-up')))
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells['Datetime'], y=sells['High'] * 1.005, mode='markers', name='Sell Signal', marker=dict(size=12, symbol='triangle-down')))

    # Draw session/levels horizontal lines
    if show_levels and levels_dict:
        color_map = {
            'Monday': 'purple', 'Prev Day': 'teal', 'Week': 'orange', 'Month': 'brown', 'Session': 'gray'
        }
        for label, price in levels_dict.items():
            if price is None:
                continue
            # choose color
            color = color_map.get(label.split()[0], 'black')
            fig.add_hline(y=price, line_dash="dash", annotation_text=label, annotation_position="right", line_color=color)

    fig.update_layout(title=f"{symbol} Price Chart with Indicators", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, height=700, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Streamlit App
# ---------------------------

st.set_page_config(page_title="Pro Multi-Asset AI Trading + Levels", layout="wide")
st.title("Pro Multi-Asset AI Trading & Backtesting with Session Levels")

# Sidebar
st.sidebar.header("Settings")
asset_class = st.sidebar.selectbox("Asset Class", list(SYMBOLS.keys()))
symbol = st.sidebar.selectbox("Symbol", SYMBOLS[asset_class])
timeframe = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()))
interval = TIMEFRAMES[timeframe]

# Session-level options
st.sidebar.subheader("Levels & Session Options")
show_monday_levels = st.sidebar.checkbox("Show Monday Levels", value=True)
show_prev_day_levels = st.sidebar.checkbox("Show Previous Day Levels", value=True)
show_week_month_levels = st.sidebar.checkbox("Show Week/Month Levels", value=True)
show_custom_session = st.sidebar.checkbox("Show Custom Session (by time range)", value=False)
session_start = st.sidebar.text_input("Session start (HH:MM)", value="09:15")
session_end = st.sidebar.text_input("Session end (HH:MM)", value="15:30")
session_days = st.sidebar.multiselect("Session Days (0=Mon..6=Sun)", default=[0,1,2,3,4], options=[0,1,2,3,4,5,6])

st.sidebar.subheader("Telegram Alerts")
enable_telegram = st.sidebar.checkbox("Enable Telegram Alerts", value=False)
alert_on_cross = st.sidebar.checkbox("Alert when price crosses any level (last bar)", value=True)

# Fetch data
with st.spinner(f"Fetching {symbol} data at {interval}..."):
    df = fetch_data(symbol, interval)
if df is None or df.empty:
    st.stop()

# Indicators & SMC
df = add_indicators(df)
df = detect_smc(df)
df = detect_support_resistance(df)
fib_levels = fibonacci_levels(df)

# Prepare ML data
features = prepare_features(df)
labels = prepare_labels(df)
features, labels = align_features_labels(features, labels)

# Load or train model
model = load_model(symbol)
col1, col2, col3, col4, col5 = st.columns(5)

if model is None:
    with col1:
        st.write("Model not found.")
    with col2:
        if st.button("Train AI Model"):
            with st.spinner("Training..."):
                model, acc = train_model(features, labels)
                save_model(model, symbol)
                st.success(f"Model trained with accuracy: {acc:.2%}")
else:
    with col1:
        st.write("Model loaded.")
    with col2:
        if st.button("Retrain AI Model"):
            with st.spinner("Retraining..."):
                model, acc = train_model(features, labels)
                save_model(model, symbol)
                st.success(f"Model retrained with accuracy: {acc:.2%}")

# Predict signal with 70% threshold
signal = predict_signal(model, features, threshold=0.7)
with col3:
    st.metric("AI Signal", signal if signal else "N/A")

# Backtest
signals_series = pd.Series(["BUY" if x == 1 else "SELL" for x in labels], index=labels.index)
metrics = backtest_strategy(df.loc[signals_series.index], signals_series)

with col4:
    st.metric("Total Return", f"{metrics['Total Return']*100:.2f}%")
with col5:
    st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")

# Compute session/week/month levels
last_dt = df['Datetime'].iloc[-1]
levels = {}

if show_monday_levels:
    mon_h, mon_l, mon_o = get_monday_high_low_open(df)
    levels.update({
        "Monday High": mon_h,
        "Monday Low": mon_l,
        "Monday Open": mon_o,
        "Monday Mid": (mon_h + mon_l) / 2 if mon_h is not None and mon_l is not None else None
    })

if show_prev_day_levels:
    prev_h, prev_l, prev_o = get_previous_day_high_low_open(df, last_dt)
    levels.update({
        "Prev Day High": prev_h,
        "Prev Day Low": prev_l,
        "Prev Day Open": prev_o,
        "Prev Day Mid": (prev_h + prev_l) / 2 if prev_h is not None and prev_l is not None else None
    })

if show_week_month_levels:
    wk_h, wk_l, wk_o = get_week_high_low_open(df, last_dt)
    mn_h, mn_l, mn_o = get_month_high_low_open(df, last_dt)
    levels.update({
        "Week High": wk_h,
        "Week Low": wk_l,
        "Week Open": wk_o,
        "Week Mid": (wk_h + wk_l) / 2 if wk_h is not None and wk_l is not None else None,
        "Month High": mn_h,
        "Month Low": mn_l,
        "Month Open": mn_o,
        "Month Mid": (mn_h + mn_l) / 2 if mn_h is not None and mn_l is not None else None
    })

if show_custom_session:
    # compute session high/low/open from provided time/days
    try:
        def get_session_high_low_open(df_local, session_start, session_end, days):
            df_local = df_local.copy()
            # ensure Datetime is tz-naive in calculations (works regardless)
            df_local['time'] = df_local['Datetime'].dt.time
            start_h, start_m = map(int, session_start.split(':'))
            end_h, end_m = map(int, session_end.split(':'))
            start_t = datetime.time(start_h, start_m)
            end_t = datetime.time(end_h, end_m)
            filtered = df_local[df_local['Datetime'].dt.dayofweek.isin(days)]
            if start_t <= end_t:
                session_df = filtered[(filtered['time'] >= start_t) & (filtered['time'] <= end_t)]
            else:
                session_df = filtered[(filtered['time'] >= start_t) | (filtered['time'] <= end_t)]
            if session_df.empty:
                return None, None, None
            high = session_df['High'].max()
            low = session_df['Low'].min()
            open_ = session_df.iloc[0]['Open']
            return high, low, open_

        sess_h, sess_l, sess_o = get_session_high_low_open(df, session_start, session_end, session_days)
        levels.update({
            "Session High": sess_h,
            "Session Low": sess_l,
            "Session Open": sess_o,
            "Session Mid": (sess_h + sess_l) / 2 if sess_h is not None and sess_l is not None else None
        })
    except Exception as e:
        st.error(f"Error computing session: {e}")

# Plot chart with levels and signals
plot_chart(df, fib_levels, symbol, signals=signals_series, show_levels=True, levels_dict=levels)

# Detect crosses on last bar
cross_messages = []
close_series = df['Close']
for label, price in levels.items():
    if price is None:
        continue
    if check_cross(close_series, price):
        msg = f"Price crossed {label} ({price:.4f}) on last bar."
        cross_messages.append(msg)

if cross_messages:
    st.subheader("Level Crosses (Latest Bar)")
    for m in cross_messages:
        st.info(m)
        if enable_telegram and alert_on_cross:
            send_telegram(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {symbol} - {m}")

# Telegram alert for AI signal (optional)
if signal and signal != "HOLD":
    st.success(f"AI Signal: {signal}")
    if enable_telegram:
        send_telegram(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {symbol} AI Signal: {signal}")

# Download backtest CSV
csv_df = df.loc[signals_series.index].copy()
csv_df['Signal'] = signals_series.values
csv_bytes = csv_df.to_csv(index=False).encode('utf-8')
st.download_button(label="Download Backtest CSV", data=csv_bytes, file_name=f"backtest_{sanitize_symbol_for_filename(symbol)}_{timeframe}.csv", mime="text/csv")
