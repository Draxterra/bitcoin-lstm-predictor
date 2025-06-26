from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ===============================
# Konfigurasi halaman
# ===============================
st.set_page_config(page_title="Backtest Simulasi", layout="centered")
st.title("\U0001F504 Backtest Simulasi Trading Bitcoin")

if "backtest_ready" not in st.session_state or "model" not in st.session_state:
    st.warning("Silakan jalankan prediksi terlebih dahulu di halaman utama.")
    st.stop()

model = st.session_state.model
X_val = st.session_state.X_val
y_val = st.session_state.y_val
future_df = st.session_state.future_df

st.subheader("\U0001F4B5 Simulasi Backtest")
initial_balance = st.number_input("Saldo Awal (USD)", value=1000.0, step=100.0)
threshold_buy = st.slider("Threshold Buy (%)", min_value=0.0, max_value=5.0, value=1.0)
threshold_sell = st.slider("Threshold Sell (%)", min_value=0.0, max_value=5.0, value=1.0)

start_backtest = st.button("\U0001F3C3 Mulai Simulasi")

if start_backtest:
    st.info("Simulasi sedang diproses...")

    balance = initial_balance
    btc_hold = 0
    log = []

    pred_prices = model.predict(X_val).flatten()
    real_prices = y_val.flatten()

    for i in range(1, len(real_prices)):
        yesterday_price = real_prices[i - 1]
        today_price = real_prices[i]
        predicted_price = pred_prices[i]

        price_change = (predicted_price - yesterday_price) / yesterday_price * 100

        if price_change > threshold_buy and balance > 0:
            # Buy BTC
            btc_bought = balance / today_price
            btc_hold += btc_bought
            log.append(f"Hari {i}: BUY {btc_bought:.4f} BTC @ ${today_price:.2f}")
            balance = 0

        elif price_change < -threshold_sell and btc_hold > 0:
            # Sell BTC
            balance += btc_hold * today_price
            log.append(f"Hari {i}: SELL {btc_hold:.4f} BTC @ ${today_price:.2f}")
            btc_hold = 0

    # Nilai akhir portofolio
    final_balance = balance + (btc_hold * real_prices[-1])
    profit = final_balance - initial_balance
    profit_pct = (profit / initial_balance) * 100

    st.success("\U0001F4C8 Simulasi Selesai")
    st.write(f"**Saldo Akhir:** ${final_balance:,.2f}")
    st.write(f"**Profit/Loss:** ${profit:,.2f} ({profit_pct:.2f}%)")

    with st.expander("\U0001F4DD Detail Transaksi"):
        for entry in log:
            st.write(entry)
