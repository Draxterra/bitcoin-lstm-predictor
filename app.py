from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential

from src.data_loader import load_btc_data, preprocess_data
from src.forecasting import evaluate_model, train_model
from src.tuning import tune_lstm
from src.utils import split_dataset

# ==========================
# Konfigurasi halaman
# ==========================
st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="centered")
st.title("\U0001F4C8 Prediksi Harga Bitcoin (LSTM)")

# ==========================
# Sidebar
# ==========================
st.sidebar.header("\U0001F527 Opsi")
st.sidebar.page_link("pages/1_Realtime_BTC.py", label="\U0001F4B0 Lihat Harga Real-Time")
st.sidebar.page_link("pages/2_Backtest_Simulation.py", label="\U0001F504 Backtest Simulasi")


start_date = st.sidebar.date_input("Tanggal Mulai", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Tanggal Akhir", value=pd.to_datetime("today"))

if start_date >= end_date:
    st.sidebar.error("❌ Tanggal mulai harus sebelum tanggal akhir.")
    st.stop()

# ==========================
# Load Data Historis
# ==========================
st.subheader("\U0001F4CA Data Historis BTC/USD")
df = load_btc_data(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

if df.empty:
    st.error("❌ Gagal memuat data historis.")
    st.stop()

st.success(f"✅ Data berhasil dimuat. Total: {len(df)} data")
st.markdown(f"Data terakhir: **{df.index[-1].strftime('%Y-%m-%d')}**")
st.dataframe(df.tail(100), use_container_width=True)
st.line_chart(df["Close"])

# ==========================
# Tombol Forecast
# ==========================
st.subheader("\U0001F9E0 Prediksi Harga")
forecast_trigger = st.button("\U0001F52E Lakukan Forecast")
n_days = 3  # default untuk prediksi ke depan

if forecast_trigger:
    with st.spinner("\U0001F50D Preprocessing data..."):
        scaler, X_train, X_val, y_train, y_val = preprocess_data(df, feature="Close", window_size=60)

    st.subheader("⚙️ Tuning Hyperparameter")
    with st.spinner("🚀 Menjalankan tuning hyperparameter (Optuna)..."):
        best_params = tune_lstm(X_train, y_train, n_trials=10)
    st.success("Tuning selesai \U0001F3AF")
    st.write("Best Hyperparameters:")
    st.json(best_params)

    def build_best_model(input_shape, params):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=params['units_1'], return_sequences=True, activation="relu"), input_shape=input_shape))
        model.add(Dropout(params['dropout_1']))
        model.add(LSTM(units=params['units_2'], activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    st.subheader("\U0001F4DA Training Model")
    model = build_best_model((X_train.shape[1], X_train.shape[2]), best_params)
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluasi
    st.subheader("\U0001F4CA Evaluasi Model")
    metrics = evaluate_model(model, X_val, y_val)
    for k, v in metrics.items():
        st.write(f"**{k}**: {v:.4f}")

    # Prediksi masa depan
    last_sequence = X_val[-1]
    future_preds = []
    for _ in range(n_days):
        pred_input = last_sequence.reshape(1, 60, 1)
        next_pred = model.predict(pred_input)[0][0]
        future_preds.append(next_pred)
        last_sequence = np.append(last_sequence[1:], [[next_pred]], axis=0)

    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=n_days)
    future_df = pd.DataFrame({"Forecasted Price": future_prices.flatten()}, index=future_dates)
    st.session_state.future_df = future_df
    st.session_state.backtest_ready = True  # Tandai bahwa model sudah siap untuk backtest

# ==========================
# Tampilkan Prediksi jika tersedia
# ==========================
if "future_df" in st.session_state:
    st.subheader("\U0001F4C5 Prediksi Tanggal Mendatang")
    future_df = st.session_state.future_df
    for i in range(1, n_days + 1):
        tanggal = future_df.index[i - 1].strftime('%Y-%m-%d')
        harga = future_df.iloc[i - 1]["Forecasted Price"]
        st.write(f"**H+{i} ({tanggal})**: ${harga:,.2f}")

    st.line_chart(future_df)

    st.session_state.model = model
    st.session_state.X_val = X_val
    st.session_state.y_val = y_val
