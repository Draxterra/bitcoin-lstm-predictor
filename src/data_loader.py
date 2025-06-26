from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_btc_data(start="2019-01-01", end=None):
    """
    Mengambil data historis BTC/USD dari Yahoo Finance.
    """
    df = yf.download("BTC-USD", start=start, end=end)
    df.dropna(inplace=True)
    return df

def preprocess_data(df, feature='Close', window_size=60):
    """
    Melakukan normalisasi dan pembentukan sequence data untuk LSTM.

    Returns:
    - scaler: objek MinMaxScaler
    - X_train, X_val, y_train, y_val: data training dan validasi
    """
    data = df[[feature]].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    # Bagi data menjadi 80% training dan 20% validation tanpa shuffle
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    return scaler, X_train, X_val, y_train, y_val

def forecast_future(model, scaler, df, days_ahead=3, feature='Close', window_size=60):
    """
    Melakukan prediksi ke depan (forecast) sejumlah hari tertentu berdasarkan data terakhir.
    
    Args:
        model: model LSTM terlatih
        scaler: scaler MinMaxScaler yang sudah fit
        df: dataframe historis
        days_ahead: jumlah hari ke depan untuk diprediksi
        feature: kolom yang dijadikan target
        window_size: panjang sequence input

    Returns:
        DataFrame dengan tanggal dan harga prediksi
    """
    last_data = df[[feature]].values
    scaled = scaler.transform(last_data)
    
    result = []
    input_seq = scaled[-window_size:].copy()  # (60, 1)

    current_date = df.index[-1]

    for i in range(days_ahead):
        prediction = model.predict(input_seq[np.newaxis, :, :], verbose=0)
        result.append({
            'date': current_date + timedelta(days=i+1),
            'predicted': scaler.inverse_transform(prediction)[0][0]
        })
        # Tambahkan prediksi ke input_seq dan geser jendela
        input_seq = np.append(input_seq, prediction, axis=0)[-window_size:]

    return pd.DataFrame(result)


def get_current_btc_price():
    """
    Mengambil harga BTC real-time dari CoinGecko.
    """
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        response = requests.get(url)
        data = response.json()
        return float(data["bitcoin"]["usd"])
    except Exception as e:
        print("Gagal ambil harga real-time:", e)
        return None
