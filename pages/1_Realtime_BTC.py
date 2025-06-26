# pages/1_Realtime_BTC.py

import streamlit as st

from src.data_loader import get_current_btc_price

st.set_page_config(page_title="BTC Realtime - CoinGecko", layout="centered")
st.title("💰 Harga Real-Time BTC/USD (CoinGecko)")

st.sidebar.page_link("app.py", label="⬅️ Kembali ke Data Historis")

price = get_current_btc_price()

if price:
    st.metric("Harga Bitcoin Saat Ini", f"${price:,.2f}")
else:
    st.error("❌ Gagal memuat harga dari CoinGecko.")
