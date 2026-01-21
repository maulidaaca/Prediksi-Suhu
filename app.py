import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import random

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Suhu", page_icon="🌡️")

# --- DEBUGGING PATH (Supaya ketahuan salahnya di mana) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(current_dir, "model")
model_path = os.path.join(model_folder, "model_suhu.onnx")
scaler_path = os.path.join(model_folder, "scaler_suhu.save")

st.write(f"📂 **Mencari file di:** `{model_path}`") # Info buat debug

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    # Cek apakah file benar-benar ada
    if not os.path.exists(model_path):
        st.error(f"❌ ERROR: File model tidak ditemukan di: {model_path}")
        st.info("Pastikan Anda sudah membuat folder 'model' dan memasukkan file 'model_suhu.onnx' ke dalamnya.")
        return None, None

    if not os.path.exists(scaler_path):
        st.error(f"❌ ERROR: File scaler tidak ditemukan di: {scaler_path}")
        return None, None

    try:
        scaler = joblib.load(scaler_path)
        ort_session = ort.InferenceSession(model_path)
        return scaler, ort_session
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

scaler, ort_session = load_resources()

# Jika gagal load, stop aplikasi
if scaler is None or ort_session is None:
    st.stop()

# --- FORM INPUT ---
st.title("🌡️ Prediksi Suhu Harian")
st.write("Masukkan suhu 7 hari terakhir:")

col1, col2 = st.columns(2)
inputs = []
with col1:
    d1 = st.number_input("H-7", value=30.0)
    d2 = st.number_input("H-6", value=31.0)
    d3 = st.number_input("H-5", value=29.0)
    d4 = st.number_input("H-4", value=33.0)
with col2:
    d5 = st.number_input("H-3", value=32.0)
    d6 = st.number_input("H-2", value=31.5)
    d7 = st.number_input("H-1 (Kemarin)", value=32.5)

inputs = [d1, d2, d3, d4, d5, d6, d7]

if st.button("Prediksi", type="primary"):
    try:
        data_np = np.array(inputs).reshape(-1, 1)
        data_scaled = scaler.transform(data_np)
        final_input = data_scaled.reshape(1, 7, 1).astype(np.float32)
        
        # Prediksi
        output_name = ort_session.get_outputs()[0].name
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run([output_name], {input_name: final_input})
        
        hasil = scaler.inverse_transform(outputs[0])[0][0]
        st.success(f"Prediksi Suhu Besok: **{hasil:.2f} °C**")
    except Exception as e:
        st.error(f"Error: {e}")

