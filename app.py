import streamlit as st
import numpy as np
import joblib
import onnxruntime as ort
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Suhu Harian", page_icon="🌡️")
st.title("🌡️ Prediksi Suhu Harian (LSTM)")
st.caption("Menggunakan Model ONNX untuk performa ringan")

# --- FUNGSI LOAD MODEL & SCALER ---
@st.cache_resource
def load_resources():
    # Cek lokasi file (apakah di folder 'model/' atau di root)
    if os.path.exists("model/scaler_suhu.save"):
        scaler_path = "model/scaler_suhu.save"
        model_path = "model/model_suhu.onnx"
    else:
        # Fallback jika file sejajar dengan app.py (untuk GitHub/Streamlit Cloud)
        scaler_path = "scaler_suhu.save"
        model_path = "model_suhu.onnx"

    try:
        scaler = joblib.load(scaler_path)
        ort_session = ort.InferenceSession(model_path)
        return scaler, ort_session
    except Exception as e:
        st.error(f"Gagal memuat file model. Error: {e}")
        return None, None

# Load Resources
scaler, ort_session = load_resources()

# Hentikan aplikasi jika model gagal dimuat
if scaler is None or ort_session is None:
    st.warning("Mohon pastikan file 'model_suhu.onnx' dan 'scaler_suhu.save' sudah ada.")
    st.stop()

# Dapatkan nama input & output layer ONNX otomatis
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# --- FORM INPUT SUHU (7 HARI) ---
st.write("Masukkan rata-rata suhu **7 hari terakhir**:")

col1, col2 = st.columns(2)
inputs = []

with col1:
    d1 = st.number_input("Hari ke-1 (7 hari lalu)", value=30.0, step=0.1)
    d2 = st.number_input("Hari ke-2 (6 hari lalu)", value=31.2, step=0.1)
    d3 = st.number_input("Hari ke-3 (5 hari lalu)", value=29.8, step=0.1)
    d4 = st.number_input("Hari ke-4 (4 hari lalu)", value=33.1, step=0.1)

with col2:
    d5 = st.number_input("Hari ke-5 (3 hari lalu)", value=32.0, step=0.1)
    d6 = st.number_input("Hari ke-6 (H-2)", value=31.5, step=0.1)
    d7 = st.number_input("Hari ke-7 (Kemarin)", value=32.5, step=0.1)

inputs = [d1, d2, d3, d4, d5, d6, d7]

# --- TOMBOL PREDIKSI ---
if st.button("Prediksi Suhu Besok", type="primary"):
    try:
        # 1. Preprocessing Data
        data_np = np.array(inputs).reshape(-1, 1) # Ubah ke array numpy
        data_scaled = scaler.transform(data_np)   # Normalisasi (0-1)
        
        # 2. Reshape untuk ONNX (Batch=1, TimeSteps=7, Features=1)
        # PENTING: Tipe data harus float32
        final_input = data_scaled.reshape(1, 7, 1).astype(np.float32)
        
        # 3. PREDIKSI (Pakai ort_session, BUKAN model.predict)
        outputs = ort_session.run([output_name], {input_name: final_input})
        prediction_scaled = outputs[0]
        
        # 4. Inverse Transform (Kembalikan ke suhu asli)
        hasil_prediksi = scaler.inverse_transform(prediction_scaled)[0][0]
        
        # 5. Tampilkan Hasil
        st.success(f"Prediksi Suhu Besok: **{hasil_prediksi:.2f} °C**")
        
        # Opsional: Grafik Tren
        st.subheader("Grafik Suhu Input")
        st.line_chart(inputs)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
