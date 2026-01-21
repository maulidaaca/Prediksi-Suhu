import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import altair as alt

# =====================
# Load Model & Scaler
# =====================
MODEL_PATH = 'model/model_suhu.h5'
SCALER_PATH = 'model/scaler_suhu.save'

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="Prediksi Suhu Besok",
    page_icon="🌡️",
    layout="wide"
)

# =====================
# Sidebar: Menu & Pelengkap
# =====================
st.sidebar.title("🌤️ Prediksi Suhu Besok")
menu = st.sidebar.radio("Pilih Menu:", ["Tentang Sistem", "Prediksi Suhu"])
st.sidebar.markdown("---")
st.sidebar.markdown("Email: ayudwinengtias@gmail.com")
st.sidebar.markdown("Teknologi: LSTM, Python, Streamlit")

# =====================
# Menu: Tentang Sistem
# =====================
if menu == "Tentang Sistem":
    st.title("Tentang Sistem Prediksi Suhu")
    st.markdown("Sistem ini membantu memprediksi suhu besok berdasarkan **data 7 hari terakhir** menggunakan **Deep Learning (LSTM)**.")

    # =======================
    # Bagian 1: Data Diri Perancang
    # =======================
    with st.container():
        st.subheader("👩‍💻 Data Diri Perancang Sistem")
        st.markdown("""
- **Nama:** Ayu Dwi Nengtias  
- **Prodi:** Teknik Informatika  
- **Universitas:** Universitas Muhammadiyah Riau  
- **Kontak:** ayudwinengtias@gmail.com
""")

    # =======================
    # Bagian 2: Deskripsi Sistem
    # =======================
    with st.container():
        st.subheader("📝 Deskripsi Sistem")
        st.info("""
Sistem ini adalah aplikasi prediksi suhu besok berbasis **Deep Learning (LSTM)**.  
Data yang digunakan: **7 hari terakhir** dengan fitur:  
- Suhu (°C)  
- Kelembaban (%)  
- Kecepatan Angin (m/s)  
- Tekanan (hPa)  

**Tujuan Sistem:**  
- Membantu memprediksi suhu besok berdasarkan tren data 7 hari terakhir.  
- Mempermudah perencanaan kegiatan harian yang tergantung cuaca.
""")

    st.markdown("---")

    # =======================
    # Bagian 3: Instruksi / Petunjuk
    # =======================
    with st.container():
        st.subheader("📖 Petunjuk / Instruksi Menggunakan Sistem")
        st.success("""
1. Pilih menu **Prediksi Suhu** di sidebar.  
2. Masukkan data 7 hari terakhir di tabel input.  
3. Klik tombol **Prediksi Suhu Besok**.  
4. Hasil prediksi dan grafik tren akan muncul di layar.
""")


# =====================
# Menu: Prediksi Suhu
# =====================
elif menu == "Prediksi Suhu":
    st.title("🌡️ Prediksi Suhu Besok")
    st.markdown("Masukkan data 7 hari terakhir di tabel berikut:")

    # ---- Tabel Input ----
    default_data = {
        "Suhu (°C)": [30.0]*7,
        "Kelembaban (%)": [70.0]*7,
        "Kecepatan Angin (m/s)": [3.0]*7,
        "Tekanan (hPa)": [1010.0]*7
    }
    df_input = pd.DataFrame(default_data, index=[f"Hari {i}" for i in range(1,8)])
    edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

    st.markdown("---")

    # ---- Layout Kolom: Tombol Prediksi & Hasil ----
    col1, col2 = st.columns([1,2])

    with col1:
        if st.button("✅ Prediksi Suhu Besok"):
            try:
                # Ambil data input
                sequence_np = edited_df.to_numpy().astype(float)
                sequence_scaled = scaler.transform(sequence_np)
                final_input = sequence_scaled.reshape(1, 7, 4)
                prediction_scaled = model.predict(final_input, verbose=0)

                # Inverse transform hanya suhu
                dummy = np.zeros((1, 4))
                dummy[0, 0] = prediction_scaled[0, 0]
                prediction_actual = scaler.inverse_transform(dummy)
                result = prediction_actual[0, 0]

                st.session_state['result'] = result  # Simpan hasil untuk kolom kedua

            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")

    with col2:
        if 'result' in st.session_state:
            # Card hasil prediksi
            st.success(f"🌞 Prediksi Suhu Besok: **{st.session_state['result']:.2f} °C**")
            st.info("Data 7 hari terakhir telah digunakan untuk prediksi.")

            # Grafik tren input suhu
            chart_suhu = alt.Chart(edited_df.reset_index()).mark_line(point=True, color='orange').encode(
                x='index',
                y='Suhu (°C)',
                tooltip=['Suhu (°C)', 'Kelembaban (%)', 'Kecepatan Angin (m/s)', 'Tekanan (hPa)']
            ).properties(title="📈 Tren Suhu 7 Hari Terakhir")
            st.altair_chart(chart_suhu, use_container_width=True)

            # Grafik tren kelembaban
            chart_hum = alt.Chart(edited_df.reset_index()).mark_line(point=True, color='blue').encode(
                x='index',
                y='Kelembaban (%)',
                tooltip=['Suhu (°C)', 'Kelembaban (%)', 'Kecepatan Angin (m/s)', 'Tekanan (hPa)']
            ).properties(title="💧 Tren Kelembaban 7 Hari Terakhir")
            st.altair_chart(chart_hum, use_container_width=True)
