import streamlit as st
import pickle
import numpy as np

# Load model, scaler, dan label encoder
@st.cache_resource
def load_model():
    with open("models/sleep_quality_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("models/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open("models/label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)

    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()

# Streamlit UI
st.title("Prediksi Kualitas Tidur")

# Input durasi tidur: jam dan menit
col1, col2 = st.columns(2)
with col1:
    jam = st.number_input("Jam", min_value=0, max_value=12, step=1)
with col2:
    menit = st.number_input("Menit", min_value=0, max_value=59, step=1)

# Konversi ke format jam desimal
sleep_duration = jam + (menit / 60)

if st.button("Prediksi"):
    if sleep_duration > 0:
        # Preprocessing input
        input_data = np.array([[sleep_duration]])
        input_scaled = scaler.transform(input_data)

        # Prediksi
        prediction = model.predict(input_scaled)
        category = label_encoder.inverse_transform(prediction)[0]

        # Tampilkan hasil prediksi
        st.success(f"Kualitas Tidur: {category}")

        # Berikan saran berdasarkan hasil prediksi
        if category == "Baik":
            st.info("Jam tidur Anda sudah optimal! Pertahankan kebiasaan tidur yang baik. 😊")
        elif category == "Cukup":
            st.warning("Tidur Anda cukup, tetapi bisa lebih baik. Cobalah tidur lebih awal untuk hasil yang optimal.")
        else:
            st.error("Jam tidur Anda kurang! Usahakan tidur lebih lama untuk meningkatkan kualitas kesehatan. 😴")
    else:
        st.warning("Masukkan durasi tidur yang valid!")

import streamlit as st

st.title("Prediksi Minum Harian")

# Input jumlah gelas yang diminum hari ini
glasses = st.number_input("Masukkan jumlah gelas yang sudah kamu minum hari ini", min_value=0, step=1)

if st.button("Cek Target Minum"):
    if glasses < 6:
        st.error("⚠ Kamu belum cukup minum hari ini! Tambahkan lagi ya 💦")
    elif 6 <= glasses < 8:
        st.warning("😐 Minummu sudah cukup, tapi masih bisa ditingkatkan untuk hasil yang optimal.")
    else:
        st.success("✅ Kamu sudah memenuhi target minum harian kamu! Good job! 💧")
        
st.title("Prediksi Olahraga Harian")

# Input durasi olahraga (menit)
duration = st.number_input("Masukkan durasi olahraga kamu hari ini (dalam menit)", min_value=0, step=1)

if st.button("Cek Prediksi Olahraga"):
    if duration < 20:
        st.error("❌ Durasi olahraga kamu belum cukup. Ayo tambah lagi agar tubuh tetap sehat! 🏋️‍♂️")
    elif 20 <= duration < 30:
        st.warning("⚠️ Olahraga kamu cukup, tapi bisa lebih baik dengan sedikit tambahan waktu.")
    elif duration > 30:
        st.success("✅ Durasi olahraga kamu baik! Pertahankan rutin ini, teruskan semangatnya! 💪🔥")