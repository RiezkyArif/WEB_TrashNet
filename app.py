import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartWaste", layout="wide")

# --- KONFIGURASI MODEL DAN LABEL ---
MODEL_PATH = "model97.h5"
class_names = ['Organik', 'Anorganik']

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.write("Model input shape:", model.input_shape)
except FileNotFoundError as e:
    st.error(e)
    st.stop()

if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Prediction", "Confidence"])

# --- HALAMAN BERANDA ---
def page_home():
    st.title("🌿 Selamat Datang di SmartWaste")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("assets/1.png", caption="SmartWaste - Klasifikasi Sampah Organik & Anorganik", width=550)
    # Tulisan di pojok kiri
    st.markdown(
        "<h4 style='text-align: center; margin-center: 5px;'>Mulai klasifikasikan sampahmu sekarang!</h4>",
        unsafe_allow_html=True
    )
    st.markdown("""
    ### Fitur Utama
    - 📷 **Klasifikasi Sampah Otomatis**: Upload foto sampah dan dapatkan hasil klasifikasi langsung (organik/anorganik) dengan tingkat kepercayaan.
    - 📊 **Riwayat Prediksi**: Lihat riwayat hasil klasifikasi yang telah Anda lakukan dan Anda bisa menyimpan hasilnya berbentuk file.csv.
    - 📚 **Artikel Edukatif**: Pelajari perbedaan dan pengelolaan sampah organik & anorganik.
    """)

    st.markdown("""
    ### Cara Menggunakan SmartWaste
    1. Pilih menu **Klasifikasi Sampah** di sidebar.
    2. Upload foto sampah yang ingin diklasifikasikan.
    3. Lihat hasil prediksi dan edukasi terkait.
    """)

    st.markdown("""
    ### Manfaat Menggunakan SmartWaste
    - Membantu memilah sampah dengan mudah dan cepat
    - Mendukung lingkungan bersih dan sehat
    - Menambah wawasan tentang pengelolaan sampah
    """)

# --- HALAMAN KLASIFIKASI ---
def page_classification():
    st.title("📸 Unggah gambar sampah yang ingin Anda klasifikasikan")
    st.info("Pastikan Anda hanya mengupload gambar sampah (organik/anorganik) sesuai dataset. Jangan upload foto wajah atau gambar lain.")
    st.write("Anda dapat mengunggah banyak gambar sekaligus, lalu memilih file mana yang ingin diproses.")

    uploaded_files = st.file_uploader(
        "Pilih satu atau beberapa gambar...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.history = pd.DataFrame(columns=["Time", "Prediction", "Confidence"])
        file_names = [f.name for f in uploaded_files]
        selected_files = st.multiselect(
            "Pilih file yang ingin diprediksi:",
            options=file_names,
            default=file_names  # default: semua terpilih
        )
        for uploaded_file in uploaded_files:
            if uploaded_file.name in selected_files:
                image = Image.open(uploaded_file).convert("RGB")
                # Tampilkan gambar di tengah hanya saat proses
                with st.spinner('🔄 Memproses gambar...'):
                    col1, col2, col3 = st.columns([1,2,1])
                    with col2:
                        st.image(image, caption=f"Gambar: {uploaded_file.name}", width=400)
                    img = image.resize((50, 50))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    try:
                        prediction = model.predict(img_array)
                        predicted_label = class_names[np.argmax(prediction)]
                        confidence = np.max(prediction) * 100
                        confidence_threshold = 60  # threshold confidence (%)
                        if confidence < confidence_threshold:
                            st.warning("Gambar ini kemungkinan besar bukan gambar sampah (organik/anorganik). Silakan upload gambar sampah yang sesuai dataset.")
                        else:
                            st.success(f"Hasil: **{predicted_label}** ({confidence:.2f}%)")
                            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                            new_entry = pd.DataFrame([[current_time, predicted_label, confidence]], columns=["Time", "Prediction", "Confidence"])
                            st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)
                    except Exception as e:
                        st.error(f"Terjadi error saat prediksi: {e}")
                # Setelah prediksi, gambar tidak ditampilkan lagi (otomatis karena di luar with st.spinner)

    st.subheader("🔍 Riwayat Prediksi")
    st.dataframe(st.session_state.history)

# --- HALAMAN ARTIKEL (SESUAI KODE KAMU) ---
def page_articles():
    st.title("📰 Edukasi Sampah")
    # Coba tampilkan gambar edukasi, jika gagal tampilkan pesan error ramah
    col1, col2, col3 = st.columns([1,2,1])
    try:
        with col2:
            st.image("assets/2.png", caption="Infografis Sampah Organik & Anorganik", width=500)
    except Exception as e:
        with col2:
            st.error("Gambar edukasi tidak dapat ditampilkan. Pastikan file 'assets/2.png' ada dan tidak rusak.")
    st.markdown("""
    Memilah sampah adalah langkah sederhana namun berdampak besar untuk lingkungan. Dengan memilah sampah organik dan anorganik, kita membantu mengurangi pencemaran dan mendukung daur ulang.
    """)
    st.header("Apa itu Sampah Organik?")
    st.write("""
    Sampah organik adalah sampah yang mudah terurai secara alami, seperti sisa makanan, daun, dan kulit buah. Sampah ini dapat diolah menjadi kompos yang bermanfaat untuk tanaman dan lingkungan.
    """)

    st.header("Apa itu Sampah Anorganik?")
    st.write("""
    Sampah anorganik adalah sampah yang sulit terurai, seperti plastik, kaleng, kaca, dan logam. Sampah ini sebaiknya dipisahkan dan didaur ulang agar tidak mencemari lingkungan.
    """)

    st.header("Kenapa Harus Memilah Sampah?")
    st.write("""
    - Mengurangi pencemaran lingkungan 🌱
    - Mendukung proses daur ulang ♻️
    - Menjaga kesehatan masyarakat 🏥
    - Menghemat biaya pengelolaan sampah
    """)

    st.header("Tips Memilah Sampah di Rumah")
    st.write("""
    1. Sediakan dua tempat sampah: organik & anorganik
    2. Buang sisa makanan ke tempat sampah organik
    3. Cuci bersih sampah anorganik sebelum dibuang
    4. Dukung program bank sampah di lingkungan Anda
    """)


# --- NAVIGASI UTAMA STREAMLIT ---
st.sidebar.title("🌱 SmartWaste Menu")
st.sidebar.markdown("**Navigasi Aplikasi**")
page = st.sidebar.radio(
    "",
    [
        "🏠 Beranda",
        "🗑️ Klasifikasi Sampah",
        "📚 Edukasi Sampah"
    ]
)

if page == "🏠 Beranda":
    page_home()
elif page == "🗑️ Klasifikasi Sampah":
    page_classification()
elif page == "📚 Edukasi Sampah":
    page_articles()