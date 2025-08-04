import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import gdown
import os

# --- KONFIGURASI MODEL DAN LABEL ---
MODEL_PATH = "model97.h5"
class_names = ['Organik', 'Anorganik']

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?1gT0XYZabCyzD4B_JgKfMb7P-vn"  # GANTI dengan ID file Drive asli kamu
    gdown.download(url, MODEL_PATH, quiet=False)

st.set_page_config(page_title="SmartWaste", layout="wide")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError as e:
    st.error(e)
    st.stop()

# Fungsi untuk validasi gambar sampah
def validate_waste_image(image, debug_mode=False):
    """
    Validasi yang lebih longgar untuk memastikan gambar adalah sampah
    """
    # Konversi ke array
    img_array = np.array(image)
    
    # Debug info (hanya jika debug mode aktif)
    if debug_mode:
        st.write(f"🔍 Debug: Ukuran gambar = {img_array.shape}")
        st.write(f"🔍 Debug: Rata-rata kecerahan = {np.mean(np.mean(img_array, axis=2)):.2f}")
        st.write(f"🔍 Debug: Standar deviasi warna = {np.std(img_array):.2f}")
    
    # Cek ukuran gambar (lebih longgar)
    if img_array.shape[0] < 50 or img_array.shape[1] < 50:
        return False, "Gambar terlalu kecil. Upload gambar dengan resolusi yang lebih tinggi (minimal 50x50 pixel)."
    
    # Cek apakah gambar terlalu terang atau terlalu gelap (lebih longgar)
    gray = np.mean(img_array, axis=2)
    if np.mean(gray) < 20 or np.mean(gray) > 240:
        return False, "Gambar terlalu terang atau terlalu gelap. Pastikan gambar sampah terlihat jelas."
    
    # Cek variasi warna (lebih longgar)
    color_std = np.std(img_array)
    if color_std < 15:
        return False, "Gambar terlalu monoton. Pastikan gambar menunjukkan sampah yang jelas."
    
    # Cek apakah gambar memiliki terlalu banyak garis lurus (lebih longgar)
    gray_img = np.mean(img_array, axis=2)
    
    # Deteksi garis horizontal dan vertikal
    horizontal_lines = np.sum(np.abs(np.diff(gray_img, axis=1)) > 40)
    vertical_lines = np.sum(np.abs(np.diff(gray_img, axis=0)) > 40)
    
    # Jika terlalu banyak garis, kemungkinan dokumen/tabel (lebih longgar)
    if horizontal_lines > img_array.shape[0] * 0.5 or vertical_lines > img_array.shape[1] * 0.5:
        return False, "Gambar terdeteksi sebagai dokumen/tabel. Hanya upload gambar sampah organik atau anorganik."
    
    # Cek apakah gambar memiliki terlalu banyak teks (lebih longgar)
    contrast_areas = np.sum(np.std(gray_img, axis=1) > 60)
    if contrast_areas > img_array.shape[0] * 0.6:
        return False, "Gambar terdeteksi mengandung teks/dokumen. Hanya upload gambar sampah."
    
    # Cek rasio aspek (lebih longgar)
    aspect_ratio = img_array.shape[1] / img_array.shape[0]
    if aspect_ratio > 5 or aspect_ratio < 0.2:
        return False, "Rasio aspek gambar tidak wajar. Pastikan gambar sampah tidak terlalu panjang atau lebar."
    
    return True, "Gambar valid"

# Fungsi tambahan untuk deteksi gambar yang bukan sampah
def is_likely_not_waste(image):
    """
    Deteksi tambahan untuk gambar yang kemungkinan bukan sampah
    """
    img_array = np.array(image)
    gray = np.mean(img_array, axis=2)
    
    # Deteksi area putih yang besar (kemungkinan dokumen)
    white_areas = np.sum(gray > 200)
    if white_areas > img_array.shape[0] * img_array.shape[1] * 0.7:
        return True, "Gambar terlalu banyak area putih. Kemungkinan dokumen atau kertas."
    
    # Deteksi pola grid (kemungkinan tabel)
    # Cek apakah ada pola garis yang teratur
    h_edges = np.abs(np.diff(gray, axis=1))
    v_edges = np.abs(np.diff(gray, axis=0))
    
    # Hitung garis horizontal dan vertikal yang kuat
    strong_h_lines = np.sum(h_edges > 40)
    strong_v_lines = np.sum(v_edges > 40)
    
    if strong_h_lines > img_array.shape[0] * 0.5 or strong_v_lines > img_array.shape[1] * 0.5:
        return True, "Gambar terdeteksi memiliki pola grid/tabel. Hanya upload gambar sampah."
    
    return False, ""

# Fungsi untuk deteksi wajah sederhana
def detect_face_simple(image):
    """
    Deteksi wajah sederhana berdasarkan karakteristik wajah
    """
    img_array = np.array(image)
    gray = np.mean(img_array, axis=2)
    
    # Deteksi area kulit (warna kulit manusia) - menggunakan RGB
    # Warna kulit biasanya memiliki R > G > B
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    
    # Deteksi area dengan karakteristik warna kulit
    skin_mask = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (abs(r - g) > 15)
    skin_ratio = np.sum(skin_mask) / (img_array.shape[0] * img_array.shape[1])
    
    # Jika terlalu banyak area kulit, kemungkinan wajah
    if skin_ratio > 0.25:
        return True, "Gambar terdeteksi mengandung wajah/orang. Hanya upload gambar sampah."
    
    # Deteksi area gelap yang bisa jadi rambut
    dark_areas = np.sum(gray < 80)
    dark_ratio = dark_areas / (img_array.shape[0] * img_array.shape[1])
    
    # Kombinasi area kulit dan area gelap yang tinggi
    if skin_ratio > 0.1 and dark_ratio > 0.15:
        return True, "Gambar terdeteksi mengandung wajah/orang. Hanya upload gambar sampah."
    
    # Deteksi pola simetris yang bisa jadi wajah
    # Wajah biasanya memiliki pola simetris
    height, width = gray.shape
    mid_width = width // 2
    
    # Bandingkan sisi kiri dan kanan untuk simetri
    if mid_width > 0:
        left_side = gray[:, :mid_width]
        right_side = gray[:, mid_width:2*mid_width]
        
        if right_side.shape[1] == left_side.shape[1] and left_side.shape[1] > 0:
            symmetry_diff = np.mean(np.abs(left_side - np.fliplr(right_side)))
            if symmetry_diff < 25 and skin_ratio > 0.05:
                return True, "Gambar terdeteksi memiliki pola simetris seperti wajah. Hanya upload gambar sampah."
    
    # Deteksi area bulat/oval yang bisa jadi kepala
    # Hitung gradient untuk deteksi tepi
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    
    # Deteksi area dengan gradient melingkar (kemungkinan kepala)
    if grad_x.shape[1] > 0 and grad_y.shape[0] > 0:
        # Pastikan dimensi kompatibel
        min_height = min(grad_x.shape[0], grad_y.shape[0])
        min_width = min(grad_x.shape[1], grad_y.shape[1])
        
        if min_height > 0 and min_width > 0:
            # Ambil bagian yang kompatibel dari kedua gradient
            grad_x_compat = grad_x[:min_height, :min_width]
            grad_y_compat = grad_y[:min_height, :min_width]
            
            circular_gradient = np.sqrt(grad_x_compat**2 + grad_y_compat**2)
            high_gradient_areas = np.sum(circular_gradient > 30)
            
            if high_gradient_areas > (img_array.shape[0] * img_array.shape[1] * 0.08) and skin_ratio > 0.05:
                return True, "Gambar terdeteksi mengandung bentuk kepala/wajah. Hanya upload gambar sampah."
    
    return False, ""

# Fungsi untuk deteksi foto/gambar yang bukan sampah
def detect_non_waste_image(image):
    """
    Deteksi komprehensif untuk gambar yang bukan sampah
    """
    img_array = np.array(image)
    
    # Deteksi wajah
    is_face, face_message = detect_face_simple(image)
    if is_face:
        return True, face_message
    
    # Deteksi dokumen/tabel
    is_not_waste, not_waste_message = is_likely_not_waste(image)
    if is_not_waste:
        return True, not_waste_message
    
    # Deteksi screenshot atau interface
    gray = np.mean(img_array, axis=2)
    
    # Deteksi area dengan warna yang sangat terang (kemungkinan UI/screenshot)
    bright_areas = np.sum(gray > 240)
    if bright_areas > img_array.shape[0] * img_array.shape[1] * 0.5:
        return True, "Gambar terdeteksi sebagai screenshot atau interface. Hanya upload gambar sampah."
    
    # Deteksi area dengan warna yang sangat gelap (kemungkinan foto gelap)
    dark_areas = np.sum(gray < 50)
    if dark_areas > img_array.shape[0] * img_array.shape[1] * 0.6:
        return True, "Gambar terlalu gelap. Pastikan gambar sampah terlihat jelas."
    
    # Deteksi gambar dengan terlalu banyak warna (kemungkinan foto atau seni)
    color_variance = np.std(img_array)
    if color_variance > 80:
        # Cek apakah ini foto yang terlalu berwarna
        bright_colors = np.sum(np.std(img_array, axis=2) > 40)
        if bright_colors > img_array.shape[0] * img_array.shape[1] * 0.3:
            return True, "Gambar terdeteksi sebagai foto berwarna. Hanya upload gambar sampah."
    
    return False, ""

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
    st.info("⚠️ **PENTING**: Hanya upload gambar sampah organik atau anorganik yang jelas. Jangan upload foto wajah, dokumen, tabel, atau gambar lain yang bukan sampah.")
    st.write("Anda dapat mengunggah banyak gambar sekaligus, lalu memilih file mana yang ingin diproses.")
    
    # Debug mode toggle
    debug_mode = st.checkbox("🔧 Debug Mode (Tampilkan info detail)")
    bypass_validation = st.checkbox("🚀 Bypass Validasi (Untuk testing dataset)")
    
    st.markdown("""
    ### 🚫 **Yang TIDAK Diperbolehkan:**
    - 📄 Dokumen, tabel, atau kertas
    - 👤 Foto wajah, manusia, atau selfie
    - 🖼️ Gambar abstrak atau seni
    - 📱 Screenshot aplikasi atau interface
    - 🖥️ Interface komputer atau menu
    - 🎨 Foto berwarna yang bukan sampah
    
    ### ✅ **Yang Diperbolehkan:**
    - 🍎 Sampah organik: sisa makanan, daun, kulit buah
    - 🥤 Sampah anorganik: plastik, kaleng, botol, kardus
    - 🗑️ Sampah dalam kondisi normal (tidak terlalu terang/gelap)
    """)

    uploaded_files = st.file_uploader(
        "Pilih satu atau beberapa gambar sampah...",
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
                
                # Validasi gambar sampah (skip jika bypass mode aktif)
                if not bypass_validation:
                    is_valid, validation_message = validate_waste_image(image, debug_mode)
                    
                    if not is_valid:
                        st.error(f"❌ **{uploaded_file.name}**: {validation_message}")
                        if debug_mode:
                            st.image(image, caption=f"Gambar ditolak: {uploaded_file.name}", width=300)
                        continue
                    
                    # Deteksi tambahan untuk gambar yang bukan sampah
                    is_not_waste, not_waste_message = detect_non_waste_image(image)
                    if is_not_waste:
                        st.error(f"❌ **{uploaded_file.name}**: {not_waste_message}")
                        if debug_mode:
                            st.image(image, caption=f"Gambar ditolak: {uploaded_file.name}", width=300)
                        continue
                else:
                    st.info(f"🚀 **{uploaded_file.name}**: Validasi dilewati (Debug Mode)")
                
                # Tampilkan gambar di tengah hanya saat proses
                with st.spinner('🔄 Memproses gambar...'):
                    col1, col2, col3 = st.columns([1,2,1])
                    with col2:
                        st.image(image, caption=f"Gambar: {uploaded_file.name}", width=400)
                    
                    img = image.resize((50, 50))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    if debug_mode:
                        st.write(f"🔍 Debug: Input shape = {img_array.shape}")
                        st.write(f"🔍 Debug: Input range = {np.min(img_array):.3f} - {np.max(img_array):.3f}")
                    
                    try:
                        prediction = model.predict(img_array, verbose=0)  # Hilangkan output verbose
                        predicted_label = class_names[np.argmax(prediction)]
                        confidence = np.max(prediction) * 100
                        confidence_threshold = 60  # Turunkan threshold confidence (%)
                        
                        if debug_mode:
                            st.write(f"🔍 Debug: Raw prediction = {prediction}")
                            st.write(f"🔍 Debug: Predicted label = {predicted_label}")
                            st.write(f"🔍 Debug: Confidence = {confidence:.2f}%")
                        
                        if confidence < confidence_threshold:
                            st.warning(f"⚠️ **{uploaded_file.name}**: Tingkat kepercayaan rendah ({confidence:.2f}%). Kemungkinan gambar bukan sampah yang sesuai. Silakan upload gambar sampah yang lebih jelas.")
                        else:
                            # Validasi tambahan berdasarkan hasil prediksi (lebih longgar)
                            gray_img = np.mean(np.array(image), axis=2)
                            white_ratio = np.sum(gray_img > 200) / (gray_img.shape[0] * gray_img.shape[1])
                            
                            if white_ratio > 0.8:  # Lebih longgar
                                st.error(f"❌ **{uploaded_file.name}**: Gambar terdeteksi sebagai dokumen/kertas meskipun confidence tinggi. Hanya upload gambar sampah.")
                            else:
                                st.success(f"✅ **{uploaded_file.name}**: **{predicted_label}** ({confidence:.2f}%)")
                                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                                new_entry = pd.DataFrame([[current_time, predicted_label, confidence]], columns=["Time", "Prediction", "Confidence"])
                                st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)
                    except Exception as e:
                        st.error(f"❌ **{uploaded_file.name}**: Terjadi error saat prediksi: {e}")
                        if debug_mode:
                            st.write(f"🔍 Debug: Error details = {str(e)}")

    st.subheader("🔍 Riwayat Prediksi")
    if not st.session_state.history.empty:
        st.dataframe(st.session_state.history)
        
        # Tombol untuk download hasil
        csv = st.session_state.history.to_csv(index=False)
        st.download_button(
            label="📥 Download Hasil Prediksi (CSV)",
            data=csv,
            file_name=f"smartwaste_predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Belum ada riwayat prediksi. Upload gambar sampah untuk melihat hasilnya di sini.")

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
# Tengahkan judul menu
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    st.sidebar.markdown("""
    <h2 style='text-align: center; color: #2E8B57; margin-bottom: 20px;'>🌱 Menu Aplikasi</h2>
    """, unsafe_allow_html=True)

# Tambahkan logo di sidebar
def create_logo():
    """Membuat logo sederhana untuk SmartWaste"""
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    
    # Background transparan
    ax.set_facecolor('none')
    
    # Botol anorganik (biru)
    ax.fill([3, 4, 4, 3], [1, 1, 4, 4], color='#87CEEB', alpha=0.8, edgecolor='#4682B4', linewidth=1)
    ax.fill([3.2, 3.8, 3.8, 3.2], [4, 4, 4.5, 4.5], color='#87CEEB', alpha=0.8, edgecolor='#4682B4', linewidth=1)
    ax.fill([3.3, 3.7, 3.7, 3.3], [4.5, 4.5, 4.8, 4.8], color='#4682B4', alpha=0.9)
    
    # Kaleng (ungu)
    ax.fill([6, 7, 7, 6], [1, 1, 3.5, 3.5], color='#DDA0DD', alpha=0.8, edgecolor='#9932CC', linewidth=1)
    ax.fill([5.8, 7.2, 7.2, 5.8], [3.5, 3.5, 3.8, 3.8], color='#DDA0DD', alpha=0.8, edgecolor='#9932CC', linewidth=1)
    
    # Daun organik (hijau)
    ax.fill([1, 2.5, 2, 1.5], [2, 3, 3.5, 2.5], color='#90EE90', alpha=0.8, edgecolor='#228B22', linewidth=1)
    ax.plot([1.5, 2.5], [2.5, 3.5], color='#228B22', linewidth=1)
    
    # Kulit buah (coklat)
    ax.fill([7.5, 9, 8.5, 7.5], [2, 2.5, 3, 2.5], color='#D2691E', alpha=0.8, edgecolor='#8B4513', linewidth=1)
    
    # Sampah kecil (titik-titik)
    ax.scatter([2, 8.5, 5], [1.5, 1.5, 1.5], color='#8B4513', s=15, alpha=0.7)
    
    ax.axis('off')
    
    return fig

# Tampilkan logo di sidebar
try:
    # Coba gunakan file logo jika ada
    logo_path = "assets/GambarSampah.png"
    if os.path.exists(logo_path):
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
        with col2:
            st.image(logo_path, width=150, caption="SmartWaste")
    else:
        # Buat logo secara dinamis
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
        with col2:
            logo_fig = create_logo()
            st.pyplot(logo_fig)
            plt.close(logo_fig)  # Tutup figure untuk menghemat memori
except Exception as e:
    # Jika gagal membuat logo, tampilkan emoji sebagai fallback
    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px; padding: 10px; background: linear-gradient(135deg, #E8F5E8, #F0FFF0); border-radius: 10px;'>
            <h2 style='color: #2E8B57; margin: 0;'>🗑️</h2>
            <h3 style='color: #2E8B57; margin: 5px 0; font-size: 16px;'>SmartWaste</h3>
        </div>
        """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**📋 Navigasi Aplikasi**")

# Buat list box untuk navigasi
navigation_options = {
    "🏠 Beranda": "beranda",
    "🗑️ Klasifikasi Sampah": "klasifikasi", 
    "📚 Edukasi Sampah": "edukasi"
}

# Gunakan selectbox untuk navigasi yang lebih modern
selected_page = st.sidebar.selectbox(
    "Pilih halaman:",
    list(navigation_options.keys()),
    label_visibility="collapsed"
)

# Konversi pilihan ke page yang sesuai
page_mapping = {
    "🏠 Beranda": "🏠 Beranda",
    "🗑️ Klasifikasi Sampah": "🗑️ Klasifikasi Sampah",
    "📚 Edukasi Sampah": "📚 Edukasi Sampah"
}

page = page_mapping[selected_page]

# Tambahkan informasi tambahan di sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #E8F5E8, #F0FFF0); padding: 10px; border-radius: 8px; margin: 10px 0;'>
    <h4 style='color: #2E8B57; margin: 0; text-align: center;'>ℹ️ Info</h4>
    <p style='color: #228B22; margin: 5px 0; font-size: 12px; text-align: center;'>
        Upload gambar sampah organik atau anorganik untuk klasifikasi otomatis
    </p>
</div>
""", unsafe_allow_html=True)

if page == "🏠 Beranda":
    page_home()
elif page == "🗑️ Klasifikasi Sampah":
    page_classification()
elif page == "📚 Edukasi Sampah":
    page_articles()
