import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import gdown
import pandas as pd
import matplotlib.pyplot as plt

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Brain Tumor AI Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Kustom untuk Tampilan Modern ---
st.markdown("""
<style>
    /* Mengurangi padding atas dari halaman utama */
    .main .block-container {
        padding-top: 2rem;
    }
    /* Gaya untuk container/kartu */
    .custom-container {
        border-radius: 15px;
        padding: 1.5rem;
        background-color: #262730;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        margin-bottom: 20px;
        border: 1px solid #444;
    }
    .custom-container:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    /* Gaya untuk header di dalam kartu */
    .custom-header {
        font-size: 22px;
        font-weight: bold;
        color: #1DB954; /* Warna hijau Spotify-like */
        margin-bottom: 10px;
    }
    /* Gaya untuk sidebar */
    [data-testid="stSidebar"] {
        background-color: #191414; /* Warna latar belakang sidebar */
    }
</style>
""", unsafe_allow_html=True)

# --- Kamus Informasi Model ---
MODEL_INFO = {
    "Model Dasar (Tanpa Perlakuan)": {
        "path": "base_model.keras",
        "gdrive_id": "17w7n0H5XH6eiOHqkq4RzScmIig8Xt1Mf"
    },
    "Model dengan AHE": {
        "path": "ahe_model.keras",
        "gdrive_id": "1Z_st-eTAEc20RcKvvAuY6bSw9Q1B9_PZ"
    },
    "Model dengan CLAHE": {
        "path": "clahe_model.keras",
        "gdrive_id": "1N-vil1LujsZ4kjFhf6_Lhh81jWUqvrQm"
    }
}

# --- Fungsi Cache untuk Memuat Semua Model ---
@st.cache_resource
def load_models():
    models = {}
    with st.spinner("Mempersiapkan model AI... Ini hanya dilakukan sekali."):
        for model_name, info in MODEL_INFO.items():
            model_path = info["path"]
            if not os.path.exists(model_path):
                try:
                    gdown.download(id=info["gdrive_id"], output=model_path, quiet=True)
                except Exception as e:
                    st.error(f"Gagal mengunduh {model_name}: {e}")
                    return None
    try:
        for model_name, info in MODEL_INFO.items():
            models[model_name] = tf.keras.models.load_model(info["path"])
        st.sidebar.success("Semua model berhasil dimuat! 😉")
        return models
    except Exception as e:
        st.error(f"Error saat memuat model dari file lokal: {e}")
        return None

# --- Fungsi Pra-pemrosesan Gambar ---
def resize_with_padding(image, target_size=224):
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.full((target_size, target_size), 0, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized
    return padded

def preprocess_base(image_array, image_size=224):
    resized_image = cv2.resize(image_array, (image_size, image_size))
    if len(resized_image.shape) == 2:
        three_channel_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    else:
        three_channel_image = resized_image
    final_image = np.expand_dims(three_channel_image, axis=0)
    return final_image, None

def preprocess_enhanced(image_array, method='CLAHE', image_size=224):
    steps = {}
    if len(image_array.shape) > 2 and image_array.shape[2] == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_array
    steps['1. Grayscale'] = gray_image

    padded_image = resize_with_padding(gray_image, image_size)
    steps['2. Resized & Padded'] = padded_image

    if method == 'AHE':
        enhancer = cv2.createCLAHE(clipLimit=0.0, tileGridSize=(8, 8))
        enhanced_image = enhancer.apply(padded_image)
        steps['3. AHE'] = enhanced_image
    else:
        enhancer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = enhancer.apply(padded_image)
        steps['3. CLAHE'] = enhanced_image

    denoised_image = cv2.fastNlMeansDenoising(enhanced_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    steps['4. Denoised'] = denoised_image

    blurred = cv2.GaussianBlur(denoised_image, (5, 5), 0)
    unsharp_image = cv2.addWeighted(denoised_image, 1.5, blurred, -0.5, 0)
    steps['5. Unsharp Masked'] = unsharp_image

    three_channel_image = cv2.cvtColor(unsharp_image, cv2.COLOR_GRAY2RGB)
    final_image = np.expand_dims(three_channel_image, axis=0)
    return final_image, steps

# --- Informasi Tumor ---
TUMOR_INFO = {
    "Glioma": "Glioma adalah jenis tumor yang tumbuh dari sel glial di otak. Tumor ini bisa bersifat jinak atau ganas dan merupakan salah satu tumor otak primer yang paling umum.",
    "Meningioma": "Meningioma adalah tumor yang terbentuk pada meninges, yaitu selaput yang melindungi otak dan sumsum tulang belakang. Sebagian besar meningioma bersifat jinak (non-kanker).",
    "Pituitary": "Tumor hipofisis (pituitary) adalah pertumbuhan abnormal yang berkembang di kelenjar hipofisis. Sebagian besar tumor ini jinak dan dapat menyebabkan masalah hormonal.",
    "Tanpa Tumor": "Hasil pemindaian tidak menunjukkan adanya tanda-tanda tumor otak yang jelas. Namun, konsultasi dengan ahli medis tetap disarankan untuk konfirmasi."
}

# --- UI Sidebar ---
with st.sidebar:
    st.title("Pengaturan")
    st.markdown("---")
    
    # Menampilkan logo
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.image("logo_itera.jpg", use_column_width=True)
        with col2:
            st.image("logo_sainsdata.png", use_column_width=True)
    except Exception:
        st.warning("Logo tidak dapat dimuat.")
    
    st.markdown("**Oleh: Muhammad Kaisar Firdaus**")
    st.caption("*Program Studi Sains Data, Institut Teknologi Sumatera*")
    st.markdown("---")

    models = load_models()
    
    st.header("1. Pilih Model")
    with st.expander("ℹ️ **Penjelasan Model**", expanded=True):
        st.info("📦 **Model Dasar**")
        st.markdown("Menggunakan gambar MRI asli. Cepat, namun akurasi bisa lebih rendah pada gambar berkontras rendah.")
        st.info("✨ **Model dengan AHE**")
        st.markdown("Meningkatkan kontras gambar secara global. Memperjelas fitur, namun bisa menimbulkan *noise*.")
        st.info("🌟 **Model dengan CLAHE** (Disarankan)")
        st.markdown("Meningkatkan kontras secara lokal. Efektif menonjolkan detail halus untuk akurasi lebih tinggi.")

    selected_model_name = st.selectbox(
        "Pilih model yang ingin Anda gunakan:",
        list(MODEL_INFO.keys()),
        index=2 # Default ke CLAHE
    )
    
    st.markdown("---")
    st.header("2. Unggah Gambar")
    uploaded_file = st.file_uploader(
        "Pilih file gambar MRI...",
        type=["jpg", "jpeg", "png"]
    )

# --- UI Halaman Utama ---
st.title("🧠 Deteksi Tumor Otak Berbasis MRI")
st.markdown("Selamat datang di aplikasi deteksi tumor otak. Silakan pilih model dan unggah gambar MRI Anda melalui panel di sebelah kiri untuk memulai analisis.")
st.markdown("---")

class_labels = ['Glioma', 'Meningioma', 'Tanpa Tumor', 'Pituitary']

if uploaded_file is not None and models is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    st.header("🔬 Hasil Analisis Citra")
    
    with st.spinner("Gambar sedang diproses dan diprediksi...🚀"):
        if selected_model_name == "Model Dasar (Tanpa Perlakuan)":
            processed_image_for_model, _ = preprocess_base(image_np)
            processing_steps = None
        elif selected_model_name == "Model dengan AHE":
            processed_image_for_model, processing_steps = preprocess_enhanced(image_np, method='AHE')
        else: # CLAHE
            processed_image_for_model, processing_steps = preprocess_enhanced(image_np, method='CLAHE')

        model = models[selected_model_name]
        prediction = model.predict(processed_image_for_model)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        confidence = np.max(prediction) * 100

    # --- Tampilan Hasil Gambar ---
    if processing_steps:
        tab1, tab2 = st.tabs(["🖼️ Gambar Asli", "✨ Gambar Setelah Pra-pemrosesan"])
        with tab1:
            st.image(image, caption='Gambar MRI Asli', use_column_width=True)
        with tab2:
            final_processed_key = '5. Unsharp Masked'
            st.image(processing_steps[final_processed_key], caption=f'Gambar Hasil {selected_model_name.split(" ")[2]}', use_column_width=True)
    else:
        st.image(image, caption='Gambar MRI Asli', width=400)

    st.markdown("---")
    st.header("📊 Hasil Prediksi")

    col_res1, col_res2 = st.columns([2, 3])
    with col_res1:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown('<p class="custom-header">Hasil Deteksi</p>', unsafe_allow_html=True)
        st.success(f"**Jenis Terdeteksi:** {predicted_class_label}")
        st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")
        st.markdown("---")
        st.markdown("##### **Deskripsi Singkat**")
        st.write(TUMOR_INFO[predicted_class_label])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res2:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown('<p class="custom-header">Distribusi Probabilitas</p>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({'Kelas': class_labels, 'Probabilitas': prediction[0] * 100})
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(prob_df['Kelas'], prob_df['Probabilitas'], color='#1DB954')
        ax.bar_label(bars, fmt='%.2f%%', padding=3, color='white', fontsize=10)
        ax.set_xlim(0, 115)
        ax.set_xlabel('Probabilitas (%)', fontsize=12, color='white')
        ax.tick_params(axis='y', colors='white', length=0)
        ax.tick_params(axis='x', colors='white', labelsize=10)
        
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#555555')
        ax.grid(axis='x', linestyle='--', alpha=0.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    if processing_steps:
        with st.expander("🔬 **Lihat Detail Langkah Pra-pemrosesan**"):
            cols = st.columns(len(processing_steps))
            for idx, (step_name, step_image) in enumerate(processing_steps.items()):
                with cols[idx]:
                    st.image(step_image, caption=step_name, use_column_width=True)

else:
    st.info("Selamat datang! Silakan pilih model dan unggah gambar di panel sebelah kiri untuk memulai. ⏱️")

