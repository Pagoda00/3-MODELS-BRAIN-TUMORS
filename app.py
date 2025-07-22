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
    page_title="Deteksi Tumor Otak Multi-Model",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Kamus Informasi Model ---
# Menyimpan semua informasi model di satu tempat agar mudah dikelola
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
    with st.spinner("Mohon tunggu, sedang mempersiapkan model..."):
        for model_name, info in MODEL_INFO.items():
            model_path = info["path"]
            if not os.path.exists(model_path):
                st.info(f"Mempersiapkan {model_name}...")
                try:
                    gdown.download(id=info["gdrive_id"], output=model_path, quiet=False)
                except Exception as e:
                    st.error(f"Gagal mengunduh {model_name}: {e}")
                    return None
    
    try:
        for model_name, info in MODEL_INFO.items():
            models[model_name] = tf.keras.models.load_model(info["path"])
        st.success("Semua model berhasil dimuat! 😉")
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
    """Pra-pemrosesan sederhana untuk model dasar."""
    # Resize gambar langsung ke 224x224
    resized_image = cv2.resize(image_array, (image_size, image_size))
    # Pastikan gambar 3 channel
    if len(resized_image.shape) == 2:
        three_channel_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    else:
        three_channel_image = resized_image
    # Tambahkan dimensi batch
    final_image = np.expand_dims(three_channel_image, axis=0)
    return final_image, None # Kembalikan None untuk steps

def preprocess_enhanced(image_array, method='CLAHE', image_size=224):
    """Pipeline pra-pemrosesan untuk AHE atau CLAHE."""
    steps = {}
    
    # 1. Grayscale
    if len(image_array.shape) > 2 and image_array.shape[2] == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_array
    steps['Grayscale'] = gray_image

    # 2. Resizing with Padding
    padded_image = resize_with_padding(gray_image, image_size)
    steps['Resized & Padded'] = padded_image

    # 3. AHE atau CLAHE
    if method == 'AHE':
        # AHE adalah kasus khusus dari CLAHE dengan clipLimit=0
        enhancer = cv2.createCLAHE(clipLimit=0.0, tileGridSize=(8, 8))
        enhanced_image = enhancer.apply(padded_image)
        steps['AHE'] = enhanced_image
    else: # Default ke CLAHE
        enhancer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = enhancer.apply(padded_image)
        steps['CLAHE'] = enhanced_image

    # 4. Denoising
    denoised_image = cv2.fastNlMeansDenoising(enhanced_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    steps['Denoised'] = denoised_image

    # 5. Unsharp Mask
    blurred = cv2.GaussianBlur(denoised_image, (5, 5), 0)
    unsharp_image = cv2.addWeighted(denoised_image, 1.5, blurred, -0.5, 0)
    steps['Unsharp Masked'] = unsharp_image

    # 6. Konversi ke 3 Channel
    three_channel_image = cv2.cvtColor(unsharp_image, cv2.COLOR_GRAY2RGB)

    # 7. Tambahkan dimensi batch
    final_image = np.expand_dims(three_channel_image, axis=0)

    return final_image, steps

# --- Informasi Tumor ---
TUMOR_INFO = {
    "Glioma": "Glioma adalah jenis tumor yang tumbuh dari sel glial di otak. Tumor ini bisa bersifat jinak atau ganas dan merupakan salah satu tumor otak primer yang paling umum.",
    "Meningioma": "Meningioma adalah tumor yang terbentuk pada meninges, yaitu selaput yang melindungi otak dan sumsum tulang belakang. Sebagian besar meningioma bersifat jinak (non-kanker).",
    "Pituitary": "Tumor hipofisis (pituitary) adalah pertumbuhan abnormal yang berkembang di kelenjar hipofisis. Sebagian besar tumor ini jinak dan dapat menyebabkan masalah hormonal.",
    "Tanpa Tumor": "Hasil pemindaian tidak menunjukkan adanya tanda-tanda tumor otak yang jelas. Namun, konsultasi dengan ahli medis tetap disarankan untuk konfirmasi."
}

# --- UI Aplikasi Utama ---
models = load_models()

st.title("🧠 Deteksi Tumor Otak Berbasis MRI")
st.markdown(
    """
    🤓Author: **Muhammad Kaisar Firdaus**  
    🏢Program Studi Sains Data, Fakultas Sains, Institut Teknologi Sumatera
    
    Website ini merupakan hasil penelitian skripsi S1 oleh author. Website ini bisa digunakan untuk memprediksi jenis tumor berdasarkan gambar scan MRI otak manusia.
    """
)
st.markdown("---")
st.markdown(
    """
    - 💻Model : Convolutional Neural Network.
    - 🕹️Arsitektur : EfficientNet-B0 + lapisan tambahan.
    - 🎛️Pendekatan : Transfer learning dengan bobot petrained model dari ImageNet.
    
    - Perlakuan Tambahan : 
    1. Adaptive Histogram Equalization (AHE)
    2. Clip Limit Adaptive Histogram Equalization (CLAHE)
    """
)
st.markdown("---")
st.markdown(
    """
    Jenis Tumor Otak yang Tersedia :
    1. Glioma
    2. Meningioma
    3. Pituitary
    4. No Tumor
    """
)
st.markdown("---")
# Pilihan Model untuk Pengguna
selected_model_name = st.selectbox(
    "Pilih model yang ingin Anda gunakan:",
    list(MODEL_INFO.keys())
)

class_labels = ['Glioma', 'Meningioma', 'Tanpa Tumor', 'Pituitary']

uploaded_file = st.file_uploader(
    "Silakan Unggah Gambar Scan MRI (format .jpg, .jpeg, atau .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and models is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    st.markdown("---")
    
    processing_steps = None
    processed_image_for_model = None
    
    with st.spinner("Gambar sedang diproses dan diprediksi...🚀"):
        # Logika kondisional untuk pra-pemrosesan
        if selected_model_name == "Model Dasar (Tanpa Perlakuan)":
            processed_image_for_model, _ = preprocess_base(image_np)
        elif selected_model_name == "Model dengan AHE":
            processed_image_for_model, processing_steps = preprocess_enhanced(image_np, method='AHE')
        elif selected_model_name == "Model dengan CLAHE":
            processed_image_for_model, processing_steps = preprocess_enhanced(image_np, method='CLAHE')

        # Prediksi menggunakan model yang dipilih
        model = models[selected_model_name]
        prediction = model.predict(processed_image_for_model)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        confidence = np.max(prediction) * 100

    # Tampilan Hasil
    st.header("Hasil Analisis")
    
    # Tampilan jika menggunakan model AHE atau CLAHE
    if processing_steps:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Gambar MRI Asli', use_column_width=True)
        with col2:
            final_processed_key = 'Unsharp Masked'
            st.image(processing_steps[final_processed_key], caption=f'Gambar Setelah Pra-pemrosesan ({selected_model_name.split(" ")[2]})', use_column_width=True)
    # Tampilan jika menggunakan model dasar
    else:
        st.image(image, caption='Gambar MRI Asli', width=300)

    st.markdown("---")
    st.header("Hasil Prediksi")

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.success(f"**Jenis Terdeteksi:** {predicted_class_label}")
        st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")
        st.markdown("##### Deskripsi:")
        st.write(TUMOR_INFO[predicted_class_label])

    with col_res2:
        st.markdown("##### Distribusi Probabilitas:")
        prob_df = pd.DataFrame({
            'Kelas': class_labels,
            'Probabilitas': prediction[0] * 100
        })
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(prob_df['Kelas'], prob_df['Probabilitas'], color='steelblue')
        ax.bar_label(bars, fmt='%.2f%%', padding=3, color='gray', fontsize=10)
        ax.set_xlim(0, 115)
        ax.set_xlabel('Probabilitas (%)', fontsize=12)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)
    
    # Expander hanya muncul jika ada langkah pra-pemrosesan (AHE/CLAHE)
    if processing_steps:
        with st.expander("Lihat Detail Langkah Pra-pemrosesan"):
            st.write(f"Berikut adalah visualisasi dari setiap langkah pra-pemrosesan {selected_model_name.split(' ')[2]}:")
            
            cols = st.columns(len(processing_steps))
            for idx, (step_name, step_image) in enumerate(processing_steps.items()):
                with cols[idx]:
                    st.image(step_image, caption=step_name, use_column_width=True)

else:
    st.info("Menunggu gambar MRI untuk diunggah...⏱️")
