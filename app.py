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
    page_title="Deteksi Tumor Otak",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Fungsi Cache untuk Memuat Model ---
@st.cache_resource
def load_model():
    """
    Memeriksa, mengunduh jika perlu, dan memuat model Keras.
    """
    model_path = 'brain-final.keras'
    if not os.path.exists(model_path):
        with st.spinner("Mohon tunggu, sedang mempersiapkan model 👌..."):
            # Ganti dengan ID file Google Drive Anda
            file_id = "1-q0H1ncvzAJt0DaEYz5yz0BOv4axQoFe"
            try:
                gdown.download(id=file_id, output=model_path, quiet=False)
                st.success("Model berhasil diunduh! 😉")
            except Exception as e:
                st.error(f"Gagal mengunduh model 🥲: {e}")
                return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
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

def preprocess_for_prediction(image_array, image_size=224):
    """
    Pipeline pra-pemrosesan lengkap.
    Mengembalikan gambar akhir untuk model dan dictionary berisi langkah-langkah pra-pemrosesan.
    """
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

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(padded_image)
    steps['CLAHE'] = clahe_image

    # 4. Denoising
    denoised_image = cv2.fastNlMeansDenoising(clahe_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    steps['Denoised'] = denoised_image

    # 5. Unsharp Mask
    blurred = cv2.GaussianBlur(denoised_image, (5, 5), 0)
    unsharp_image = cv2.addWeighted(denoised_image, 1.5, blurred, -0.5, 0)
    steps['Unsharp Masked'] = unsharp_image

    # 6. Konversi ke 3 Channel untuk model
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
model = load_model()

st.title("🧠 Deteksi Tumor Otak Berbasis MRI")
st.markdown(
    """
    🤓**Author:** Muhammad Kaisar Firdaus  
    🏢*Program Studi Sains Data, Fakultas Sains, Institut Teknologi Sumatera*
    
    Website ini merupakan hasil penelitian skripsi S1 oleh author, dan merupakan karya sendiri tanpa campur tangan pihak lain.
    """
)
st.markdown(
    """
    Website ini menggunakan model *Convolutional Neural Network* dengan *transfer learning* **EfficientNet-B0**
    dan menerapkan pra-pemrosesan citra, yaitu *Clip Limit Adaptive Histogram Equalization* (CLAHE).
    """
)
st.markdown("---")

class_labels = ['Glioma', 'Meningioma', 'Tanpa Tumor', 'Pituitary']

uploaded_file = st.file_uploader(
    "Silakan Unggah Gambar Scan MRI (format .jpg, .jpeg, atau .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Gambar MRI Asli', use_column_width=True)

    with st.spinner("Gambar sedang diproses dan diprediksi...🚀"):
        # Pra-pemrosesan dan prediksi
        processed_image_for_model, processing_steps = preprocess_for_prediction(image_np)
        prediction = model.predict(processed_image_for_model)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        confidence = np.max(prediction) * 100

    with col2:
        st.image(processing_steps['Unsharp Masked'], caption='Gambar dengan Pra-pemrosesan CLAHE', use_column_width=True)

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

        # --- MENGGUNAKAN MATPLOTLIB UNTUK KUSTOMISASI GRAFIK ---
        fig, ax = plt.subplots(figsize=(7, 4))
        
        # Membuat bar chart horizontal agar label selalu terbaca
        bars = ax.barh(prob_df['Kelas'], prob_df['Probabilitas'], color='steelblue')
        
        # Menambahkan label persentase di ujung setiap bar
        ax.bar_label(bars, fmt='%.2f%%', padding=3, color='gray', fontsize=10)

        # Kustomisasi tampilan plot
        ax.set_xlim(0, 115)
        ax.set_xlabel('Probabilitas (%)', fontsize=12)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=10)
        
        # Sembunyikan bingkai dan judul yang tidak perlu
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        fig.tight_layout()
        st.pyplot(fig)
    
    with st.expander("Lihat Detail Langkah Pra-pemrosesan"):
        st.write("Berikut adalah visualisasi dari setiap langkah pra-pemrosesan CLAHE yang diterapkan pada gambar:")
        
        cols = st.columns(len(processing_steps))
        for idx, (step_name, step_image) in enumerate(processing_steps.items()):
            with cols[idx]:
                st.image(step_image, caption=step_name, use_column_width=True)

else:
    st.info("Menunggu gambar MRI untuk diunggah...⏱️")
