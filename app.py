import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
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
    initial_sidebar_state="auto",
)

# --- CSS Kustom untuk Tampilan Modern ---
st.markdown("""
<style>
    /* Gaya untuk container/kartu */
    .custom-container {
        border-radius: 15px;
        padding: 20px;
        background-color: #262730; /* Warna latar belakang kartu */
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        margin-bottom: 20px;
        height: 100%;
    }
    .custom-container:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    /* Gaya untuk header di dalam kartu */
    .custom-header {
        font-size: 24px;
        font-weight: bold;
        color: #1DB954; /* Warna hijau Spotify-like */
        margin-bottom: 10px;
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

# --- PERBAIKAN UTAMA 1: Fungsi untuk Membangun Arsitektur Model ---
def create_model(image_size=224):
    """
    Fungsi ini secara eksplisit membangun arsitektur model yang sama
    persis seperti pada notebook pelatihan. Ini lebih andal daripada load_model().
    """
    # Gunakan weights=None karena kita hanya butuh arsitekturnya.
    # Bobot (termasuk dari imagenet) akan dimuat dari file .keras Anda.
    effnet = EfficientNetB0(weights=None, include_top=False, input_shape=(image_size, image_size, 3))

    model_output = effnet.output
    model_output = tf.keras.layers.GlobalAveragePooling2D()(model_output)
    model_output = tf.keras.layers.Dropout(rate=0.50)(model_output)
    model_output = tf.keras.layers.BatchNormalization()(model_output)
    model_output = tf.keras.layers.Dense(4, activation='softmax')(model_output)
    model = tf.keras.models.Model(inputs=effnet.input, outputs=model_output)

    # Compile model agar bisa digunakan untuk prediksi
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

# --- PERBAIKAN UTAMA 2: Fungsi Cache untuk Memuat Bobot Model ---
@st.cache_resource
def load_model_weights():
    models = {}
    # Langkah 1: Unduh semua file model jika belum ada
    with st.spinner("Mohon tunggu, sedang mempersiapkan model... Ini hanya dilakukan sekali."):
        for model_name, info in MODEL_INFO.items():
            model_path = info["path"]
            if not os.path.exists(model_path):
                st.info(f"Mempersiapkan {model_name}...")
                try:
                    gdown.download(id=info["gdrive_id"], output=model_path, quiet=False)
                except Exception as e:
                    st.error(f"Gagal mengunduh {model_name}: {e}")
                    return None
    
    # Langkah 2: Buat arsitektur dan muat bobotnya
    try:
        for model_name, info in MODEL_INFO.items():
            # Buat kerangka arsitektur yang baru
            model = create_model()
            # Muat hanya bobot (weights) ke dalam kerangka tersebut
            model.load_weights(info["path"])
            models[model_name] = model
        st.success("Semua model berhasil dimuat! 🎉")
        return models
    except Exception as e:
        st.error(f"Error saat memuat bobot model dari file: {e}")
        st.error("Pastikan file model tidak korup dan arsitektur di kode ini cocok dengan saat training.")
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
    final_image = np.expand_dims(resized_image, axis=0)
    return final_image, None

def preprocess_enhanced(image_array, method='CLAHE', image_size=224):
    steps = {}
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
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

# --- UI Aplikasi Utama ---
models = load_model_weights() # Memanggil fungsi yang sudah diperbaiki

# --- HEADER DENGAN LOGO ---
col_logo1, col_logo_mid, col_logo2 = st.columns([1, 4, 1])
with col_logo1:
    logo_col1, logo_col2 = st.columns(2)
    try:
        with logo_col1:
            st.image("logo_itera.jpg", width=100)
        with logo_col2:
            st.image("logo_sainsdata.png", width=100)
    except Exception:
        st.warning("Logo tidak dapat dimuat.")
with col_logo_mid:
    st.title("🧠 Deteksi Tumor Otak Berbasis MRI")
    st.markdown("**Oleh: Muhammad Kaisar Firdaus** | *Sains Data, Institut Teknologi Sumatera*")

st.markdown("---")

# --- DESKRIPSI APLIKASI ---
with st.container():
    st.markdown(
        """
        Aplikasi ini memungkinkan perbandingan tiga model *Deep Learning* untuk deteksi tumor otak berdasarkan gambar scan MRI. 
        Pilih model, unggah gambar, dan lihat hasilnya secara instan.
        """
    )
    with st.expander("ℹ️ **Detail Teknis Model**"):
        st.markdown(
            """
            - **💻 Model**: Convolutional Neural Network (CNN)
            - **🕹️ Arsitektur**: EfficientNet-B0 + Lapisan Tambahan
            - **🎛️ Pendekatan**: *Transfer Learning* dengan bobot *pre-trained* dari ImageNet
            - **🖼️ Perlakuan Tambahan**: 
              1. **AHE** (*Adaptive Histogram Equalization*)
              2. **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*)
            """
        )

# --- KONTROL PENGGUNA ---
st.markdown("### 1. Pilih Model")

# --- PENJELASAN MODEL ---
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📦 **Model Dasar**")
        st.markdown("Model ini menggunakan citra MRI asli pada proses pelatihan, tanpa menggunakan metode peningkatan kontras sama sekali. Model ini lebih cepat tetapi mungkin kurang akurat dan akurasi yang sedikit lebih rendah pada gambar dengan kontras yang rendah.")
    with col2:
        st.info("✨ **Model dengan AHE**")
        st.markdown("Model ini menggunakan gambar yang telah ditingkatkan kontrasnya secara global dengan AHE. Ini membantu memperjelas fitur di seluruh gambar, namun terkadang, hasil peningkatan kontras banyak memunculkan noise atau bercak putih pada gambar.")
    with col3:
        st.info("🌟 **Model dengan CLAHE** *(Recomended)*")
        st.markdown("Model ini menggunakan CLAHE untuk meningkatkan kontras secara lokal. Ini sangat efektif untuk menonjolkan detail halus di area spesifik gambar MRI, dengan hasil yang lebih baik dan detail yang jelas, sehingga akurasi dan prediksi model lebih akurat.")

selected_model_name = st.selectbox(
    "Pilih model yang ingin Anda gunakan untuk prediksi:",
    list(MODEL_INFO.keys()),
    label_visibility="collapsed" 
)

st.markdown("### 2. Unggah Gambar")
uploaded_file = st.file_uploader(
    "Pilih file gambar MRI (format .jpg, .jpeg, atau .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and models is not None:
    # Buka gambar dan langsung konversi ke format RGB untuk konsistensi.
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    st.markdown("---")
    st.header("🔬 Hasil Analisis")
    
    processing_steps = None
    processed_image_for_model = None
    
    with st.spinner("Gambar sedang diproses dan diprediksi...🚀"):
        if selected_model_name == "Model Dasar (Tanpa Perlakuan)":
            processed_image_for_model, _ = preprocess_base(image_np)
        elif selected_model_name == "Model dengan AHE":
            processed_image_for_model, processing_steps = preprocess_enhanced(image_np, method='AHE')
        else: # CLAHE
            processed_image_for_model, processing_steps = preprocess_enhanced(image_np, method='CLAHE')

        model = models[selected_model_name]
        prediction = model.predict(processed_image_for_model)
        predicted_class_index = np.argmax(prediction)
        
        # Label harus sesuai urutan abjad yang dipelajari model
        class_labels_from_model = ['Glioma', 'Meningioma', 'notumor', 'Pituitary']
        predicted_label_from_model = class_labels_from_model[predicted_class_index]

        # Mapping untuk tampilan yang lebih ramah pengguna
        display_mapping = {
            'Glioma': 'Glioma',
            'Meningioma': 'Meningioma',
            'notumor': 'Tanpa Tumor',
            'Pituitary': 'Pituitary'
        }
        predicted_class_label = display_mapping[predicted_label_from_model]
        confidence = np.max(prediction) * 100

    # --- Tampilan Hasil ---
    if processing_steps:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Gambar MRI Asli', use_column_width=True)
        with col2:
            final_processed_key = '5. Unsharp Masked'
            st.image(processing_steps[final_processed_key], caption=f'Gambar Setelah Pra-pemrosesan', use_column_width=True)
    else:
        st.image(image, caption='Gambar MRI Asli', width=300)

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
        
        display_labels_for_chart = list(display_mapping.values())
        prob_df = pd.DataFrame({'Kelas': display_labels_for_chart, 'Probabilitas': prediction[0] * 100})
        
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
            num_steps = len(processing_steps)
            cols = st.columns(num_steps)
            for idx, (step_name, step_image) in enumerate(processing_steps.items()):
                with cols[idx]:
                    st.image(step_image, caption=step_name, use_column_width=True)

elif models is None:
    st.error("Model tidak berhasil dimuat. Aplikasi tidak dapat melanjutkan. Silakan periksa log error di atas dan coba muat ulang halaman.")
else:
    st.info("Silakan unggah gambar MRI untuk memulai analisis. ⏱️")
