import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import gdown
import os
import requests
from io import BytesIO

# --- KONFIGURASI AWAL ---
# Pastikan folder yang diperlukan ada
for folder in ["Model", "upload_images", "meta"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ID file dari Google Drive
file_id = "100tjmivqh1OAXejHcCz-YHKKOB9q_so4"
url = f"https://drive.google.com/uc?id={file_id}"
output = "Model/BC.h5"

# Unduh model jika belum ada
if not os.path.exists(output):
    with st.spinner("Mengunduh model dari Google Drive..."):
        gdown.download(url, output, quiet=False)

# Load model (Gunakan cache agar tidak lambat saat refresh)
@st.cache_resource
def get_model():
    return load_model(output, compile=False)

model = get_model()

# Dictionary Label (Dipangkas untuk contoh, pastikan 270 kelas Anda lengkap di sini)
lab = {
    0: "AFRICAN CROWNED CRANE", 1: "AFRICAN FIREFINCH", 2: "ALBATROSS", 3: "ALEXANDRINE PARAKEET",
    4: "AMERICAN AVOCET", 5: "AMERICAN BITTERN", 6: "AMERICAN COOT", 7: "AMERICAN GOLDFINCH",
    8: "AMERICAN KESTREL", 9: "AMERICAN PIPIT", 10: "AMERICAN REDSTART", 11: "ANHINGA",
    12: "ANNAS HUMMINGBIRD", 13: "ANTBIRD", 14: "ARARIPE MANAKIN", 15: "ASIAN CRESTED IBIS",
    16: "BALD EAGLE", 17: "BALI STARLING", 18: "BALTIMORE ORIOLE", 19: "BANANAQUIT",
    20: "BANDED BROADBILL", 21: "BAR-TAILED GODWIT", 22: "BARN OWL", 23: "BARN SWALLOW",
    24: "BARRED PUFFBIRD", 25: "BAY-BREASTED WARBLER", 26: "BEARDED BARBET", 27: "BEARDED REEDLING",
    28: "BELTED KINGFISHER", 29: "BIRD OF PARADISE", 30: "BLACK & YELLOW bROADBILL",
    # ... pastikan semua kelas 0-269 ada di sini sesuai kodingan asli Anda ...
}

# --- FUNGSI PEMPROSESAN ---

def processed_img(img_path):
    # Load dan preprocess
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Prediksi
    answer = model.predict(img)
    
    # Perbaikan pengambilan confidence agar tidak TypeError
    predicted_class_index = np.argmax(answer, axis=-1)[0]
    confidence = float(answer.flatten()[predicted_class_index])
    
    predicted_label = lab.get(predicted_class_index, "Unknown")
    return predicted_label, confidence

def processed_img_from_url(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize((224, 224))
    
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    answer = model.predict(img_array)
    
    predicted_class_index = np.argmax(answer, axis=-1)[0]
    confidence = float(answer.flatten()[predicted_class_index])
    
    res = lab.get(predicted_class_index, "Unknown")
    return res, confidence

# --- UI STREAMLIT ---

def run():
    st.set_page_config(page_title="Bird Classification", page_icon=":bird:")
    
    # Header Gambar (Opsional jika file meta/bird.png tidak ada, beri proteksi)
    try:
        img1 = Image.open("./meta/bird.png")
        st.image(img1, width=350)
    except:
        st.info("Logo burung tidak ditemukan, melewati...")

    st.title(":bird: Klasifikasi Spesies Burung")
    st.subheader("Kelompok Pert15 - Digital Image Processing")
     st.subheader("Ardo Pakusadewo-Faisal")
    option = st.radio("Pilih sumber gambar:", ("Unggah Gambar", "Gunakan URL"))
    img_file = None
    img_url = ""

    if option == "Unggah Gambar":
        img_file = st.file_uploader("Pilih Gambar Burung", type=["jpg", "png", "jpeg"])
        if img_file is not None:
            st.image(img_file, caption="Gambar yang Diunggah", width=400)
            save_image_path = os.path.join("upload_images", img_file.name)
            with open(save_image_path, "wb") as f:
                f.write(img_file.getbuffer())
    else:
        img_url = st.text_input("Masukkan URL Gambar")
        if img_url:
            try:
                response = requests.get(img_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="Gambar dari URL", width=400)
            except:
                st.error("Gagal memuat gambar dari URL.")

    if st.button("Prediksi"):
        with st.spinner("Menganalisis gambar..."):
            try:
                if option == "Unggah Gambar" and img_file is not None:
                    result, confidence = processed_img(save_image_path)
                elif option == "Gunakan URL" and img_url:
                    result, confidence = processed_img_from_url(img_url)
                else:
                    st.warning("Silakan masukkan gambar terlebih dahulu.")
                    return

                st.success(f"**Hasil Prediksi:** {result}")
                st.info(f"**Tingkat Kepercayaan:** {confidence:.2%}")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

if __name__ == "__main__":
    run()
