import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2

# Load model YOLOv5 pretrained dari torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

st.title("Deteksi Objek Hewan dengan YOLOv5")
st.markdown("Upload gambar berisi hewan untuk mendeteksi keberadaannya.")

uploaded_file = st.file_uploader("Upload gambar", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diupload', use_column_width=True)

    st.write("🔍 Mendeteksi objek...")

    # Konversi ke numpy
    img_np = np.array(image)

    # Deteksi objek
    results = model(img_np)

    # Ambil gambar hasil dengan bounding box
    results.render()
    hasil_img = results.ims[0]

    # Tampilkan hasil deteksi
    st.image(hasil_img, caption='Hasil Deteksi Objek Hewan', use_column_width=True)

    # Tampilkan label yang terdeteksi
    detected = results.pandas().xyxy[0]
    if not detected.empty:
        st.markdown("### Objek yang Terdeteksi:")
        for index, row in detected.iterrows():
            st.write(f"- **{row['name']}** dengan confidence {row['confidence']:.2f}")
    else:
        st.write("Tidak ada hewan terdeteksi.")
