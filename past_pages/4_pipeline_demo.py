import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

from core.segmentation import preprocess, otsu_threshold, morph_cleanup
from core.texture import histogram_lbp, histogram_gabor

st.set_page_config(layout="wide")
st.title("🔬 End-to-End Vision Pipeline Demo")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.image(gray, caption="Original Image", use_container_width=True)

    st.subheader("1️⃣ Segmentation")

    filtered = preprocess(gray)
    binary = otsu_threshold(filtered)
    clean = morph_cleanup(binary)

    c1, c2, c3 = st.columns(3)
    c1.image(filtered, caption="Filtered", use_container_width=True)
    c2.image(binary, caption="Otsu Mask", use_container_width=True)
    c3.image(clean, caption="Clean Mask", use_container_width=True)

    st.subheader("2️⃣ Texture Extraction")

    method = st.selectbox("Select Texture Method", ["LBP", "Gabor"])

    if method == "LBP":
        hist, fmap = histogram_lbp(gray)
    else:
        hist, fmap = histogram_gabor(gray)

    c1, c2 = st.columns(2)
    c1.image(fmap, caption="Feature Map", use_container_width=True)

    fig, ax = plt.subplots()
    ax.plot(hist)
    ax.set_title("Feature Histogram")
    c2.pyplot(fig)

    st.subheader("3️⃣ Feature Vector Summary")

    st.write("Feature Dimension:", len(hist))
    st.write("First 10 Values:", hist[:10])

    st.markdown("""
    ### 🧠 Explainable Pipeline
    - Segmentation isolates objects from background.
    - Texture encoding extracts discriminative patterns.
    - Histogram represents numerical feature vector.
    - Features are ready for classification models.
    """)

else:
    st.info("👆 Upload an image to run the pipeline.")