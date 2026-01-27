import streamlit as st
import cv2
import numpy as np
from core.segmentation import *

st.title("🧩 Segmentation Playground (A3)")

uploaded = st.file_uploader("Upload image", type=["jpg","png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.image(gray, caption="Grayscale", use_container_width=True)

    sigma = st.slider("Bilateral Sigma", 10, 100, 30)
    filtered = preprocess(gray, sigma=sigma)

    binary = otsu_threshold(filtered)
    st.image(binary, caption="Otsu Segmentation")

    close_k = st.slider("Closing Kernel", 3, 15, 3, step=2)
    open_k  = st.slider("Opening Kernel", 3, 15, 5, step=2)

    cleaned = morph_cleanup(binary, close_k, open_k)
    st.image(cleaned, caption="Morphological Cleanup")

    min_area = st.slider("Min Object Area", 100, 3000, 500)
    final, count = remove_small_objects(cleaned, min_area)

    st.image(final, caption=f"Final Objects ({count})")