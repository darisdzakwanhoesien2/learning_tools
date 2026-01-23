import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

from core.texture import histogram_lbp, histogram_gabor

st.set_page_config(layout="wide")
st.title("🧵 Texture Feature Playground (Assignment 4)")

uploaded = st.file_uploader("Upload grayscale or RGB image", type=["png", "jpg", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    col1, col2 = st.columns(2)
    col1.image(gray, caption="Input Grayscale Image", use_container_width=True)

    feature_type = st.selectbox(
        "Select Texture Feature",
        ["LBP Histogram", "Gabor Filter Bank"]
    )

    if feature_type == "LBP Histogram":
        hist, lbp_map = histogram_lbp(gray)

        col2.image(lbp_map, caption="LBP Map", use_container_width=True)

        st.subheader("📊 LBP Histogram (256 bins)")
        fig, ax = plt.subplots()
        ax.plot(hist)
        ax.set_xlabel("LBP Code")
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        st.success(f"Feature Dimension: {hist.shape[0]}")

        st.markdown("""
        **Explainability**
        - Each pixel encodes local texture using an 8-bit binary pattern.
        - Histogram represents texture distribution across the image.
        - Robust to illumination changes.
        """)

    elif feature_type == "Gabor Filter Bank":
        hist, texture_map = histogram_gabor(gray)

        col2.image(texture_map, caption="Gabor Texture Map", use_container_width=True)

        st.subheader("📊 Gabor Histogram (16 bins)")
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(hist)), hist)
        ax.set_xlabel("Texture Code")
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        st.success(f"Feature Dimension: {hist.shape[0]}")

        st.markdown("""
        **Explainability**
        - Gabor filters capture orientation and frequency information.
        - Binary responses encode texture patterns.
        - Histogram summarizes spatial frequency distribution.
        """)

else:
    st.info("👆 Upload an image to begin.")