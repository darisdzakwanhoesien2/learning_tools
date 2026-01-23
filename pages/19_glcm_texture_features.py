# GLCM (Gray-Level Co-occurrence Matrix) + texture feature extraction

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="GLCM Texture Features", layout="wide")
st.title("🧩 Co-occurrence Matrix & Texture Features")

# =====================================================
# INPUT IMAGES
# =====================================================

I1 = np.array([
    [2,0,0,1],
    [1,2,0,0],
    [0,2,1,0],
    [0,0,2,2]
])

I2 = np.array([
    [0,2,0,1],
    [1,0,2,0],
    [0,2,0,1],
    [1,0,2,0]
])

levels = [0,1,2]
L = len(levels)

st.header("📥 Input Images")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Image 1")
    st.dataframe(I1)

with col2:
    st.subheader("Image 2")
    st.dataframe(I2)

# =====================================================
# GLCM COMPUTATION
# =====================================================

st.divider()
st.header("🧮 Step 1 — GLCM Computation")

def compute_glcm(image, dx, dy, levels):
    glcm = np.zeros((len(levels), len(levels)), dtype=int)
    h, w = image.shape

    for y in range(h):
        for x in range(w):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < w and 0 <= ny < h:
                i = image[y, x]
                j = image[ny, nx]
                glcm[i, j] += 1

    return glcm

C10_I1 = compute_glcm(I1, dx=1, dy=0, levels=levels)
C11_I1 = compute_glcm(I1, dx=1, dy=1, levels=levels)

C10_I2 = compute_glcm(I2, dx=1, dy=0, levels=levels)
C11_I2 = compute_glcm(I2, dx=1, dy=1, levels=levels)

st.markdown("### Image 1 GLCMs")
st.write("C[1,0]:")
st.dataframe(C10_I1)
st.write("C[1,1]:")
st.dataframe(C11_I1)

st.markdown("### Image 2 GLCMs")
st.write("C[1,0]:")
st.dataframe(C10_I2)
st.write("C[1,1]:")
st.dataframe(C11_I2)

# =====================================================
# NORMALIZATION
# =====================================================

st.divider()
st.header("📏 Step 2 — Normalize GLCM")

def normalize_glcm(C):
    return C / np.sum(C)

P10_I1 = normalize_glcm(C10_I1)
P11_I1 = normalize_glcm(C11_I1)

P10_I2 = normalize_glcm(C10_I2)
P11_I2 = normalize_glcm(C11_I2)

# =====================================================
# TEXTURE FEATURES
# =====================================================

st.divider()
st.header("📊 Step 3 — Texture Features")

def texture_features(P):
    i, j = np.indices(P.shape)

    energy = np.sum(P**2)
    entropy = -np.sum(P[P>0] * np.log2(P[P>0]))
    contrast = np.sum((i - j)**2 * P)
    homogeneity = np.sum(P / (1 + np.abs(i - j)))

    mu_i = np.sum(i * P)
    mu_j = np.sum(j * P)
    sigma_i = np.sqrt(np.sum((i - mu_i)**2 * P))
    sigma_j = np.sqrt(np.sum((j - mu_j)**2 * P))

    correlation = np.sum((i - mu_i)*(j - mu_j)*P) / (sigma_i * sigma_j + 1e-12)

    return dict(
        energy=energy,
        entropy=entropy,
        contrast=contrast,
        homogeneity=homogeneity,
        correlation=correlation
    )

features = {
    "I1 C[1,0]": texture_features(P10_I1),
    "I1 C[1,1]": texture_features(P11_I1),
    "I2 C[1,0]": texture_features(P10_I2),
    "I2 C[1,1]": texture_features(P11_I2),
}

for name, feat in features.items():
    st.subheader(name)
    st.json(feat)

# =====================================================
# VISUALIZATION
# =====================================================

st.divider()
st.header("📈 GLCM Visualization")

fig, axes = plt.subplots(2,2, figsize=(8,8))

mats = [P10_I1, P11_I1, P10_I2, P11_I2]
titles = ["I1 C[1,0]", "I1 C[1,1]", "I2 C[1,0]", "I2 C[1,1]"]

for ax, mat, title in zip(axes.flat, mats, titles):
    im = ax.imshow(mat)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

st.pyplot(fig)

# =====================================================
# FORMULAS
# =====================================================

st.divider()
st.header("📘 Feature Formulas")

st.latex(r"""
\text{Energy} = \sum_{i,j} P(i,j)^2
""")

st.latex(r"""
\text{Entropy} = -\sum_{i,j} P(i,j)\log_2 P(i,j)
""")

st.latex(r"""
\text{Contrast} = \sum_{i,j} (i-j)^2 P(i,j)
""")

st.latex(r"""
\text{Homogeneity} = \sum_{i,j} \frac{P(i,j)}{1 + |i-j|}
""")

st.latex(r"""
\text{Correlation} =
\frac{
\sum_{i,j} (i-\mu_i)(j-\mu_j) P(i,j)
}{
\sigma_i \sigma_j
}
""")

st.caption("🚀 GLCM texture feature extraction complete.")
