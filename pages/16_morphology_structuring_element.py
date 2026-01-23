import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Binary Morphology Explorer", layout="wide")
st.title("🧩 Structuring Element — Binary Morphology")

# =====================================================
# INPUT BINARY IMAGE B (from problem)
# =====================================================

B = np.array([
    [0,0,0,0,0,0,0,1],
    [0,1,1,1,1,1,1,1],
    [0,1,0,1,1,1,1,1],
    [0,0,0,1,1,1,1,0],
    [0,0,1,1,1,1,1,1],
    [0,0,1,0,0,0,1,1],
    [0,0,1,0,0,0,1,1],
    [0,0,1,0,0,0,1,1],
], dtype=np.uint8)

# Structuring element S (3x3 ones)
S = np.ones((3,3), dtype=np.uint8)

# =====================================================
# DISPLAY INPUTS
# =====================================================

st.header("📥 Input Binary Image B and Structuring Element S")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Binary Image B")
    st.dataframe(B)

with col2:
    st.subheader("Structuring Element S")
    st.dataframe(S)

# =====================================================
# MORPHOLOGY OPERATORS (PURE NUMPY)
# =====================================================

def pad_image(img, pad):
    return np.pad(img, pad, mode="constant", constant_values=0)

def erosion(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = pad_image(img, (pad_h, pad_w))
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+kh, j:j+kw]
            out[i,j] = 1 if np.all(window[kernel==1] == 1) else 0
    return out

def dilation(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = pad_image(img, (pad_h, pad_w))
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+kh, j:j+kw]
            out[i,j] = 1 if np.any(window[kernel==1] == 1) else 0
    return out

def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

# =====================================================
# COMPUTE RESULTS
# =====================================================

B_dilate  = dilation(B, S)
B_erode   = erosion(B, S)
B_open    = opening(B, S)
B_close   = closing(B, S)

# =====================================================
# DISPLAY RESULTS
# =====================================================

st.divider()
st.header("🧮 Morphological Results")

def show_result(title, mat):
    st.subheader(title)
    st.dataframe(mat)

col1, col2 = st.columns(2)

with col1:
    show_result("Dilation  B ⊕ S", B_dilate)
    show_result("Opening  B ∘ S", B_open)

with col2:
    show_result("Erosion   B ⊖ S", B_erode)
    show_result("Closing   B • S", B_close)

# =====================================================
# VISUALIZATION
# =====================================================

st.divider()
st.header("📊 Visualization")

fig, axes = plt.subplots(1,5, figsize=(16,4))

images = [
    ("Original B", B),
    ("Dilation", B_dilate),
    ("Erosion", B_erode),
    ("Opening", B_open),
    ("Closing", B_close),
]

for ax, (title, img) in zip(axes, images):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

st.pyplot(fig)

# =====================================================
# THEORY EXPLANATION
# =====================================================

st.divider()
st.header("📘 Mathematical Definitions")

st.latex(r"""
\textbf{Dilation:} \quad B \oplus S = \{ x \mid (S)_x \cap B \neq \emptyset \}
""")

st.latex(r"""
\textbf{Erosion:} \quad B \ominus S = \{ x \mid (S)_x \subseteq B \}
""")

st.latex(r"""
\textbf{Opening:} \quad B \circ S = (B \ominus S) \oplus S
""")

st.latex(r"""
\textbf{Closing:} \quad B \bullet S = (B \oplus S) \ominus S
""")

st.markdown("""
### Intuition

- **Dilation** expands foreground regions.
- **Erosion** shrinks foreground regions.
- **Opening** removes small objects and noise.
- **Closing** fills small holes and gaps.
""")

st.caption("🚀 Fully reproducible binary morphology pipeline.")
