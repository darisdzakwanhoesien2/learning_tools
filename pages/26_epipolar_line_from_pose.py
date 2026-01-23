import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Epipolar Line from Camera Pose",
    layout="wide"
)
st.title("📐 Epipolar Line from Rotation and Translation")

# =====================================================
# GIVEN ROTATION AND TRANSLATION
# =====================================================
R = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0]
], dtype=float)

t = np.array([1.0, 1.0, 1.0])

# Homogeneous image points
p  = np.array([0.0, 0.0, 1.0])          # principal point in image 1
p2 = np.array([0.5, 0.5, 1.0])          # candidate point in image 2

# =====================================================
# DISPLAY GIVEN DATA
# =====================================================
st.header("📥 Given Pose and Image Points")

st.subheader("Rotation R")
st.write(R)

st.subheader("Translation t")
st.write(t)

st.subheader("Image Points")
st.write("p  =", p)
st.write("p' =", p2)

# =====================================================
# HELPERS
# =====================================================
def skew(v):
    """Skew-symmetric matrix for cross product."""
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0  ]
    ], dtype=float)

# =====================================================
# PART (a) — ESSENTIAL MATRIX
# =====================================================
st.divider()
st.header("🧮 (a) Essential Matrix and Epipolar Line")

t_x = skew(t)
E = t_x @ R

st.subheader("Skew Matrix [t]×")
st.write(t_x)

st.subheader("Essential Matrix E = [t]× R")
st.write(E)

# Epipolar line in image 2
l = E @ p

st.subheader("Epipolar Line ℓ' = E p")
st.write("Line coefficients (a, b, c):")
st.write(l)

# =====================================================
# PART (b) — EPIPOLAR CONSTRAINT CHECK
# =====================================================
st.divider()
st.header("🔍 (b) Epipolar Constraint Check")

val = float(p2.T @ l)

st.write("Epipolar constraint value  p'^T ℓ' =")
st.code(round(val, 6))

if abs(val) < 1e-6:
    st.success("✅ The point p' satisfies the epipolar constraint.")
else:
    st.error("❌ The point p' does NOT satisfy the epipolar constraint.")

# =====================================================
# VISUALIZATION
# =====================================================
st.divider()
st.header("📊 Epipolar Line Visualization")

a, b, c = l

xs = np.linspace(-2, 2, 300)

if abs(b) < 1e-9:
    st.warning("Epipolar line is vertical — cannot plot y(x).")
else:
    ys = -(a * xs + c) / b

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(xs, ys, linewidth=2, label="Epipolar line ℓ'")
    ax.scatter(p2[0], p2[1], s=120, marker="x", label="Point p'")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Epipolar Line in Image 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

# =====================================================
# EXPLANATION
# =====================================================
st.divider()
st.header("🧠 Interpretation")

st.markdown("""
### Essential Matrix
The essential matrix encodes the relative camera pose:

E = [t]× R

It maps a point in the first image to its epipolar line in the second image.

---

### Epipolar Line
The epipolar line ℓ' is computed as:

ℓ' = E p

Any corresponding point p' must lie on this line.

---

### Epipolar Constraint
Correspondence must satisfy:

p'^T ℓ' = 0

If this value is close to zero, the match is geometrically consistent.

---

### Practical Use
- Reduces correspondence search from 2D → 1D
- Used in stereo matching, SLAM, structure-from-motion
""")

# =====================================================
# OPTIONAL INTERACTIVE TEST
# =====================================================
st.divider()
st.header("🎛 Test Another Candidate Point")

tx = st.slider("p'.x", -2.0, 2.0, float(p2[0]), 0.05)
ty = st.slider("p'.y", -2.0, 2.0, float(p2[1]), 0.05)

p_test = np.array([tx, ty, 1.0])
val_test = float(p_test.T @ l)

st.write("Test point:", p_test)
st.write("Constraint value:", round(val_test, 6))

if abs(val_test) < 1e-6:
    st.success("✅ This point satisfies the epipolar constraint.")
else:
    st.warning("⚠️ This point violates the epipolar constraint.")

st.caption("🚀 Epipolar geometry computation from camera rotation and translation.")
