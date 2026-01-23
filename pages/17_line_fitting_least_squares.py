import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Least Squares Line Fitting",
    layout="wide"
)
st.title("📐 Line Fitting using Least Squares (Normal Equations)")

# =====================================================
# GIVEN DATA
# =====================================================
points = np.array([
    [0, -7],
    [2, -1],
    [4,  5]
], dtype=float)

x = points[:, 0]
y = points[:, 1]
n = len(points)

# =====================================================
# DISPLAY DATA
# =====================================================
st.header("📥 Observed Points")
st.write(points)

st.markdown("We fit a line:")
st.latex("y = c_1 x + c_0")

# =====================================================
# PART (a) — NORMAL EQUATIONS
# =====================================================
st.divider()
st.header("🧮 (a) Normal Equations")

st.markdown("The normal equations can be written as:")

st.latex(
    r"\begin{bmatrix}"
    r"\sum x_j^2 & \sum x_j\\"
    r"\sum x_j   & n"
    r"\end{bmatrix}"
    r"\begin{bmatrix} c_1 \\ c_0 \end{bmatrix}"
    r"="
    r"\begin{bmatrix}"
    r"\sum x_j y_j \\ \sum y_j"
    r"\end{bmatrix}"
)

# Compute sums
Sx  = np.sum(x)
Sy  = np.sum(y)
Sxx = np.sum(x * x)
Sxy = np.sum(x * y)

st.subheader("Computed Sums")
st.write("Σ x =", Sx)
st.write("Σ y =", Sy)
st.write("Σ x² =", Sxx)
st.write("Σ x y =", Sxy)
st.write("n =", n)

# =====================================================
# PART (b) — SOLVE NORMAL EQUATIONS
# =====================================================
st.divider()
st.header("🧩 (b) Solve for c₁ and c₀")

A = np.array([
    [Sxx, Sx],
    [Sx,  n ]
], dtype=float)

b = np.array([
    Sxy,
    Sy
], dtype=float)

st.subheader("Matrix A")
st.write(A)

st.subheader("Vector b")
st.write(b)

# Solve linear system
c1, c0 = np.linalg.solve(A, b)

st.success("✅ Fitted Parameters")
st.write("c₁ =", round(c1, 4))
st.write("c₀ =", round(c0, 4))

st.markdown("Final fitted line:")
st.latex("y = " + f"{c1:.4f}" + r"x + " + f"{c0:.4f}")

# =====================================================
# PREDICTION AND ERROR
# =====================================================
st.divider()
st.header("📏 Prediction and Error")

y_pred = c1 * x + c0
errors = y_pred - y

st.write("Predicted y values:")
st.write(y_pred)

st.write("Residuals:")
st.write(errors)

mse = np.mean(errors**2)
st.write("Mean Squared Error:", round(mse, 6))

# =====================================================
# VISUALIZATION
# =====================================================
st.divider()
st.header("📊 Visualization")

xx = np.linspace(min(x)-1, max(x)+1, 200)
yy = c1 * xx + c0

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(x, y, s=120, label="Observed points")
ax.plot(xx, yy, linewidth=2, label="Least squares line")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Least Squares Line Fit")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# =====================================================
# INTERACTIVE EXPERIMENT
# =====================================================
st.divider()
st.header("🎛 Interactive Experiment")

st.markdown("Move the sliders to change the data points.")

x1 = st.slider("Point 1 x", -10.0, 10.0, float(points[0,0]), 0.5)
y1 = st.slider("Point 1 y", -10.0, 10.0, float(points[0,1]), 0.5)

x2 = st.slider("Point 2 x", -10.0, 10.0, float(points[1,0]), 0.5)
y2 = st.slider("Point 2 y", -10.0, 10.0, float(points[1,1]), 0.5)

x3 = st.slider("Point 3 x", -10.0, 10.0, float(points[2,0]), 0.5)
y3 = st.slider("Point 3 y", -10.0, 10.0, float(points[2,1]), 0.5)

points_new = np.array([[x1, y1], [x2, y2], [x3, y3]])
x_new = points_new[:,0]
y_new = points_new[:,1]

Sx  = np.sum(x_new)
Sy  = np.sum(y_new)
Sxx = np.sum(x_new * x_new)
Sxy = np.sum(x_new * y_new)

A_new = np.array([[Sxx, Sx],
                  [Sx,  3 ]], dtype=float)
b_new = np.array([Sxy, Sy], dtype=float)

c1_new, c0_new = np.linalg.solve(A_new, b_new)

st.subheader("Updated Line Parameters")
st.write("c₁ =", round(c1_new, 4))
st.write("c₀ =", round(c0_new, 4))

xx = np.linspace(min(x_new)-1, max(x_new)+1, 200)
yy = c1_new * xx + c0_new

fig2, ax2 = plt.subplots(figsize=(7, 6))
ax2.scatter(x_new, y_new, s=120, label="New points")
ax2.plot(xx, yy, linewidth=2, label="Updated fit")
ax2.set_title("Interactive Line Fit")
ax2.grid(True)
ax2.legend()

st.pyplot(fig2)

st.caption("🚀 Least squares line fitting using normal equations.")
