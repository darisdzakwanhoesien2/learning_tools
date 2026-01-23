import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Machine Vision - Question 3 Solver", layout="wide")

def solve_ex1_q3():
    st.title("Exercise 1: Pixel Coordinate Transformation")
    st.markdown("""
    This page calculates pixel coordinates $p = [u, v]^\top$ from normalized image coordinates $[x_n, y_n]^\top$.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Inputs")
        xn = st.number_input("Normalized x (xn)", value=0.05)
        yn = st.number_input("Normalized y (yn)", value=0.03)
        mu = st.number_input("Pixels per unit (mu)", value=100.0)
        mv = st.number_input("Pixels per unit (mv)", value=100.0)
        u0 = st.number_input("Principal point u0", value=320.0)
        v0 = st.number_input("Principal point v0", value=240.0)
        
        mode = st.radio("Coordinate System Type", ["Parallel Axes (3a)", "Skewed Axes (3b)"])
        theta = 90.0
        if mode == "Skewed Axes (3b)":
            theta = st.slider("Angle between u and v axes (degrees)", 10.0, 170.0, 80.0)

    # Calculation logic based on Source and
    theta_rad = np.radians(theta)
    
    if mode == "Parallel Axes (3a)":
        u = mu * xn + u0
        v = mv * yn + v0
        formula = r"u = m_u x_n + u_0, \quad v = m_v y_n + v_0"
    else:
        # Transformation with skew
        # u = mu*xn - mu*cot(theta)*yn + u0
        # v = (mv/sin(theta))*yn + v0
        u = mu * xn - mu * (1/np.tan(theta_rad)) * yn + u0
        v = (mv / np.sin(theta_rad)) * yn + v0
        formula = r"u = m_u x_n - m_u \cot(\theta) y_n + u_0, \quad v = \frac{m_v}{\sin\theta} y_n + v_0"

    with col2:
        st.subheader("Results")
        st.latex(formula)
        st.success(f"Pixel Coordinates: u = {u:.2f}, v = {v:.2f}")
        
        # Simple coordinate plot
        fig, ax = plt.subplots()
        ax.scatter(u, v, color='red', label='Projected Pixel')
        ax.axhline(v0, color='gray', linestyle='--', label='Principal Point Axes')
        ax.axvline(u0, color='gray', linestyle='--')
        ax.set_xlabel("u (pixels)")
        ax.set_ylabel("v (pixels)")
        ax.invert_yaxis() # Standard image coordinate orientation
        ax.legend()
        st.pyplot(fig)

def solve_ex5_q3():
    st.title("Exercise 5: Bayesian Decision Making")
    st.markdown("""
    Analysis of decision boundaries for Gaussian class conditional distributions.
    """)

    st.subheader("Mathematical Proof Summary")
    
    st.markdown("""
    **Posterior Probability:** The classifier assigns class $\omega_i$ that maximizes:
    """)
    st.latex(r"P(\omega_{i}|x)=\frac{p(x|\omega_{i})P(\omega_{i})}{\sum_{k=1}^{m}p(x|\omega_{k})P(\omega_{k})}")

    st.info("**Part (a): Quadratic Surface**")
    st.write("""
    When classes have different covariance matrices ($\Sigma_1 \neq \Sigma_2$), the log-likelihood 
    contains a quadratic term $x^\top \Sigma_i^{-1} x$. This results in a **quadratic decision boundary** 
    (ellipses, parabolas, or hyperbolas).
    """)

    st.info("**Part (b): Linear Hyperplane**")
    st.write("""
    If $\Sigma_1 = \Sigma_2 = \Sigma$, the quadratic terms $x^\top \Sigma^{-1} x$ in the discriminant 
    functions cancel out. The resulting boundary equation becomes linear in $x$:
    """)
    st.latex(r"w^\top x + w_0 = 0")
    st.write("This represents a **hyperplane** in d-dimensional space.")

    # Interactive Visualization
    st.subheader("Boundary Visualization Demo")
    cov_type = st.selectbox("Covariance Condition", ["Equal (Linear Boundary)", "Unequal (Quadratic Boundary)"])
    
    # Generate dummy data for visualization
    mean1, mean2 =,
    if cov_type == "Equal (Linear Boundary)":
        cov1 = cov2 = []
    else:
        cov1 = [[1, 0.5], [0.5, 1]]
        cov2 = [, [0, 0.5]]
    
    st.write(f"Class 1 Mean: {mean1}, Class 2 Mean: {mean2}")
    st.write(f"Class 1 Covariance: {cov1}")
    st.write(f"Class 2 Covariance: {cov2}")

# Navigation
page = st.sidebar.selectbox("Select Exercise", ["Ex 1: Pixel Coordinates", "Ex 5: Bayesian Decision"])

if page == "Ex 1: Pixel Coordinates":
    solve_ex1_q3()
else:
    solve_ex5_q3()