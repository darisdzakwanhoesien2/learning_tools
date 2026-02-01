import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def laplace_solver(domain_size=10, boundary_conditions="dirichlet"):
    """Solve Laplace equation on a grid."""
    x = np.linspace(0, domain_size, 50)
    y = np.linspace(0, domain_size, 50)
    X, Y = np.meshgrid(x, y)

    # Example: Dirichlet BC (fixed potential at edges)
    phi = np.zeros_like(X)
    for i in range(len(y)):
        phi[:, i] = 1.0  # Top edge
        phi[:, -i-1] = 0.0  # Bottom edge

    return X, Y, phi

def plot_solution(X, Y, phi):
    """Plot the solution."""
    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, phi)
    st.pyplot(fig)

st.title("Laplace Equation Visualization")
domain_size = st.slider("Domain size (10-50)", 10, 50, 20)
bc_type = st.radio("Boundary Conditions", ["Dirichlet", "Neumann"])

X, Y, phi = laplace_solver(domain_size, bc_type)
plot_solution(X, Y, phi)
