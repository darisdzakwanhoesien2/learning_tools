import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_surface(f, x_range=(-5, 5), y_range=(-5, 5), resolution=100):
    """Plot a 3D surface of the function f(x1, x2)."""
    x1 = np.linspace(*x_range, resolution)
    x2 = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x1, x2)
    Z = f(X, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.title("3D Surface Plot")
    return fig

def plot_contour(f, x_range=(-5, 5), y_range=(-5, 5), resolution=100):
    """Plot a contour of the function f(x1, x2)."""
    x1 = np.linspace(*x_range, resolution)
    x2 = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x1, x2)
    Z = f(X, Y)

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.title("Contour Plot")
    return fig
