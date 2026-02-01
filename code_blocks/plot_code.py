import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x1, x2):
    return 2 * x1**3 - 6 * x1 * x2 + 3 * x2**2

# Generate x1 and x2 values
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
x1, x2 = np.meshgrid(x1, x2)

# Calculate f(x1, x2)
z = f(x1, x2)

# Create the plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(x1, x2, z, cmap='viridis', edgecolor='k', alpha=0.8)

# Add labels and title
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('3D Surface Plot of f(x1, x2)')

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=10)

# Show the plot
plt.show()