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

# Create the 2D heatmap
plt.figure(figsize=(8, 6))
plt.contourf(x1, x2, z, levels=50, cmap='viridis')

# Add color bar
plt.colorbar(label='f(x1, x2)')

# Add labels and title
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2D Heatmap of f(x1, x2)')

# Show the plot
plt.show()