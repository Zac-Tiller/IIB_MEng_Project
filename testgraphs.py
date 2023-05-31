import numpy as np
import matplotlib.pyplot as plt
# Create some sample data
num_particles = 100
num_timesteps = 100
x = np.random.normal(0, 1, size=(num_particles, num_timesteps))
y = np.random.normal(0, 1, size=(num_particles, num_timesteps))

# Create a figure and an axis
fig, ax = plt.subplots()

# Loop through each particle and create a line
for i in range(num_particles):
    # Create a line for the current particle
    line, = ax.plot(x[i], y[i], '-', linewidth=1, alpha=0.05, color='black')

# Create a 2D histogram of the lines
hist, xedges, yedges = np.histogram2d(x.flatten(), y.flatten(), bins=50, range=[[-5, 5], [-5, 5]])

# Convert the histogram to a density and transpose it
density = hist / np.max(hist)
density = density.T

# Plot the density as an image
im = ax.imshow(density, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='gray_r', alpha=0.5)

# Add a colorbar
fig.colorbar(im, ax=ax)

# Set the title and axis labels
ax.set_title('Particle Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Show the final figure
plt.show()


x = [0,1,2,3,4,5,6,7,8,9,10]

l1 = [1,1,2,3,4,5,6,6,6,6,6]

l2 = [0,0,2,7,3,5,5,5,6,6,6]

l3 = [1,1,2,3,4,5,5,9,6,6,6]

l4 = [0,0,0,0,0,5,5,9,6,6,6]

fig, ax = plt.subplots()

ax.plot(x, l1, alpha=0.05, color='blue')
ax.plot(x, l2, alpha=0.05, color='blue')
ax.plot(x, l3, alpha=0.05, color='blue')
ax.plot(x, l4, alpha=0.05, color='blue')
plt.show()



