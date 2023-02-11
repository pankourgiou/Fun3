import numpy as np
from scipy.linalg import eigh

# Define the number of grid points
N = 1000

# Define the grid
x = np.linspace(-10, 10, N)
dx = x[1] - x[0]

# Define the potential
V = 0.5 * x**2

# Define the Hamiltonian matrix
H = -0.5 * (np.diag(np.ones(N-1), -1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), 1)) / dx**2 + np.diag(V)

# Diagonalize the Hamiltonian to obtain the eigenvalues and eigenvectors
eigvals, eigvecs = eigh(H)

# Normalize the eigenvectors
eigvecs /= np.sqrt(dx)

# Select the ground-state wavefunction
psi_0 = eigvecs[:, 0]

# Plot the wavefunction
import matplotlib.pyplot as plt
plt.plot(x, psi_0**2)
plt.xlabel('x')
plt.ylabel(r'$|\psi(x)|^2$')
plt.show()
