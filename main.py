import numpy as np
import matplotlib.pyplot as plt

def initial_state(N, dims):
    return np.random.choice([-1, 1], size=(N,)*dims)

def compute_energy(state):
    # Initialize the total energy of the system to zero.
    # In the Ising model, the energy is the sum of interactions between neighboring spins.
    energy = 0

    # Iterate over each dimension (axis) of the lattice.
    # In a 2D lattice, this would loop over the two axes (x and y).
    for axis in range(len(state.shape)):

        # Roll the lattice along the current axis to align each spin with its neighbor.
        # Periodic boundary conditions are applied, meaning the grid is conceptually
        # wrapped onto itself so that each edge is connected to the opposite edge.
        # This rolling simulates the interaction of a spin with its immediate neighbor.
        rolled_state = np.roll(state, 1, axis=axis)

        # Multiply the original state with the rolled state.
        # Since the Ising model considers the energy contribution from neighboring spins,
        # this multiplication will pair each spin with its neighbor along the current axis.
        interaction = np.multiply(state, rolled_state)

        # Sum the interactions and subtract from the total energy.
        # The Ising model defines energy as negative for aligned spins (favorable configuration).
        # The np.sum aggregates the interaction energy over the entire lattice for the current axis.
        # As each pair of neighboring spins contribute to the energy, we sum over all pairs.
        energy -= np.sum(interaction)

    # Return the computed total energy of the spin configuration.
    # This energy represents the sum of pairwise interactions over the entire lattice,
    # considering all dimensions, and correctly implementing periodic boundary conditions.
    return energy


def compute_magnetization(state):
    magnetization = np.sum(state)
    return magnetization


def monte_carlo_step(state, beta):
    N = state.shape[0]  # Assuming a square lattice for simplicity
    for index in np.ndindex(state.shape):
        current_spin = state[index]
        neighbor_sum = 0
        for i in range(len(state.shape)):
            # Use modulo N to wrap the index for periodic boundary conditions
            for offset in [-1, 1]:
                neighbor_index = list(index)
                neighbor_index[i] = (neighbor_index[i] + offset) % N
                neighbor_sum += state[tuple(neighbor_index)]

        delta_E = 2 * current_spin * neighbor_sum

        if delta_E < 0.0 or np.random.rand() < np.exp(-delta_E * beta):
            state[index] = -current_spin
    return state


def ising_model_monte_carlo(N, dims, steps, T):
    state = initial_state(N, dims)
    beta = 1.0 / T
    energies = []
    magnetizations = []

    # Each step will generate a random configuration
    for step in range(steps):
        state = monte_carlo_step(state, beta)
        if step % 100 == 0:  # Sample every 100 steps
            energy = compute_energy(state)
            magnetization = compute_magnetization(state)
            energies.append(energy)
            magnetizations.append(magnetization)

    return energies, magnetizations

# Simulation parameters
N = 10  # Linear dimension of the lattice
dims = 2  # Dimensionality of the lattice
steps = 10000  # Number of Monte Carlo steps
T = 2.5  # Temperature in units where Boltzmann constant k_B = 1

energies, magnetizations = ising_model_monte_carlo(N, dims, steps, T)

print("Energies:", energies)
print("Magnetizations:", magnetizations)

# Simulation parameters
N = 10  # Linear dimension of the lattice
dims = 2  # Dimensionality of the lattice
steps = 100000  # Number of Monte Carlo steps
T = 2.5  # Temperature in units where Boltzmann constant k_B = 1

energies, magnetizations = ising_model_monte_carlo(N, dims, steps, T)

# Plotting the energy of the system as a function of Monte Carlo step
plt.figure(figsize=(10, 5))
plt.plot(range(0, steps, 100), energies, label='Energy')
plt.xlabel('Monte Carlo step')
plt.ylabel('Energy')
plt.title('Energy vs. Monte Carlo Step')
plt.legend()
plt.show()

# Plotting the magnetization of the system as a function of Monte Carlo step
plt.figure(figsize=(10, 5))
plt.plot(range(0, steps, 100), magnetizations, label='Magnetization')
plt.xlabel('Monte Carlo step')
plt.ylabel('Magnetization')
plt.title('Magnetization vs. Monte Carlo Step')
plt.legend()
plt.show()

# Assuming 'magnetizations' is your array of magnetization values

average_magnetization = np.mean(magnetizations)
std_dev_magnetization = np.std(magnetizations, ddof=1)/np.sqrt(len(magnetizations))

print("Average Magnetization:", average_magnetization)
print("Standard Deviation of Magnetization:", std_dev_magnetization)
