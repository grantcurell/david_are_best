import numpy as np
import matplotlib.pyplot as plt

def initial_state(N, dims):
    return np.random.choice([-1, 1], size=(N,)*dims)

def compute_energy(state):
    energy = 0
    for axis in range(len(state.shape)):
        rolled_state = np.roll(state, 1, axis=axis)
        interaction = np.multiply(state, rolled_state)
        energy -= np.sum(interaction)
    return energy

def compute_magnetization(state):
    magnetization = np.sum(state)
    return magnetization

def monte_carlo_step(state, beta):
    N = state.shape[0]
    for index in np.ndindex(state.shape):
        current_spin = state[index]
        neighbor_sum = 0
        for i in range(len(state.shape)):
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
    for step in range(steps):
        state = monte_carlo_step(state, beta)
        if step % 100 == 0:
            energy = compute_energy(state)
            magnetization = compute_magnetization(state)
            energies.append(energy)
            magnetizations.append(magnetization)
        # Print progress every 100 steps as a percentage
        if step % (steps // 100) == 0:
            print(f"Progress at T={T:.2f}: {100 * step / steps:.0f}%")
    return energies, magnetizations

N = 10
dims = 2
steps = 100000
temperatures = np.linspace(2.3, 2.3, 1)
average_magnetizations = []
std_dev_magnetizations = []
threshold = 0.05


for i, T in enumerate(temperatures):
    _, magnetizations = ising_model_monte_carlo(N, dims, steps, T)
    average_magnetization = np.mean(magnetizations) / N**dims
    std_dev_magnetization = np.std(magnetizations, ddof=1) / np.sqrt(len(magnetizations)) / N**dims
    average_magnetizations.append(average_magnetization)
    std_dev_magnetizations.append(std_dev_magnetization)
    print("Average Magnetization: ", average_magnetization)
    print("Standard Deviation: ", std_dev_magnetization)
    # Print progress for temperature steps
    print(f"Temperature progress: {100 * (i + 1) / len(temperatures):.0f}%")

plt.errorbar(temperatures, average_magnetizations, yerr=std_dev_magnetizations, fmt='-o', label='Magnetization')
plt.xlabel('Temperature (T)')
plt.ylabel('Average Magnetization per Spin')
plt.title('Classical 3D Heisenberg Model - Monte Carlo Simulation using Metropolis Algorithm')
plt.legend()
plt.show()
