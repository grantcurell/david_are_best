import numpy as np
import matplotlib.pyplot as plt
import csv

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
        if step % (steps // 10) == 0 or step == steps - 1:  # Also save data on the last step
            energy = compute_energy(state)
            magnetization = compute_magnetization(state)
            energies.append(energy)
            magnetizations.append(magnetization)
            print(f"Progress: {step/steps*100:.2f}%")  # Print progress
    return state, energies, magnetizations

N = 10
dims = 3
steps = 5000
temperatures = np.linspace(1.5, 2.5, 10)
average_magnetizations = []
std_dev_magnetizations = []
collected_states = []
collected_magnetizations = []

for T_idx, T in enumerate(temperatures):
    print(f"Simulating temperature {T:.2f}")
    final_state, energies, magnetizations = ising_model_monte_carlo(N, dims, steps, T)
    average_magnetization = np.mean(magnetizations) / N**dims
    std_dev_magnetization = np.std(magnetizations, ddof=1) / np.sqrt(len(magnetizations)) / N**dims
    average_magnetizations.append(average_magnetization)
    std_dev_magnetizations.append(std_dev_magnetization)
    collected_states.append(final_state)  # Store the final state
    collected_magnetizations.extend(magnetizations)

# Plotting the chart
plt.errorbar(temperatures, average_magnetizations, yerr=std_dev_magnetizations, fmt='-o', label='Magnetization')
plt.xlabel('Temperature (T)')
plt.ylabel('Average Magnetization per Spin')
plt.title('3D Ising Model - Monte Carlo Simulation')
plt.legend()
plt.show()

# Exporting data to CSV
with open('ising_model_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['state_' + str(i) for i in range(N**dims)] + ['magnetization'])
    for state, magnetization in zip(collected_states, collected_magnetizations):
        writer.writerow(list(state.flatten()) + [magnetization])

print("Data export complete.")
