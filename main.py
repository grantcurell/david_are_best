import numpy as np
import matplotlib.pyplot as plt
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        energy = compute_energy(state)
        magnetization = compute_magnetization(state)
        energies.append(energy)
        magnetizations.append(magnetization)
        # Print progress
        print(f"Temperature {T:.2f}: {100.0 * (step + 1) / steps:.2f}% complete.")
    return state, energies, magnetizations


def simulation_wrapper(args):
    N, dims, steps, T = args
    final_state, energies, magnetizations = ising_model_monte_carlo(N, dims, steps, T)

    # Export data to CSV for each temperature
    filename = f'ising_model_T_{T:.2f}.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['state_' + str(i) for i in range(N ** dims)] + ['magnetization'])
        for magnetization in magnetizations:
            writer.writerow(list(final_state.flatten()) + [magnetization])

    # Return summary statistics for plotting
    average_magnetization = np.mean(magnetizations) / N ** dims
    std_dev_magnetization = np.std(magnetizations, ddof=1) / np.sqrt(len(magnetizations)) / N ** dims

    return T, average_magnetization, std_dev_magnetization

N = 60
dims = 2
steps = 200000
temperatures = np.linspace(1, 3, 10)

# Setting up the multiprocessing pool
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(simulation_wrapper, (N, dims, steps, T)) for T in temperatures]

    results = []
    for future in as_completed(futures):
        results.append(future.result())

# Sorting results by temperature for plotting
results.sort()

# Unpacking results
sorted_temperatures, average_magnetizations, std_dev_magnetizations = zip(*results)
abs_average_magnetizations = np.abs(average_magnetizations)

# Creating a subplot with 2 rows and 1 column
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# First subplot for Average Magnetization
axs[0].errorbar(sorted_temperatures, average_magnetizations, yerr=std_dev_magnetizations, fmt='-o', label='Magnetization')
axs[0].set_xlabel('Temperature (T)')
axs[0].set_ylabel('Average Magnetization per Spin')
axs[0].set_title('2D Ising Model - Monte Carlo Simulation')
axs[0].legend()

# Second subplot for Absolute Average Magnetization
axs[1].plot(sorted_temperatures, abs_average_magnetizations, '-o', label='|Magnetization|')
axs[1].set_xlabel('Temperature (T)')
axs[1].set_ylabel('Absolute Average Magnetization per Spin')
axs[1].legend()

# Adjusting the layout to prevent overlapping
plt.tight_layout()

# Display the plot
plt.show()
