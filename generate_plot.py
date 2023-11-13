import matplotlib.pyplot as plt
import pandas as pd

# Load the data from a CSV file
data = pd.read_csv('h2o_ising_model_predictions.csv')

# Group the data by temperature and calculate the mean of the absolute values of the magnetizations
grouped_data = data.groupby('temperature').agg(lambda x: abs(x).mean()).reset_index()

# Set up the figure and axes for two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot the average of the absolute actual magnetization on the first subplot
ax1.plot(grouped_data['temperature'], grouped_data['magnetization (actual)'],
         label='Average of Absolute Actual Magnetization', color='blue')
ax1.set_xlabel('Temperature (T)')
ax1.set_ylabel('Average of Absolute Actual Magnetization')
ax1.set_title('Average of Absolute Actual Magnetization by Temperature')
ax1.legend()
ax1.grid(True)

# Plot the average of the absolute estimated magnetization on the second subplot
ax2.plot(grouped_data['temperature'], grouped_data['magnetization'],
         label='Average of Absolute Estimated Magnetization', color='orange')
ax2.set_xlabel('Temperature (T)')
ax2.set_ylabel('Average of Absolute Estimated Magnetization')
ax2.set_title('Average of Absolute Estimated Magnetization by Temperature')
ax2.legend()
ax2.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
