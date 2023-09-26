import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def read_file(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file]
    return data


data_sun_early = read_file('results_sun_early.txt')
data_sun_late = read_file('results_sun_late.txt')
data_rain_early = read_file('results_rain_early.txt')
data_rain_late = read_file('results_rain_late.txt')

data_sun = data_sun_late + data_sun_early
data_rain = data_rain_early + data_rain_late

data_late = data_sun_late + data_rain_late
data_early = data_rain_early + data_sun_early

kde1 = gaussian_kde(data_sun)
kde2 = gaussian_kde(data_rain)

x_range = np.linspace(min(min(data_sun), min(data_rain)), max(max(data_sun), max(data_rain)), 1000)

# Plot the KDEs
plt.figure(figsize=(10, 6))
plt.plot(x_range, kde1(x_range), color='orange', label='sun', alpha=1)
plt.fill_between(x_range, kde1(x_range), color='orange', alpha=0.5)
plt.plot(x_range, kde2(x_range), color='purple', label='rain', alpha=0.5)
plt.fill_between(x_range, kde2(x_range), color='purple', alpha=0.3)

# Add legend and display the plot
plt.legend()
plt.title("Overlay of Distributions")
plt.xlabel("Value")
plt.ylabel("Density")

# # Create the histograms
# # plt.hist(data_sun, bins=50, alpha=0.5, label='sun', color='red')
# # plt.hist(data_rain, bins=50, alpha=0.5, label='rain', color='blue')
# plt.hist(data_early, bins=50, alpha=0.5, label='early', color='grey')
# plt.hist(data_late, bins=50, alpha=0.5, label='late', color='darkorange')
#
# # Add labels and legend
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.legend(loc='upper right')

# Display the plot
plt.show()
