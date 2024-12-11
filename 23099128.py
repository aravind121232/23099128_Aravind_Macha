#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Load the dataset
file_name = "airline2.csv"
data = pd.read_csv(file_name)

# Ensure the 'Date' column is in datetime format and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Extract necessary columns
passenger_counts = data['Number'].values  # Passenger numbers from 'Number' column
dates = data.index
N = len(passenger_counts)
dt = (dates[1] - dates[0]).days  # Assumes daily data

# (A) Fourier Transform of daily passenger numbers
y_fft = fft(passenger_counts)
power = np.abs(y_fft[:N // 2]) ** 2  # Power spectrum
freqs = fftfreq(N, d=dt)[:N // 2]    # Frequency values (cycles per day)

# Find the main period (highest power)
max_power_index = np.argmax(power[1:]) + 1  # Exclude 0-frequency
main_period = 1 / freqs[max_power_index]    # Convert frequency to period

# Monthly and Yearly grouping
data['Year'] = data.index.year  # Extract year from 'Date'

# (B) Calculate average daily passengers per month
data['Month'] = data.index.month  # Extract month from 'Date'
avg_monthly = data.groupby('Month')['Number'].mean()

# Plot 1: Fourier series and monthly averages
plt.figure(figsize=(12, 6))

# Monthly averages as a bar chart
plt.bar(avg_monthly.index, avg_monthly, color='blue', label='Average Daily Passengers per Month')

# Fourier series approximation
terms = 8  # Number of terms for Fourier series
approx = np.zeros(N)
for k in range(terms):
    coeff = y_fft[k]
    approx += (coeff.real * np.cos(2 * np.pi * freqs[k] * np.arange(N) * dt) -
               coeff.imag * np.sin(2 * np.pi * freqs[k] * np.arange(N) * dt)) / N

plt.plot(range(1, 13), approx[:12], color='red', label='Fourier Approximation (8 terms)')
# Add student ID annotation to the plot
student_id = "23099128"
plt.text(0.05, 0.95, f"Student ID: {student_id}", transform=plt.gca().transAxes, fontsize=18, color='green', verticalalignment='bottom')

# Finalize plot 1
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.title('Average Monthly Passengers and Fourier Approximation')
plt.legend()
plt.grid()
plt.savefig("Figure1.png")
plt.show()

# (C) Power spectrum plot
plt.figure(figsize=(12, 6))

# Power spectrum
plt.plot(1 / freqs[1:], power[1:], label='Power Spectrum', color='purple')  # Ignore 0-frequency
plt.xscale('log')
plt.xlabel('Period (days)')
plt.ylabel('Power')
plt.title('Power Spectrum of Daily Passenger Variations')
plt.grid()

# Annotate main period
plt.annotate(f'Main Period = {main_period:.2f} days', xy=(0.6, 0.9), xycoords='axes fraction', fontsize=12, color='blue')
# Add student ID annotation to the power spectrum plot
plt.text(0.05, 0.95, f"Student ID: {student_id}", transform=plt.gca().transAxes, fontsize=12, color='black', verticalalignment='top')

# Save and display plot 2
plt.savefig("Figure2.png")
plt.legend()
plt.show()

# (D) Total passengers in 2022
passengers_2022 = data[data['Year'] == 2022]['Number'].sum()

# Print calculated values
print(f"Total passengers in 2022 (X): {passengers_2022}")
print(f"Main period in days (Y): {main_period:.2f}")

