# Import libraries
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

# Import data
my_data = genfromtxt('../Data/natgas.data.csv', delimiter=',',dtype='f8')
my_data = my_data[1:,]
target = my_data[:,9]


# Deseasonalization using Savitzky-Golay (LOESS)
# Code from Stack Overflow
# https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


cutoff = 4
fs = 52
trend = butter_lowpass_filtfilt(target, cutoff, fs)

# Plot the original signal and trend
plt.plot(target)
plt.plot(trend)
plt.title("Signal and trend") 
plt.show()

# Plot the detrended version
detrended = target - trend

plt.plot(detrended)
plt.title("Detrended signal") 
plt.show()