# Import libraries
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from numpy import linalg as lin

# Import data
my_data = genfromtxt('../Data/natgas.data.csv', delimiter=',',dtype='f8')
my_data = my_data[1:,]
features=my_data[:,4:9]
target = my_data[:,9]


#plt.plot(my_data[:,4:9])
#plt.title('Raw features')
#plt.show()

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
for k in range(0,features.shape[1]):
    features[:,k] = features[:,k] - butter_lowpass_filtfilt(features[:,k], cutoff, fs)

# Plot the detrended version
target = target - trend

plt.plot(target)
plt.title("Detrended signal") 
plt.show()

plt.plot(features)
plt.title("Detrended features") 
plt.show()



# Fit a crude linear model  target = features @ A
linmdl = np.linalg.lstsq(features, target, rcond=None)
targetpred = features @ linmdl[0] 

plt.figure(figsize=(18, 8))
plt.plot(target)
plt.plot(targetpred)
plt.title("Signal vs LM of other features") 
plt.show()
