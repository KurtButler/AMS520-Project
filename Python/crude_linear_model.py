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


plt.plot(my_data[:,4:9])
plt.title('Raw features')
plt.show()

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


cutoff = 13
fs = 52
trend = butter_lowpass_filtfilt(target, cutoff, fs)
for k in range(0,features.shape[1]):
    features[:,k] = features[:,k] - butter_lowpass_filtfilt(features[:,k], cutoff, fs)

# Plot the detrended version
target = target - trend

# ARMA
X = np.concatenate( (features[1:834,:], features[0:833,:]), axis=1 ) 
target.shape = (target.shape[0],1)
X = np.concatenate((X, target[0:833]),axis=1)
y = target[1:834]


plt.plot(target)
plt.title("Detrended signal") 
plt.show()

plt.plot(features)
plt.title("Detrended features") 
plt.show()



# Fit a crude linear model  target = features @ A
linmdl = np.linalg.lstsq(X, y, rcond=None)
targetpred = X @ linmdl[0] 

plt.figure(figsize=(2, 6))
plt.plot(target)
plt.plot(targetpred)
plt.title("Signal vs ARMA(1,2) (detrended)") 
plt.show()
