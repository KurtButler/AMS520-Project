import torch
import torch.nn as nn
import torch.nn.functional as F
# Import libraries
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from numpy import linalg as lin


# Torch neural net boilerplate 
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

