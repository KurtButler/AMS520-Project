# Import libraries
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import statsmodels.api as sm

# Import data
my_data = genfromtxt('../Data/natgas.data.csv', delimiter=',',dtype='f8')
X = my_data[1:,1:] # Ignore the first row, which contains headers
names = ["STOCKS","JFKTEMP","CLTTEMP","ORDTEMP","HOUTEMP","LAXTEMP","NXT_CNG_STK"]

# Import my own modules
import sys
sys.path.insert(0, '..')
from manifolds import embed
from signals import maimpute

# Delete the unused features
X = np.delete(X,[1,2],axis=1)
#X = maimpute(X)
X = normalize(X ,axis=0)
Xnames = ["STOCKS","JFKTEMP","CLTTEMP","ORDTEMP","HOUTEMP","LAXTEMP","NXT_CNG_STK"]



# Decompose our guy into trend, seasonal and residuals
Xd = sm.tsa.seasonal_decompose(X,model='additive',freq=52)

# Plot model fit
t = np.transpose(np.arange(0,X.shape[0]))
plt.figure(figsize=(16, 6))
plt.subplot(3,1,1)
plt.plot(t,Xd.seasonal)
plt.legend(names, loc='lower right')
plt.ylabel('Seasonal',FontSize=14)
plt.subplot(3,1,2)
plt.plot(t,Xd.trend)
plt.ylabel('Trend',FontSize=14)
plt.subplot(3,1,3)
plt.plot(t,Xd.resid)
plt.ylabel('Residual',FontSize=14)
plt.grid(True)
plt.show()
