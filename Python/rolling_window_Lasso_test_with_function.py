# Import libraries
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import statsmodels.api as sm
from sklearn import linear_model

# Import data
my_data = genfromtxt('../Data/natgas.data.csv', delimiter=',',dtype='f8')
X = my_data[1:,1:] # Ignore the first row, which contains headers
names = ["STOCKS","JFKTEMP","CLTTEMP","ORDTEMP","HOUTEMP","LAXTEMP","NXT_CNG_STK"]

# Import my own modules
import sys
sys.path.insert(0, '..')
from manifolds import embed
from signals import maimpute, lassobox

# Delete the unused features
X = np.delete(X,[1,2],axis=1)
X = maimpute(X)
X = normalize(X ,axis=0)
Xnames = ["STOCKS","JFKTEMP","CLTTEMP","ORDTEMP","HOUTEMP","LAXTEMP","NXT_CNG_STK"]



# Decompose our guy into trend, seasonal and residuals
Xd = sm.tsa.seasonal_decompose(X,model='additive',freq=52)

# Parameters
W =106 # Window size
eta = 0.9  # Learning rate for LASSO sub-ensemble weights
I = 10     # No. of LASSO estimators
biasEnable=False # Set intercept in LASSO linear model

plt.figure(figsize=(16, 6))

# Seasonal
plt.subplot(3,1,1)
X = np.array(Xd.seasonal[:,0:6])
y = Xd.seasonal[:,6]
ypp = lassobox(X,y,W,I,biasEnable,eta)
t = np.transpose(np.arange(0,y.shape[0]))
tpp = np.transpose(np.arange(0,ypp.shape[0]))
plt.plot(t,y,tpp,ypp)
plt.title('LASSO sub-ensemble')
plt.ylabel('Seasonal',FontSize=14)
plt.legend(['Observed','LASSO ensemble'], loc='lower right')


# Residual
plt.subplot(3,1,3)
X = np.array(Xd.resid[:,0:6])
y = Xd.resid[:,6]
ypp = lassobox(X,y,W,I,biasEnable,eta)
t = np.transpose(np.arange(0,y.shape[0]))
tpp = np.transpose(np.arange(0,ypp.shape[0]))
plt.plot(t,y,tpp,ypp)
plt.ylabel('Residual',FontSize=14)
plt.legend(['Observed','LASSO ensemble'], loc='lower right')


# Trend
plt.subplot(3,1,2)
X = np.array(Xd.trend[:,0:6])
y = Xd.trend[:,6]
ypp = lassobox(X,y,W,I,biasEnable,eta)
t = np.transpose(np.arange(0,y.shape[0]))
tpp = np.transpose(np.arange(0,ypp.shape[0]))
plt.plot(t,y,tpp,ypp)
plt.ylabel('Trend',FontSize=14)
plt.legend(['Observed','LASSO ensemble'], loc='lower right')



plt.show()

