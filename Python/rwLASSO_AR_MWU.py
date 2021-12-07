# Import libraries
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import statsmodels.api as sm
#from sklearn import linear_model

# Import my own modules
import sys
sys.path.insert(0, '..')
from manifolds import embed
from signals import *

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Configuration
Ntt = 200 # Test-train split
enable_detrend = False 
enable_normalize=True
enable_affine=False # Set to True if LASSO predictors can have a bias term

Q = 4 # Order of ARX Model
W =106 # Rolling window length (memory size)
eta = 0.7  # Learning rate for LASSO sub-ensemble weights
I = 12     # No. of LASSO estimators


# Import data
my_data = genfromtxt('../Data/natgas.data.csv', delimiter=',',dtype='f8')
my_data = my_data[1:,1:] # Ignore the first row, which contains headers

# Delete the unused features
X = np.delete(my_data,[1,2],axis=1)
X = maimpute(X)
X = normalize(X ,axis=0)
Xnames = ["STOCKS","JFKTEMP","CLTTEMP","ORDTEMP","HOUTEMP","LAXTEMP","NXT_CNG_STK"]

# Delay embed all of our data. The last column will be our target
Z = np.empty((X.shape[0]-Q+1,Q*X.shape[1] ))
for k in range(0,X.shape[1]):
    Z[:,np.arange(0,Q) + k*Q] = embed(X[:,k],Q,1)

# Separate target and predictors
y = Z[:,Z.shape[1]-1]  
Z = Z[:,0:Z.shape[1]-1]

# Normalize the input predictors
if enable_normalize:
    Z = normalize(Z, axis=0)

# Initialize matrix of predictions
NoModels = 1 # No. of models that I will have
Yp = 0*np.empty((y.shape[0],NoModels))




# <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>

# Decompose our guys into trend, seasonal and residuals
Zd = sm.tsa.seasonal_decompose(Z,model='additive',freq=52)
yd = sm.tsa.seasonal_decompose(y,model='additive',freq=52)


plt.figure(figsize=(16, 6))

# Seasonal
plt.subplot(3,1,1)
Zsea = np.array(Zd.seasonal)
ysea = yd.seasonal
ypp = mwubox(Zsea,ysea,W,I,enable_affine,eta)
t = np.transpose(np.arange(0,y.shape[0]))
tpp = np.transpose(np.arange(0,ypp.shape[0]))
plt.plot(t,ysea,tpp,ypp)
plt.title('LASSO sub-ensemble with MWU')
plt.ylabel('Seasonal',FontSize=14)
plt.legend(['Observed','LASSO ensemble'], loc='lower right')


# Residual
plt.subplot(3,1,3)
Zres = np.array(Zd.resid)
yres = yd.resid
ypp = mwubox(Zres,yres,W,I,enable_affine,eta)
t = np.transpose(np.arange(0,y.shape[0]))
tpp = np.transpose(np.arange(0,ypp.shape[0]))
plt.plot(t,yres,tpp,ypp)
plt.ylabel('Residual',FontSize=14)
plt.legend(['Observed','LASSO ensemble'], loc='lower right')


# Trend
plt.subplot(3,1,2)
Ztre = np.array(Zd.trend)
ytre = yd.trend
ypp = mwubox(Ztre,ytre,W,I,enable_affine,eta)
t = np.transpose(np.arange(0,y.shape[0]))
tpp = np.transpose(np.arange(0,ypp.shape[0]))
plt.plot(t,ytre,tpp,ypp)
plt.ylabel('Trend',FontSize=14)
plt.legend(['Observed','LASSO ensemble'], loc='lower right')



plt.show()

