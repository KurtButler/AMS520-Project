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
from signals import maimpute

# Delete the unused features
X = np.delete(X,[1,2],axis=1)
X = maimpute(X)
X = normalize(X ,axis=0)
Xnames = ["STOCKS","JFKTEMP","CLTTEMP","ORDTEMP","HOUTEMP","LAXTEMP","NXT_CNG_STK"]



# Decompose our guy into trend, seasonal and residuals
Xd = sm.tsa.seasonal_decompose(X,model='additive',freq=52)


# For now, my signals will not be AR
X = np.array(Xd.seasonal[:,0:6])
y = Xd.seasonal[:,6]
#X = np.array(Xd.resid[:,0:6])
#y = Xd.resid[:,6]

D = X.shape[1]
N = X.shape[0]

# Parameters
W = D + 100 # Window size
eta = 0.5  # Learning rate for LASSO sub-ensemble weights
I = 10     # No. of LASSO estimators
biasEnable=False # Set intercept in LASSO linear model

# Init window
XX = X[26:int(26+W),:]
yy = y[26:int(26+W)]
gmax = np.nanmax(yy @ XX)
g = np.exp(np.linspace( np.log(1e-8),np.log(gmax),I))
w = np.ones(g.shape)/I
mdlcoeff = np.zeros((D,I))
yp = np.zeros((y.shape[0],I))
ypp = np.zeros((y.shape[0]))*np.nan
wu = np.zeros((I))
cost = np.zeros((I))


# Rolling windows
for n in range(27+W,N-W):
    XX = X[n:int(n+W),:]
    yy = y[n:int(n+W)]
    
    # LASSO predictors
    for i in range(0,I):
        LASSOmdl = linear_model.Lasso(alpha=g[i],max_iter=3000, fit_intercept=biasEnable)
        LASSOmdl.fit(XX, yy)
        yp[n+W,i] = X[int(n+W),:] @ LASSOmdl.coef_
        
    # Weight update
    for i in range(0,I):
        cost[i] = np.nanmean( np.square(yp[(n-W):n,i] - y[(n-W):n] ))
    cost = cost/ np.sum(cost)
    for i in range(0,I):
        wu[i] = w[i]*(1 - eta*cost[i]) 
    w = wu / np.sum(wu)
    
    # Lambda update
    gmax = np.nanmax(yy @ XX)
    g = np.exp(np.linspace( np.log(1e-8),np.log(gmax),I))
        
    # Forecast
    ypp[n+W] = yp[n+W,:] @ w

plt.figure(figsize=(16, 2))
t = np.transpose(np.arange(0,y.shape[0]))
tpp = np.transpose(np.arange(0,ypp.shape[0]))
plt.plot(t,y,tpp,ypp)
plt.title('LASSO sub-ensemble')
plt.legend(['Observed','LASSO ensemble'], loc='lower right')
plt.show()


# Plot model fit
t = np.transpose(np.arange(0,Xd.trend.shape[0]))
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
