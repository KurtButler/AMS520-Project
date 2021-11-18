import numpy as np
from sklearn import linear_model
# Functions useful for this forecasting project

def maimpute(my_data):
    """
    This is a simple moving-average (MA) NaN-imputer.

    :param my_data: input signals, arranged as a TxD matrix (numpy array)
    :returns: my_data with NaNs in each column smoothed over
    """
    for p in range(0,my_data.shape[1]):
        nanlist = np.argwhere(np.isnan(my_data[:,p]))
        if nanlist.size > 0:
            #Since I know that no NaN occur near n=0, I can take moving-averages without boundary conditions.
            for k in range(0,nanlist.size):
                my_data[nanlist[k],p] = np.mean(my_data[int(nanlist[k]-1):int(nanlist[k]),p])
    return my_data

def lassobox(X,y,W,I,biasEnable,eta):
    """
    Wrapper script for the rolling-window LASSO sub-ensemble predictor

    :param X: predictor variables, arranged as a TxD numpy array
    :param y: target variables, arranged as a T dimensional numpy array
    :param W: Rolling window length (int)
    :param I: Number of LASSO predictors in the ensemble
    :param biasEnable: Boolean, True if LASSO models have offset
    :param eta: Learning rate of ensemble weight updates. Float in (0,1).
    :returns: ypp, the NaN-padded predictions of the model
    """
    D = X.shape[1]
    N = X.shape[0]

    # Init window
    XX = X[26:int(26+W),:]
    yy = y[26:int(26+W)]
    gmax = np.nanmax(yy @ XX)
    g = np.exp(np.linspace( np.log(1e-8),np.log(gmax),I))
    w = np.ones(g.shape)/I
    yp = np.zeros((y.shape[0],I))
    ypp = np.zeros((y.shape[0]))*(np.nan)
    wu = np.zeros((I))
    cost = np.zeros((I))
    
    
    # Rolling windows
    for n in range(27+W,N-W):
        XX = X[n:int(n+W),:]
        yy = y[n:int(n+W)]
        if np.isnan(X[n+W,:]).any() or np.isnan(y[n+W]):
            break
        
        # LASSO predictors
        for i in range(0,I):
            LASSOmdl = linear_model.Lasso(alpha=g[i],max_iter=1000, fit_intercept=biasEnable)
            LASSOmdl.fit(XX, yy)
            yp[n+W,i] = X[int(n+W),:] @ LASSOmdl.coef_
            
        # Weight update
        for i in range(0,I):
            cost[i] = np.nanmean( np.square(yp[(n-W):n,i] - y[(n-W):n] ))
        cost = cost/ np.sum(cost)
        for i in range(0,I):
            wu[i] = w[i]*(1 - eta*cost[i]) 
        w = wu / np.sum(wu)
        
        # Forecast
        ypp[n+W] = yp[n+W,:] @ w
        
    return ypp