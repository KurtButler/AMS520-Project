import numpy as np


def embed(x, Q, tau):
    """
    This is a simple delay embedding function for scalar signals.

    :param x: input time series, arranged as a Nx1 column matrix (numpy array)
    :param Q: embedding dimension (dimension of the output matrix)
    :param tau: embedding delay
    :returns: delay embedding matrix
    """
    N = x.shape[0]
    Mx = np.empty((N - (Q - 1) * tau, Q))
    for q in range(0, Q):
        Mx[:, q] = x[np.arange(0, N - (Q - 1) * tau) + q * tau]
    return Mx


def vembed(X, Q, tau):
    """
    This is a delay embedding function for vectorial signals.

    :param X: input time series, arranged as a Nx1 column matrix (numpy array)
    :param Q: embedding dimension (dimension of the output matrix)
    :param tau: embedding delay
    :returns: delay embedding matrix
    """
    # Vector delay embedding is just the concatenation of the scalar delay embeddings.
    MX = np.empty((X.shape[0] - Q + 1, Q * X.shape[1]))
    for k in range(0, X.shape[1]):
        MX[:, np.arange(0, Q) + k * Q] = embed(X[:, k], Q, 1)
    return MX


def simplex(X, Y, Xp):
    """
    Computes the simplex projection estimates of Yp =F(Xp) given training data Y = F(X).

    Equivalently, it estimates Yp using k-nearest neighbor regression, where k=dimension(X)+1.

    :param X: numpy matrix whose rows are Q-dimensional vectors representing training input points
    :param Y: numpy matrix whose rows are R-dimensional vectors representing training output points
    :param Xp: numpy matrix whose rows are Q-dimensional vectors representing prediction input points
    :return: numpy matrix whose rows are R-dimensional vectors representing the simplex projection estimates of Yp=F(Xp)
    """
    # if X.shape[1] != Xp.shape[1]:
    #    raise ValueError("X and Xp must have the same number of columns")
    # if X.shape[0] != Y.shape[0]:
    #    raise ValueError("X and Y must have the same number of rows (training points)")
    k = X.shape[1] + 1
    Yp = np.zeros([Xp.shape[0], Y.shape[1]])
    for n in range(0, Xp.shape[0]):
        distances = np.sum((X - Xp[n,]) ** 2, axis=1)
        idx = np.argpartition(distances, k)
        distx = distances[idx]
        w = np.array(np.exp(-distx))
        w = w / np.sum(w)
        w.shape = [1, w.shape[0]]
        Yp[n,] = np.matmul(w, Y[idx,])
    return Yp


def ccm(x, y, Q, tau):
    #x.shape = [x.shape[0], 1]
    #y.shape = [y.shape[0], 1]
    Mx = vembed(x,Q,tau)
    My = vembed(y, Q, tau)
    mx = Mx[:,Mx.shape[1]-1]
    xp = np.zeros([My.shape[0],1])
    for n in range(1,xp.shape[0]):
        xp[n,1] = simplex(My[0:n,:],mx,My[n,:])

