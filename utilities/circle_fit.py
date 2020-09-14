import numpy as np

def circle_fit(A):
    
    '''
    This function returns a linear least squares estimate of a circle fit
    Points used must be stored in the columns of A as (x,y) coordinates
    
    Inputs
    -----
    A - matrix containing points stored as (x,y) coordinates in columns

    Returns
    -----
    x - Coordinates of estimated circle
    r - Radius of estimated circle
    residual - Residual error of the fit
    '''

    [n,m] = A.shape
    A_augmented = np.hstack((A.T, np.ones((m,1))))
    A_augmented.shape

    # Linear least squares fit estimate
    y, _,_,_ = np.linalg.lstsq(A_augmented, np.sum(np.multiply(A_augmented, A_augmented), axis=1).T, rcond=None)
    x = 0.5*y[:n]
    r = np.sqrt(y[n] + np.dot(x.T,x))

    # Residual
    residual = abs(np.sum((np.sum(np.square(np.expand_dims(x, axis=1) - A), axis=0) - r**2)))
    
    return x, r, residual

