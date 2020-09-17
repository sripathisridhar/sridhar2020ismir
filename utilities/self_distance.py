import numpy as np
from utilities.d_squared import d_squared

def self_distance(coords):
    '''
    Returns matrix of distances between points in coords

    Inputs
    -----
    coords - 2-D numpy array with each point occupying a row, dimensions along columns

    Returns
    -----
    D_self - Upper triangular self distance matrix 
    '''
    coords_length = coords.shape[0]
    D_self = np.zeros((coords_length, coords_length))
    
    for i in range(coords_length):
        for j in range(i, coords_length):
            D_self[i,j] = d_squared(coords[i,:], coords[j,:])

    return D_self