import numpy as np

def frobenius_distance(A, B):
    # Given two nxn matrices, return the 'Frobenius distance' between them

    '''
    Inputs
    -----
    A - matrix A with dimensions n*n
    B - matrix B with dimensions n*n

    Returns
    -----
    Frobenius distance between A and B
    '''
    return np.sqrt(np.sum(np.square(A - B)))