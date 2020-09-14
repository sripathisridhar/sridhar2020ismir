import numpy as np

def centered(A, Q=24, J=3):
    # Returns centered distance matrix
    
    '''
    Inputs
    -----
    A - squared distance matrix
    Q - quality factor, 24 by default
    J - number of octaves, 3 by default
    
    Returns
    -----
    tau - MDS style diagonalized matrix of A
    '''
    
    coords_length = A.shape[0]
    H = np.zeros((coords_length, coords_length))

    const = 1/(Q*J)
    for i in range(coords_length):
        for j in range(coords_length):
            if j==i:
                H[i,j] = 1 - const
            else:
                H[i,j] = -const
                
    return -0.5 * np.matmul(np.matmul(H, A), H)

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