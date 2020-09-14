import numpy as np

def d_squared(a, b):
    # Takes two n-D tuples and returns square euclidean distance between them
    
    '''
    Inputs
    -----
    a - n-D tuple, array, or SymPy object
    b - n-D tuple, array, or SymPy object

    Returns
    -----
    Square euclidean distance between a and b
    '''

    # Cast to array for computation 
    # Cast first to tuple in case a or b are Sympy Point objects
    p_a = np.array(tuple(a), dtype='float')
    p_b = np.array(tuple(b), dtype='float')
    
    return np.sum(np.square(p_a - p_b))