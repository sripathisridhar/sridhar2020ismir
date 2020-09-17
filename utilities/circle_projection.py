import numpy as np

def circle_projection(xy_center, radius, xy_coords):
    '''
    This functions return coordinates of points projected onto given circle
    
    Inputs
    -----
    xy_center : (x,y) coordinates of center of the circle; 
    radius : radius of circle;
    xy_coords : (m,2) array of m points to be projected;

    Returns
    -----
    xy_prime : (m,2) array of m projected coordinates
    '''
    vectors = xy_coords - np.transpose(xy_center)
    azimuths = np.arctan2(vectors[:, 1], (1e-6 + vectors[:, 0])) 
    xy_prime = np.asarray([xy_center[0] + radius * np.cos(azimuths),
                         xy_center[1] + radius * np.sin(azimuths)])

    return np.transpose(xy_prime)