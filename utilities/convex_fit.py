import numpy as np
import os 
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import ConvexHull

def convex_fit(xy_coords, n_iterations=500):
    '''
    This function returns a circle fit within the bounds of a convex hull, optimized using the Frank-Wolfe algorithm

    Inputs
    -----
    xy_coords - (m,2) array of m point coordinates in two dimensions
    n_iterations - number of iterations on the gradient descent optimization of the circle fit
    
    Returns
    -----
    x - tuple of (best_center_coordinates, radius, returns_dict), 
    where
    returns_dict contains (hull_vertices, hull_center, centers, losses) 
    for center and loss convergence plots
    '''

    hull = ConvexHull(xy_coords)
    hull_vertices = xy_coords[hull.vertices, :]
    hull_center = np.mean(hull_vertices, axis=0)

    center = np.copy(hull_center)
    centers = []
    losses = []
    radius = 0

    for i in range(n_iterations):

        # Compute gradient
        # Equation 5, Coope (1992) Circle fitting by linear and non-linear least squares
        xy_vectors = xy_coords - center
        radius = np.mean(np.linalg.norm(xy_vectors, axis=1))
        azimuths = xy_vectors / np.linalg.norm(xy_vectors, axis=1)[:, np.newaxis]
        gradient = 2 * (np.sum(xy_vectors, axis=0) - radius * np.sum(azimuths, axis=0))

        # Compute candidate directions
        directions = hull_vertices - center
        inner_product = np.dot(directions, gradient)
        best_direction = directions[np.argmin(inner_product)]

        # Gradient descent update
        learning_rate = 0.01 * 2 / (2 + i) # linear decay
        center += learning_rate * best_direction

        # Compute loss
        loss = np.linalg.norm(xy_vectors) - np.sum(np.linalg.norm(xy_vectors, axis=1))**2 / xy_vectors.shape[0]
        losses.append(loss)
        centers.append(np.copy(center))
    
    centers = np.array(centers)
    losses = np.array(losses)

    returns_dict = {}
    returns_dict['hull_vertices'] = hull_vertices
    returns_dict['hull_center'] = hull_center
    returns_dict['centers'] = centers
    returns_dict['losses'] = losses

    return (centers[-1, :], radius, returns_dict)