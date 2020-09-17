import numpy as np
import os 
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import ConvexHull

def convex_fit(xy_coords, n_iterations=500, save=True):
    '''
    This function returns a circle fit within the bounds of a convex hull, optimized using the Frank-Wolfe algorithm

    Inputs
    -----
    xy_coords - (m,2) array of m point coordinates in two dimensions
    n_iterations - number of iterations on the gradient descent optimization of the circle fit
    save - user interface option. Default option is to save the circle fit plot convergence and return [(circle_center), best_radius] only.
           'False' returns [convergence_losses, circle_centers, best_radius] where convergence_losses and circle_centers are vectors
    
    Returns
    -----
    x - list of values depending on save parameter. Refer above
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

        #Gradient descent update
        learning_rate = 0.01 * 2 / (2 + i) # linear decay
        center += learning_rate * best_direction

        # Compute loss
        loss = np.linalg.norm(xy_vectors) - np.sum(np.linalg.norm(xy_vectors, axis=1))**2 / xy_vectors.shape[0]
        losses.append(loss)
        centers.append(np.copy(center))
    
    centers = np.array(centers)
    losses = np.array(losses)

    if save==True:
        plt.figure(figsize=(2.5,2.5))
        plt.plot(np.concatenate([hull_vertices[:, 0], hull_vertices[0:, 0]]), 
                 np.concatenate([hull_vertices[:, 1]], hull_vertices[0:, 1]))
        plt.plot(hull_center[np.newaxis, 0], hull_center[np.newaxis, 1], 'd', color='r')
        plt.plot(centers[:, 0], centers[:, 1], '-', color='g')
        plt.plot(centers[-1, 0], centers[-1, 1], 's', color='g')
        plt.plot(xy_coords[:, 0], xy_coords[:, 1], '.', color='k', alpha=1.0)
        plt.savefig(f'{int(losses[-1])}.pdf')
        return [centers[-1,:], radius]
    elif save==False:
        return [losses, centers, radius]
    else:
        raise AttributeError('Invalid save parameter')