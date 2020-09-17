import numpy as np
import os
import h5py
import json
from sklearn.model_selection import ParameterGrid
from scipy.stats import linregress
from mirdata import tinysol
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt

from utilities.isomapEmbedding import isomapEmbedding
from utilities.convex_fit import convex_fit
from utilities.d_squared import d_squared
from utilities.self_distance import self_distance
from utilities.frobenius_distance import frobenius_distance
from utilities.centered import centered
from utilities.circle_projection import circle_projection

def main():

    track_ids = tinysol.track_ids()
    track_instrs = [track_id.split('-')[0] for track_id in track_ids]
    instr_list = list(Counter(track_instrs).keys())

    Q = 24
    settings = {
        'Q': [24],
        'k': [3],
        'comp': ['log'],
        'instr': instr_list
    }
    settings_list = ParameterGrid(settings)
    
    losses = {}
    for setting in settings_list:
        # read precomputed features
        with h5py.File("TinySOL.h5", "r") as f:
            features_dict = {
                key:f[key][()]
                for key in f.keys()
                if setting["instr"] in key
            }
        batch_features = np.stack(list(features_dict.values()), axis=1)

        # compute isomap for subset
        isomap, freqs, rho_std = isomapEmbedding(batch_features)
        xyz_coords = isomap.fit_transform(rho_std)

        # convex hull fit, line fit
        xy_coords = xyz_coords[:, :2]
        xy_center, radius, returns_dict = convex_fit(xy_coords)
        xy_prime = circle_projection(xy_center, radius, xy_coords) # projected (x,y) coordinates

        z = xyz_coords[:, -1]
        z_fit = linregress(np.arange(len(z)), z)
        z_prime = np.asarray([i * z_fit.slope + z_fit.intercept for i, _ in enumerate(z)]) # projected z coordinates

        xyz_prime = np.concatenate((xy_prime, z_prime[:, np.newaxis]), axis=1)
        
        # store loss
        D_proj = self_distance(xyz_prime)
        D_isomap = self_distance(xyz_coords)

        loss = frobenius_distance(centered(D_proj), centered(D_isomap)) / xyz_prime.shape[0]
        losses[setting["instr"]] = loss

        # Plot Frank-wolfe convergence plot
        hull_vertices = returns_dict['hull_vertices']
        hull_center = returns_dict['hull_center']
        centers = returns_dict['centers']
        plt.figure(figsize=(8,8))
        plt.plot(
            np.concatenate([hull_vertices[:, 0], hull_vertices[0:, 0]]), 
            np.concatenate([hull_vertices[:, 1], hull_vertices[0:, 1]]),
            '-s', color='b')
        plt.plot(hull_center[np.newaxis, 0], hull_center[np.newaxis, 1], 'd', color='r')
        plt.plot(centers[:, 0], centers[:, 1], '-', color='g')
        plt.plot(centers[-1, 0], centers[-1, 1], 's', color='g')
        plt.plot(xy_coords[:, 0], xy_coords[:, 1], '.', color='k', alpha=1.0)
        plt.savefig("./convexHull/{}.pdf".format(setting["instr"]))

    with open("TinySOL.json", "w") as f:
        json.dump(losses, f)

if __name__ == "__main__":
    main()



