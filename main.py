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
import colorcet as cc

from utilities.isomapEmbedding import isomapEmbedding
from utilities.convex_fit import convex_fit
from utilities.d_squared import d_squared
from utilities.self_distance import self_distance
from utilities.frobenius_distance import frobenius_distance
from utilities.centered import centered
from utilities.circle_projection import circle_projection

def main(dataset="TinySOL"):

    if dataset.lower() == "tinysol":
        track_ids = tinysol.track_ids()
        track_instrs = [track_id.split('-')[0] for track_id in track_ids]
        instr_list = list(Counter(track_instrs).keys()) + [''] # 14 instruments + all instruments 
    else:
        instr_list = ['']

    Q = 24
    settings = {
        'Q': [24],
        'k': [3],
        'comp': ['log'],
        'instr': instr_list
    }
    settings_list = ParameterGrid(settings)
    
    losses = {}
    radii = {}
    for setting in settings_list:
        # read precomputed features
        with h5py.File("{}.h5".format(dataset), "r") as f:
            features_dict = {
                key:f[key][()]
                for key in f.keys()
                if setting['instr'] in key
            }
        batch_features = np.stack(list(features_dict.values()), axis=1)

        # compute isomap for subset
        isomap, _, rho_std = isomapEmbedding(batch_features)
        xyz_coords = isomap.fit_transform(rho_std)

        # convex hull fit, line fit
        xy_coords = xyz_coords[:, :2]
        xy_center, radius, returns_dict = convex_fit(xy_coords)
        xy_prime = circle_projection(xy_center, radius, xy_coords) # projected (x,y) coordinates

        z = xyz_coords[:, -1]
        z_fit = linregress(np.arange(len(z)), z)
        z_prime = np.asarray([i * z_fit.slope + z_fit.intercept for i, _ in enumerate(z)]) # projected z coordinates

        xyz_prime = np.concatenate((xy_prime, z_prime[:, np.newaxis]), axis=1)
        
        # Euclidean projection loss
        diffs_squared = (xyz_coords- xyz_prime) ** 2
        d_euclideans = np.sqrt(np.sum(diffs_squared, axis=1))
        loss = np.mean(d_euclideans)

        losses[setting['instr']] = loss
        radii[setting['instr']] = radius

        # Plot Frank-wolfe convergence plot
        hull_vertices = returns_dict['hull_vertices']
        hull_center = returns_dict['hull_center']
        centers = returns_dict['centers']

        plt.figure(figsize=(2.5, 2.5))
        plt.axis("off")
        plt.plot(
            np.concatenate([hull_vertices[:, 0], hull_vertices[0:, 0]]), 
            np.concatenate([hull_vertices[:, 1], hull_vertices[0:, 1]]),
            '-s', color='k', linewidth=0.5, markersize=2.0)
        plt.plot(hull_center[np.newaxis, 0], hull_center[np.newaxis, 1], 
                'd', color='r', markersize=4.0, fillstyle='none', linewidth=0.5)
        plt.plot(centers[:, 0], centers[:, 1], '-', color='g', linewidth=1.0)
        plt.plot(centers[-1, 0], centers[-1, 1], 's', color='g', markersize=4.0, fillstyle='none', linewidth=0.5)

        color_ids = np.floor(np.linspace(0, 256, Q, endpoint=False)).astype("int")
        color_list = [cc.cyclic_mygbm_30_95_c78[i] for i in color_ids]

        # Plot embedding with color
        n_points = xy_coords.shape[0]
        for i in range(n_points):
            plt.scatter(xy_coords[i, 0], xy_coords[i, 1], color = color_list[i%Q], s=25.0)
        plt.plot(xy_coords[:, 0], xy_coords[:, 1], color="black", linewidth=0.2)

        pdf_name = "./convexHull/{}.pdf".format(dataset + '_' + setting['instr'])
        plt.savefig(pdf_name) # store with dataset and instrument name

    # Sort by euclidean loss for readability
    sorted_losses = {k : v for k, v in sorted(losses.items(), key = lambda item : item[1])}
    sorted_radii = {k : v for k, v in sorted(radii.items(), key = lambda item : item[1])}

    with open("{}_euclideanLosses.json".format(dataset), "w") as f:
        json.dump(sorted_losses, f)

    with open("{}_radii.json".format(dataset), "w") as f:
        json.dump(sorted_radii, f)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="case-insensitive dataset name [TinySOL, NTVow, ENST]")
    args = parser.parse_args()

    if not args.dataset:
        main()
    elif args.dataset.lower() not in ["tinysol", "ntvow", "enst"]:
        raise ValueError("Invalid argument")
    elif args.dataset.lower() == "ntvow":
        main("NTVow")
    elif args.dataset.lower() == "enst":
        main("ENST-drums-public")
    else:
        main()
    



