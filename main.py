import numpy as np
import os
import h5py
from sklearn.model_selection import ParameterGrid
from scipy.stats import linregress
from mirdata import tinysol
from collections import Counter

from utilities.isomapEmbedding import isomapEmbedding
from utilities.convex_fit import convex_fit
from utilities.d_squared import d_squared
from utilities.self_distance import self_distance
from utilities.frobenius_distance import frobenius_distance
from utilities.centered import centered

DATA_DIR = "/Users/sripathisridhar/Documents/GitHub/embedding-bio"

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
    
    for setting in settings_list:
        # read precomputed features
        with h5py.File("TinySOL.h5", "r") as f:
            features_dict = {
                key:f[key][()]
                for key in f.keys()
                if setting["instr"] in key
            }
        batch_features = np.stack(np.array(features_dict.values()), axis=1)

        #2 compute isomap for subset
        isomap, freqs, rho_std = isomapEmbedding(batch_features)
        xyz_coords = isomap.fit_transform(rho_std)

        # convex fit, line fit
        xy_fit = convex_fit(xyz_coords[:, :2])

        z = xyz_coords[:, -1]
        z_fit = linregress(np.arange(len(z)), z)
        # store loss

if __name__ == "__main__":
    main()



