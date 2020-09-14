#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.manifold import Isomap

import librosa
from librosa.core import amplitude_to_db


# In[2]:


def isomapEmbedding(batch_features, q=24, comp='log', n_neighbors=3, n_octaves=3, n_dimensions=3):
    '''
    Returns an isomap embedding in n_dimensions, with n_neighbors neighborhood graph and
    restricted to n_octaves
    
    Inputs
    ------
    batch_features- np.ndarray of CQT features stacked horizontally
    q- frequency bins per octave used in CQT features
    comp- compression, 'log' by default
    n_neighbors- number of neighbors in k-nearest neighbor graph, 3 by default
    n_octaves- number of octaves in output embedding with range of relative importance is chosen, 3 by default
    n_dimensions- number of dimensions, 3 by default
    
    Returns
    ------
    isomap - Isomap object with learned embedding
    freqs - Frequency bin array
    rho_std - Pearson correlation matrix
    '''
    
    
    CQT_OCTAVES = 7
    
    if comp!='log':
        raise Exception("Only log compression currently supported")
    else:
        features = amplitude_to_db(batch_features)

    # Prune feature matrix
    bin_low = np.where((np.std(features, axis=1) / np.std(features)) > 0.1)[0][0] + q
    bin_high = bin_low + n_octaves*q 
    X = features[bin_low:bin_high, :]

    # Z-score Standardization- improves contrast in correlation matrix
    mus = np.mean(X, axis=1)
    sigmas = np.std(X, axis=1)
    X_std = (X - mus[:, np.newaxis]) / (1e-6 + sigmas[:, np.newaxis]) # 1e-6 to avoid runtime division by zero

    # Pearson correlation matrix
    rho_std = np.dot(X_std, X_std.T) / X_std.shape[1]
    
    # Isomap embedding
    isomap = Isomap(n_components=n_dimensions, n_neighbors=n_neighbors)
    isomap.fit_transform(rho_std)
    
    # Get note value
    freqs = librosa.cqt_frequencies(q*CQT_OCTAVES, fmin=librosa.note_to_hz('C1'), bins_per_octave=q) #librosa CQT default fmin is C1
    chroma_list = librosa.core.hz_to_note(freqs[bin_low:bin_high])

    notes = []
    reps = q//12
    for chroma in chroma_list:
        for i in range(reps):
            notes.append(chroma)
            
    # Return embedding object
    return isomap, freqs, rho_std

