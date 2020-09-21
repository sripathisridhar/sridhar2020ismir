import numpy as np
import os
import glob
from sklearn.model_selection import ParameterGrid
import librosa
from librosa import load
from librosa import cqt
from librosa.display import specshow
from librosa.feature import rms
from tqdm import tqdm
import h5py

DATA_DIR = '/Users/sripathisridhar/Desktop/ENST-drums-public'
MAIN_DIR = '/Users/sripathisridhar/Documents/GitHub/embedding-bio'

def preCompute_music():
    '''
    Compute the CQT features of a music dataset

    Returns
    -----
    features_dict: {wav_name : CQT_feature_vector}
    '''
    wav_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*/audio/dry_mix", "*hits*.wav")))

    HOP_SIZE = 512
    Q = 24
    CQT_OCTAVES = 7

    features_dict = {}
    for wav_path in tqdm(wav_paths):

        # Read audio files
        wav, sr = load(path=wav_path, sr=None)

        # Compute CQTs
        cqt_complex = cqt(
                        y=wav, 
                        sr=sr,
                        hop_length=HOP_SIZE, 
                        n_bins=Q*CQT_OCTAVES, 
                        bins_per_octave=Q, 
                        sparsity=1e-6)
        scalogram = np.abs(cqt_complex)**2
    
        # Find frame of maximum RMS value
        wav_rms = rms(y=wav, hop_length=HOP_SIZE)
        rms_argmax = np.argmax(wav_rms)
        frame = scalogram[:, rms_argmax]

        # Store in features_dict
        wav_key = os.path.basename(wav_path)
        features_dict[wav_key] = frame
    
    dataset = os.path.basename(DATA_DIR)
    h5py_path = os.path.join(MAIN_DIR, "{}.h5".format(dataset))
    
    with h5py.File(h5py_path, "w") as f:
        for key in features_dict.keys():
            f[key] = features_dict[key]
 
if __name__ == "__main__":
    preCompute_music()
