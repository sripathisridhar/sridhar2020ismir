# sridhar2020ismir
In this paper, we introduce a novel algorithm to measure the octave-equivalence of audio datasets. To that end
Octave equivalence serves as domain-knowledge in MIR systems, including chromagram, spiral convolutional networks, and harmonic CQT. 
Prior work has applied the Isomap manifold learning algorithm to unlabeled audio data to embed frequency sub-bands in 3-D space where 
the Euclidean distances are inversely proportional to the strength of their Pearson correlations. 
However, discovering octave equivalence via Isomap requires visual inspection and is not scalable. 
To address this problem, we define "helicality" as the goodness of fit of the 3-D Isomap embedding to a Shepherd-Risset helix. 
Our method is unsupervised and uses a custom Frank-Wolfe algorithm to minimize a least-squares objective inside a convex hull. 
Numerical experiments indicate that isolated musical notes have a higher helicality than speech, followed by drum hits. 

## Dependencies
mir-data
sklearn, scipy, numpy (core numerical computation)
librosa (audio feature extraction)
matplotlib, colorcet (plotting)
h5py, json (data handling)

## Download and run
Dataset features are pre-computed and stored in the corresponding .h5 files in the root directory.
Execute main.py from command line with the name of the dataset you want to test

## Datasets
TinySOL (Isolated notes played on 14 different instruments)
ENST-drums (dry_mix subset which contains isolated hits on drums)
NTVOW (North Texas Vowel Dataset, containing 12 vowel utterances from 50 speakers)

## Links
Link to the pre-print: https://arxiv.org/abs/2010.00673
Link to presentation video: https://youtu.be/ayflseXZ3-c


