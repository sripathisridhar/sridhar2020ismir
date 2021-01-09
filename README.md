# Helicality: An Isomap-based Measure of Octave Equivalence in Audio Data
This is the repository pertaining to the above-titled Late-Breaking Demo presented at ISMIR 2020.
In this paper, we introduce a novel algorithm to measure the octave-equivalence of audio datasets. 
Octave equivalence serves as domain-knowledge in MIR systems, including chromagram, spiral convolutional networks, and harmonic CQT. 
Prior work has applied the Isomap manifold learning algorithm to unlabeled audio data to embed frequency sub-bands in 3-D space where 
the Euclidean distances are inversely proportional to the strength of their Pearson correlations. 
However, discovering octave equivalence via Isomap requires visual inspection and is not scalable. 
To address this problem, we define "helicality" as the goodness of fit of the 3-D Isomap embedding to a Shepherd-Risset helix. 
Our method is unsupervised and uses a custom Frank-Wolfe algorithm to minimize a least-squares objective inside a convex hull. 
Numerical experiments indicate that isolated musical notes have a higher helicality than speech, followed by drum hits. 

## Dependencies
mir-data <br/>
sklearn, scipy, numpy (core numerical computation) <br/>
librosa (audio feature extraction) <br/>
matplotlib, colorcet (plotting) <br/>
h5py, json (data handling) <br/>

## Download and run
Dataset features are pre-computed and stored in the corresponding .h5 files in the root directory. <br/>
Execute main.py from a command line terminal with the name of the dataset you want to test. <br/>

`python3 main.py -d tinysol`

Plots are stored in the `./convexHull` sub-directory by default. <br/>
Numerical results are stored in the `<dataset>_helicality.json` format in the main directory. <br/>

## Datasets
TinySOL (Isolated notes played on 14 different instruments) <br/>
ENST-drums (dry_mix subset which contains isolated hits on drums) <br/>
NTVOW (North Texas Vowel Dataset, containing 12 vowel utterances from 50 speakers) <br/>

## Links
[Pre-print](https://arxiv.org/abs/2010.00673) <br/>
[ISMIR Presentation Video](https://youtu.be/ayflseXZ3-c) <br/>


