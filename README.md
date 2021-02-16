# Cosmology-from-the-CMB-with-advanced-sampling-techniques
The aim of this repository is to accurately sample from the CMB using Tensorflow probability and advanced sampling techniques. 

Code is written for python and uses the package healpy - which is only supported by linux and macos. Windows users must therefore either use Google Colab or a virtualbox to use the healpy funcitons in this repository. The other packages which must be installed prior to use are CAMB, Tensorflow and Tensorflow Probability.

The code is split into two files:
- 'Main.py' which includes the functions which are involved.
- 'CMB_with_advanced_sampling_techniques.ipynb' which gives a run through the CMB sampling for a few different NSIDE and lmax values - and measures the convergence and efficiency of the sampling.
