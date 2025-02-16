# Cosmology-from-the-CMB-with-advanced-sampling-techniques
The aim of this repository is to accurately sample from the CMB using Tensorflow probability and advanced sampling techniques. 

Code is written for python and uses the package healpy - which is only supported by linux and macos. Windows users must therefore either use Google Colab or a virtualbox to use the healpy functions in this repository. The other packages which must be installed prior to use are CAMB, Tensorflow and Tensorflow Probability.

The code is split into two files:
- 'Main.ipynb':
This has the method definitions for the sampling, and has a cell that can generate a .py file with these methods, which can then be imported and used. This conversion ignores cells with "#EXCLUDE_FROM_PY".

- 'CMB_with_advanced_sampling_techniques.ipynb' an example run through the CMB sampling for a few different NSIDE and lmax values and measures the convergence and efficiency of the sampling.