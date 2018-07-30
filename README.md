# model_utils

This utility is an add on to Keras to perform model 'surgery'. The utility provides functions to modify the model json and saved model h5 definition to restructure and rearrange the model architecture kernels, WITHOUT retraining. 

The current version is tested with Tensorflow. 

It is recommended to copy the model h5 file before performing model surgery using the following command:

h5repack -i <model_h5_file> -o <new_model_h5_file>

and run the provided functions on the new h5 file generated

# Reduction_utils

reduction_utils.py can be used to prune or modify the model architecture. The flexibility of changing kernel matrices, not just removing them enables advanced pruning methods like cosine similarity based pruning (included in reduction_utils.py), correlation coffecient based pruning and just average percentage of zeros pruning methods. 

# Documentation

The docstrings is the documentation that is provided with the software. Questions/suggestions welcome.
