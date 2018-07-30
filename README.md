# model_utils

This utility is an add on to Keras to perform model 'surgery'. The utility provides functions to modify the model json and saved model h5 definition to restructure and rearrange the model architecture kernels, WITHOUT retraining. 

The current version is tested with Tensorflow. 

It is recommended to copy the model h5 file before performing model surgery using the following commands:

h5repack -i <model_h5_file> -o <new_model_h5_file>
