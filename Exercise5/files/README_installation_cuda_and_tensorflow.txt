Installation of Cuda and cudnn:
-------------------------------

1.) Install Cuda (after downloading from https://developer.nvidia.com/cuda-downloads) 
    NOTE: Code requires Cuda v8 to be downloaded, not v9!

2.) Install cudnn as explained in 
    http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows
    NOTE: Code requires cudnn v5 and v6, not v7
          first, v5 has been installed, then v6 (overwriting include and lib files)

Cuda directory:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\



Installation of Tensorflow:
---------------------------

after installing cuda:

pip install --upgrade tensorflow-gpu


Installation of Keras:
----------------------

pip install keras


The installation of Tensorflow and Keras is described in 
http://www.jakob-aungiers.com/articles/a/Installing-TensorFlow-GPU-Natively-on-Windows-10
