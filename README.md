# Description
This repository contains the detector described in ["Vehicular Multi-object Tracking with Persistent Detector Failures"](https://arxiv.org/abs/1907.11306). The tracker described there is in the [kittitracking-temht](https://github.com/motrom/kittitracking-temht) repository.

Vehicle detection using 64-laser lidar on the Kitti dataset. The model is a boosted tree as in the classic Viola-Jones object detector for images. The features are the presence of lidar returns in 3D box regions (aka voxels).  
The idea is simple, but some details prevent adaptation of existing code from the vision community. Objects are assumed to have relatively fixed size but will be rotated wrt the sensor. The 'image' is obviously 3-D, and the number of possible box features is immense enough to demand a non-exhaustive feature search during training.  
This method was not expected to outperform SOTA deep network detectors (and it doesn't), but it is good to have a performance marker for alternative methods on the Kitti 3D benchmark, which was only established after deep methods became the norm. It also runs fairly fast on a single CPU, which is nice if you spent all your money on lidar.

# Dependencies
Python w/ fairly recent numpy, numba, imageio, and scikit-learn  
opencv and matplotlib for displaying results  
C compiler and Intel AVX for fast inference

# Performance
Bird's-eye view car detection AP on the test set is **.65**,**.55**,**.50** for easy, moderate, and hard partitions respectively. This puts it a step below the first successful deep network, 3DFCN, but superior to some experimental deep network architectures including the only CPU-amenable network, LMnet.

The current runtime on a single CPU is 180ms. This was sped up in the paper using a track-before-detect principle to only run detection on certain areas at each timestep. A GPU implementation is underway...