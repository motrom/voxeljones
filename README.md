# Description
arxiv paper released soon...  
Vehicle detection using 64-laser lidar on the Kitti dataset. The model is a boosted tree as in the classic Viola-Jones object detector for images. The features are the presence of lidar returns in 3D box regions (aka voxels).  
The idea is simple, but some details prevent adaptation of existing code from the vision community. Objects are assumed to have relatively fixed size but will be rotated wrt the sensor. The 'image' is obviously 3-D, and the number of possible box features is immense enough to demand a non-exhaustive feature search during training.  
This method was not expected to outperform SOTA deep network detectors (and it doesn't), but it is good to have a performance marker for alternative methods on the Kitti 3D benchmark, which was only established after deep methods became the norm. It also runs fairly fast on a single CPU, which is nice if you spent all your money on lidar.

# Dependencies
Python w/ fairly recent numpy, numba, imageio, and scikit-learn  
opencv and matplotlib for displaying results  
C compiler and Intel AVX for fast inference

# Performance
It is not possible to submit to the Kitti benchmark website until September, so currently the method is tested on a split of the training set. Specifically, the validation split contains 1761 frames corresponding to the first 10 scenes from the tracking benchmark's training set. Bird's-eye view car detection AP using the standard Kitti evaluation code is **.66**,**.55**,**.55** for easy, moderate, and hard partitions respectively. This puts it roughly even with the first deep network, 3DFCN, for the hard partition, and superior to some experimental deep network architectures including the only CPU-amenable network, LVMnet. Of course, these results assume that the 10-scene validation set is a good approximation of the test set.

The current runtime on a single CPU is 180ms. A GPU implementation is underway...