# -*- coding: utf-8 -*-
"""
last mod 7/16/19

modify this to run on specific data (and to fit your file format)
files are formatted with the index numbers in files2use
"""

from config import training_split, validation_split

files2use = validation_split[:200]

lidar_files = '../../object/training/velodyne/{:06d}.bin'

gt_files = '../../object/training_labels_orig/{:06d}.txt'

ground_planes_by_file = '../../dataApril19/groundplane/{:04d}.npy'

img_files = '../../object/training/image_2/{:06d}.png'

output_files = None#'../../object/estimates/d/data/{:06d}.txt'

model_file = '../trainedmodels/BT630.npz'


#lidar_files = '../../object/testing/velodyne/{:06d}.bin'
#img_files = '../../object/testing/image_2/{:06d}.png'
#gt_files = None
#output_files = '../../object/estimates/test/data/{:06d}.txt'
#model_file = '../trainedmodels/BT630.npz'
#files2use = range(10)