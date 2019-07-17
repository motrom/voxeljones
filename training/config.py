#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 6/6/19
"""
import numpy as np

#anchorstep = np.array((.1, .1, .1))
#anchorstart = np.array((-30, -20, 2))
#anchorlen = np.array((60, 40, 24))
anchorstep = np.array((.125, .125, .125))
anchorstart = np.array((-24, -16, 1))
anchorlen = np.array((48, 32, 20))
anchornangles = 16 # in top half, actual number of angles is this*2

stride = np.array((4, 4, 2))

predictionview = np.array((4., 50., -40., 40.))

grndstep = np.array((3., 3.))
max_road_slope = .1 # tangent
# slope ~= .15 , current highest kitti slope seems to be .998

lidar_files = '../object/training/velodyne/{:06d}.bin'
gt_files = '../object/training_labels_orig/{:06d}.txt'


positive_points_file = '../dataApril19/positivepoints617.npy'
positive_points_index_file = '../dataApril19/positiveidxs617.npy'
present_boxes_file = '../dataApril19/avoidboxes617.npy'

ground_planes_by_file = '../dataApril19/groundplane/{:04d}.npy'



def floor(arr):
    # casting rounds up for negative numbers, which is a problem for grid lookup
    return np.floor(arr).astype(int) if type(arr) == np.ndarray else int(arr//1)

anchorangles = np.linspace(0, np.pi, anchornangles, endpoint=False)

# the local grid must extend this far beyond the center ground tile
localgridlen = floor(grndstep/anchorstep[:2]) + anchorlen[:2]

# make array of the anchor boxes that will be considered for each local grid
# note we assume the anchors have an even number of cells in each dimension
# ->grids are all left-skewed
testlen = floor(grndstep/2./anchorstep[:2]) + 1
anchorinlocalgrid = np.mgrid[0:localgridlen[0]-anchorlen[0],
                             0:localgridlen[1]-anchorlen[1]]
anchorinlocalgrid = anchorinlocalgrid.reshape((2,-1)).T.copy()
#anchorinlocalgrid = np.append(anchorinlocalgrid, anchorinlocalgrid+anchorlen[:2], axis=1)

anchornangles_strided = anchornangles // stride[2]
anchorangles_strided = anchorangles[(stride[2]-1)//2::stride[2]].copy()
anchorinlocalgrid_strided = np.mgrid[(stride[0]-1)//2:localgridlen[0]-anchorlen[0]:stride[0],
                                     (stride[1]-1)//2:localgridlen[1]-anchorlen[1]:stride[1]]
anchorinlocalgrid_strided = anchorinlocalgrid_strided.reshape((2,-1)).T.copy()

# determine which adjacent ground tiles will be used to make the local grid
bufferlen = np.hypot(anchorlen[0]*anchorstep[0], anchorlen[1]*anchorstep[1])/2.
nlocaltiles = floor(bufferlen/grndstep) + 1
# determine range of ground tiles
grndstart = floor(predictionview[[0,2]]/grndstep) - nlocaltiles
grndlen = floor(predictionview[[1,3]]/grndstep) + nlocaltiles+1 - grndstart
if grndlen[0]%2 == 1: grndlen[0] += 1
if grndlen[1]%2 == 1: grndlen[1] += 1

# grid of cells that will be checked for objects
# go ahead and remove cells that are outside of kitti annotated range
grnd2checkgrid = np.mgrid[nlocaltiles[0]:grndlen[0]-nlocaltiles[0],
                          nlocaltiles[1]:grndlen[1]-nlocaltiles[1]].reshape((2,-1)).T.copy()
_grnd2checkclosery = np.maximum(grnd2checkgrid[:,1]+grndstart[1],
                               -grnd2checkgrid[:,1]-grndstart[1]-1)*grndstep[1]
_grnd2checkfartherx = (grnd2checkgrid[:,0]+grndstart[0]+1)*grndstep[0]
_grnd2checkgridinclude = _grnd2checkfartherx*.87 > _grnd2checkclosery + 1
_grnd2checkgridinclude &= _grnd2checkfartherx > 3
grnd2checkgrid = grnd2checkgrid[_grnd2checkgridinclude]

# grid of local tiles to include
grnd4localgrid = np.mgrid[-nlocaltiles[0]:nlocaltiles[0]+1,
                          -nlocaltiles[1]:nlocaltiles[1]+1].reshape((2,-1)).T.copy()


_trainvalsplit = np.load('trainvalsplit.npz')
training_split = _trainvalsplit['train']
validation_split = _trainvalsplit['val']