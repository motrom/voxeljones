#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 5/1/19
get all positives from training dataset and save them
saves:
    [N 3] float array of points
    [n 3] int array of start-end indices for each sample
          final column is 1 if it is safe to randomly shift this guy (not near edge of view)
    [n+ 6] float array of boxes (xyalw format) to avoid when mining negatives
          with file idx first
"""

import numpy as np
import cv2

from config import lidar_files, gt_files, training_split
from config import floor
from config import positive_points_file, positive_points_index_file
from config import present_boxes_file
from config import grndstart, grndstep, grndlen, ground_planes_by_file
from config import anchorstart, anchorstep, anchorlen#, anchorangles
from config import predictionview, nlocaltiles
from ground2 import planes2Transforms, tilePoints
from calibs import calib_extrinsics, calib_map
from analyzeGT import readGroundTruthFile


estimate_ncars_per_file = 4
estimate_npoints_per_car = 800 # 400 for classifier, 600 for regressor

#random_position_error = .06 # m in either direction, uniform
#random_angle_error = np.diff(anchorangles)*.6 # rad in either direction, uniform
# going to randomly shift points during training, keep nearby points
include_buffer = 1.5 # .5 for classifier, 1.5 for regressor
#nlocaltiles += 1


#def plotCar(pts):
#    plt.figure()
#    plt.xlim((-3.,3))
#    plt.ylim((-2.,2))
#    plt.scatter(pts[:,0], pts[:,1], c=((pts[:,2]-.7)/3.1*256).astype(int))
canvas = np.zeros((7*80, 5*80, 3), dtype=np.uint8)


nfiles = len(training_split)
maxncars = estimate_ncars_per_file * nfiles
maxnotscoredcars = maxncars*2
maxnpoints = estimate_npoints_per_car * maxncars
out_pts = np.zeros((maxnpoints,3))
out_idxs = np.zeros((maxncars, 3), dtype=int)
out_positiveboxes = np.zeros((maxnotscoredcars, 7))
ncars = 0
npoints = 0
nnotscoredcars = 0

for file_idx in training_split:
    if file_idx % 1000 == 0: print("file {:d}".format(file_idx))
    calib_extrinsic = calib_extrinsics[calib_map[file_idx]].copy()
    calib_extrinsic[2,3] += 1.65
    
    # load relevant data
    data = np.fromfile(lidar_files.format(file_idx),
                       dtype=np.float32).reshape((-1,4))[:,:3]
    with open(gt_files.format(file_idx), 'r') as fd: gtstr = fd.read()
    gtstuff = readGroundTruthFile(gtstr, ('Car', 'Van'))
    gt = [{'box':gtstuff[1][gtidx],
           'scored':gtstuff[4][gtidx]} for gtidx in xrange(len(gtstuff[0]))]
    ground = np.load(ground_planes_by_file.format(file_idx))
    
    # get ground
    full_data_xyz = data.dot(calib_extrinsic[:3,:3].T) + calib_extrinsic[:3,3]
    full_data_tiled, data_tile_idxs = tilePoints(full_data_xyz,
                                                 grndstart, grndstep, grndlen)
    groundTs = planes2Transforms(ground)
    
    for gtobj in gt:
        out_positiveboxes[nnotscoredcars,0] = file_idx
        out_positiveboxes[nnotscoredcars,1:6] = gtobj['box']
        nnotscoredcars += 1
        if not gtobj['scored']: continue
        gtx, gty, gtang, gtl, gtw = gtobj['box']
        if not (gtx > predictionview[0] and gtx < predictionview[1] and
                gty > predictionview[2] and gty < predictionview[3]):
            continue
        
        out_positiveboxes[nnotscoredcars, 6] = 1.
        
        gtgroundidx = floor((gtx,gty) / grndstep) - grndstart
        groundT = groundTs[gtgroundidx[0], gtgroundidx[1]]
        safetoshift = all(gtgroundidx > nlocaltiles)
        safetoshift &= all(gtgroundidx < grndlen-nlocaltiles-1)
        
        # specify anchor grid of box
        gtz = -(groundT[2,3]+groundT[2,0]*gtx+groundT[2,1]*gty)/groundT[2,2]
        gtx2, gty2, gtz2 = groundT[:3,:3].dot([gtx, gty, gtz]) + groundT[:3,3]
        assert np.isclose(gtz2, 0.)
        gtc = np.cos(gtang)
        gts = np.sin(gtang)
        carT = np.array(((gtc,-gts, 0, gtx2),
                         (gts, gtc, 0, gty2),
                         (  0,   0, 1, 0),
                         (  0,   0, 0, 1)))
        gtT = np.linalg.inv(carT).dot(groundT)
        ulo, vlo, zlo = anchorstart*anchorstep
        uhi, vhi, zhi = (anchorstart + anchorlen) * anchorstep
        zlo -= .1
        zhi += .1
        nlocal = nlocaltiles
        if safetoshift:
            ulo -= include_buffer
            uhi += include_buffer
            vlo -= include_buffer
            vhi += include_buffer
            nlocal = nlocaltiles + 1
        
        localtilegrid = np.mgrid[-nlocal[0]:nlocal[0]+1,-nlocal[1]:nlocal[1]+1]
        localtilegrid = localtilegrid.reshape((2,-1)).T
        
        # gather points from nearby ground tiles
        # and transform to box's ground reference
        pts = []
        for tilex, tiley in localtilegrid:
            tile_idx = (tilex+gtgroundidx[0])*grndlen[1] + tiley + gtgroundidx[1]
            data_start, data_end = data_tile_idxs[tile_idx:tile_idx+2]
            tilepts = full_data_tiled[data_start:data_end].dot(gtT[:3,:3].T) + gtT[:3,3]
            include_pts = tilepts[:,0] > ulo
            include_pts &= tilepts[:,0] < uhi
            include_pts &= tilepts[:,1] > vlo
            include_pts &= tilepts[:,1] < vhi
            include_pts &= tilepts[:,2] > zlo
            include_pts &= tilepts[:,2] < zhi
            pts.append(tilepts[include_pts])
        pts = np.concatenate(pts, axis=0)
        
#        canvas[:] = 255
#        ptsi = (pts - (ulo,vlo,zlo))/(uhi-ulo,vhi-vlo,zhi-zlo)
#        ptsred = (ptsi[:,2] * 256).astype(np.uint8)
#        ptsi = (ptsi[:,:2] * (canvas.shape[0]-4, canvas.shape[1]-4)).astype(int) + 2
#        for spacex in xrange(-2,3):
#            for spacey in xrange(-2,3):
#                canvas[ptsi[:,0]+spacex, ptsi[:,1]+spacey,0] = ptsred
#                canvas[ptsi[:,0]+spacex, ptsi[:,1]+spacey,2] = 255-ptsred
#                canvas[ptsi[:,0]+spacex, ptsi[:,1]+spacey,1] = 190
#        cv2.imshow('a', canvas)
#        exitkey = cv2.waitKey(500)
#        if exitkey == 113: raise Exception("wowzers")
          
        npts = pts.shape[0]
        out_pts[npoints:npoints+npts] = pts
        out_idxs[ncars] = (npoints, npoints+npts, safetoshift)
        npoints += npts
        ncars += 1


np.save(positive_points_file, out_pts[:npoints])
np.save(positive_points_index_file, out_idxs[:ncars])
np.save(present_boxes_file, out_positiveboxes[:nnotscoredcars])

print "{:d} files, {:d} cars, {:d} boxes, {:d} points".format(nfiles, ncars,
       nnotscoredcars, npoints)