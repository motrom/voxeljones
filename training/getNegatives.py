# -*- coding: utf-8 -*-
"""
last mod 5/22/19
"""

import numpy as np
import numba as nb
from math import floor as mathfloor

from config import lidar_files
from config import present_boxes_file
from config import grndstart, grndstep, grndlen, ground_planes_by_file
from config import anchorstart, anchorstep, anchorlen
from config import nlocaltiles, localgridlen
from config import grnd2checkgrid, grnd4localgrid
from config import anchorinlocalgrid_strided as anchorinlocalgrid
from config import anchorangles_strided as anchorangles
from config import anchornangles_strided as anchornangles
from ground2 import planes2Transforms, tilePoints
from calibs import calib_extrinsics, calib_map

anchorcossins = np.column_stack((np.cos(anchorangles), np.sin(anchorangles)))

anchorcenterpoints = (anchorinlocalgrid - localgridlen//2 -
                          anchorstart[:2])*anchorstep[:2]
anchorcenter2 = np.einsum(anchorcenterpoints, [0,1], anchorcossins[:,0], [2], [2,0,1])
anchorcenter2[:,:,0] -= np.outer(anchorcossins[:,1], anchorcenterpoints[:,1])
anchorcenter2[:,:,1] += np.outer(anchorcossins[:,1], anchorcenterpoints[:,0])
anchorcenterpoints = anchorcenter2

#@nb.njit(nb.void(nb.f8[:,:], nb.f8[:], nb.f8, nb.b1[:,:,:]))
#def fillPositiveSample(pts, positionnoise, anglenoise, grid):
#    for pt in pts

@nb.njit(nb.void(nb.f8[:,:], nb.i8[:], nb.i8, nb.i8, nb.f8[:,:], nb.b1[:,:,:,:]))
def fillLocalGrid(pts, tileidxs, tilex, tiley, groundT, grid):
    grid[:] = False
    for grnd4localidx in xrange(grnd4localgrid.shape[0]):
        tilex2, tiley2 = grnd4localgrid[grnd4localidx]
        tile = (tilex+tilex2)*grndlen[1] + tiley+tiley2
        pts_idxstart, pts_idxend = tileidxs[tile:tile+2]
        for ptsidx in xrange(pts_idxstart, pts_idxend):
            pt = pts[ptsidx]
            grndpt = np.dot(groundT[:3,:3], pt) + groundT[:3,3]
            z = int(mathfloor(grndpt[2]/anchorstep[2])) - anchorstart[2]
            if z < 0 or z >= anchorlen[2]: continue
            for angle in xrange(anchornangles):
                xf = anchorcossins[angle,0]*grndpt[0] + anchorcossins[angle,1]*grndpt[1]
                x = int(mathfloor(xf/anchorstep[0])) + localgridlen[0]//2
                yf = anchorcossins[angle,0]*grndpt[1] - anchorcossins[angle,1]*grndpt[0]
                y = int(mathfloor(yf/anchorstep[1])) + localgridlen[1]//2
                if x >= 0 and x < localgridlen[0] and y >= 0 and y < localgridlen[1]:
                    grid[angle,x,y,z] = True


@nb.njit(nb.b1(nb.f8,nb.f8,nb.f8,nb.f8,nb.f8,nb.f8,
               nb.f8,nb.f8,nb.f8,nb.f8,nb.f8,nb.f8, nb.f8))
def rectOverlap(x1,y1,c1,s1,l1,w1, x2,y2,c2,s2,l2,w2, overlap_buffer):
    x2in1 = (x2-x1)*c1 + (y2-y1)*s1
    y2in1 = (y2-y1)*c1 - (x2-x1)*s1
    x1in2 = (x1-x2)*c2 + (y1-y2)*s2
    y1in2 = (y1-y2)*c2 - (x1-x2)*s2
    cos = abs(c1*c2+s1*s2)
    sin = abs(c1*s2-c2*s1)
    return not (l1 + l2*cos + w2*sin - abs(x2in1) < overlap_buffer or
                w1 + l2*sin + w2*cos - abs(y2in1) < overlap_buffer or
                l2 + l1*cos + w1*sin - abs(x1in2) < overlap_buffer or
                w2 + l1*sin + w1*cos - abs(y1in2) < overlap_buffer)
#    return not (x2in1 + l2*cos + w2*sin + l1 < overlap_buffer or
#                l1 - x2in1 + l2*cos + w2*sin < overlap_buffer or
#                y2in1 + l2*sin + w2*cos + w1 < overlap_buffer or
#                w1 - y2in1 + l2*sin + w2*cos < overlap_buffer or
#                x1in2 + l1*cos + w1*sin + l2 < overlap_buffer or
#                l2 - x1in2 + l1*cos + w1*sin < overlap_buffer or
#                y1in2 + l1*sin + w1*cos + w2 < overlap_buffer or
#                w2 - y1in2 + l1*sin + w1*cos < overlap_buffer)

@nb.njit(nb.b1[:,:,:,:]())
def prepLocalNms():
    nanchors = anchorinlocalgrid.shape[0]
    overlaps = np.zeros((anchornangles, anchornangles, nanchors, nanchors),
                        dtype=np.bool8)
    # length in each direction
    # set a little low to only catch close objs
    obj_len = 2.
    obj_wid = 1.
    obj_hypot = np.hypot(obj_len, obj_wid)
    overlap_buffer = .4
    for angleidx1, angleidx2, anchoridx1, anchoridx2 in np.ndindex(
                    anchornangles, anchornangles, nanchors, nanchors):
        if angleidx2 < angleidx1 or anchoridx2 < anchoridx1:
            continue
        
        x1, y1 = anchorcenterpoints[angleidx1, anchoridx1]
        x2, y2 = anchorcenterpoints[angleidx2, anchoridx2]
        overlap = False
        centerdist = np.hypot(x1-x2, y1-y2)
        if centerdist < obj_wid*2 - overlap_buffer:
            overlap = True
        elif centerdist > obj_hypot*2 - overlap_buffer:
            overlap = False
        else:
            cos1, sin1 = anchorcossins[angleidx1]
            cos2, sin2 = anchorcossins[angleidx2]
            overlap = rectOverlap(x1,y1,cos1,sin1,obj_len,obj_wid,
                                  x2,y2,cos2,sin2,obj_len,obj_wid, overlap_buffer)
        if overlap:
            overlaps[angleidx1, angleidx2, anchoridx1, anchoridx2] = True
            overlaps[angleidx2, angleidx1, anchoridx2, anchoridx1] = True
    return overlaps
        


@nb.njit(nb.void(nb.b1[:,:,:], nb.b1[:,:,:]))
def prepRough(grid, roughX):
    xc, yc, zc = grid.shape
    roughX[:] = False
    for x,y,z in np.ndindex(xc,yc,zc):
        roughX[x//3,y//3,z//3] |= grid[x,y,z]
@nb.njit(nb.b1(nb.b1[:,:,:], nb.b1[:,:,:],
               nb.i8, nb.i8, nb.i8, nb.i8, nb.i8, nb.i8))
def splitGrid(grid, roughgrid, x1,y1,z1,x2,y2,z2):
    #x1,x2,y1,y2,z1,z2 = split
    largex1 = x1//3
    smallx1 = largex1 if x1 == largex1*3 else largex1 + 1
    smallx2 = x2//3
    largex2 = smallx2 if x2 == smallx2*3 else smallx2 + 1
    largey1 = y1//3
    smally1 = largey1 if y1 == largey1*3 else largey1 + 1
    smally2 = y2//3
    largey2 = smally2 if y2 == smally2*3 else smally2 + 1
    largez1 = z1//3
    smallz1 = largez1 if z1 == largez1*3 else largez1 + 1
    smallz2 = z2//3
    largez2 = smallz2 if z2 == smallz2*3 else smallz2 + 1
    if np.any(roughgrid[smallx1:smallx2, smally1:smally2, smallz1:smallz2]):
        return True
    if not np.any(roughgrid[largex1:largex2, largey1:largey2, largez1:largez2]):
        return False
    return np.any(grid[x1:x2, y1:y2, z1:z2])
@nb.njit(nb.f8(nb.b1[:,:,:], nb.b1[:,:,:], nb.i8, nb.i8, nb.i8,
               nb.i8[:,:,:], nb.f8[:,:], nb.i8))
def useBoostedTree2(grid, roughgrid, anchorx, anchory, direction,
                   btsplits, btleaves, ntrees):
    score = 0.
    for tree in range(ntrees):
        splitidx = 0
        for depth in range(3):
            tsplit = btsplits[tree, splitidx]
            if direction == 0:
                x1 = anchorx + tsplit[0]
                x2 = anchorx + tsplit[3]
                y1 = anchory + tsplit[1]
                y2 = anchory + tsplit[4]
            else:
                x1 = anchorx + 48 - tsplit[3] ### change when changing anchor!!!
                x2 = anchorx + 48 - tsplit[0] ### change when changing anchor!!!
                y1 = anchory + 32 - tsplit[4] ### change when changing anchor!!!
                y2 = anchory + 32 - tsplit[1] ### change when changing anchor!!!
            z1 = tsplit[2]
            z2 = tsplit[5]
            splitidx = splitidx*2+2
            if splitGrid(grid, roughgrid, x1,y1,z1,x2,y2,z2):
                splitidx -= 1
        score += btleaves[tree, splitidx - 7]
        if score < btleaves[tree, 8]:
            score = -50.
            break
    return score

"""
returns the samples with the top predictions for a single lidar sweep
"""
@nb.njit(nb.i8(nb.f8[:,:], nb.i8[:], nb.f8[:,:,:,:], nb.i8[:,:,:], nb.f8[:,:],
                 nb.f8[:,:], nb.b1[:,:,:,:], nb.i8))
def predictNegs(pts, tileidxs, groundTs, btsplits, btleaves,
            pts2suppress, detections, detectioncount):
    gridshape = (anchornangles, localgridlen[0], localgridlen[1], anchorlen[2])
    grid = np.zeros(gridshape, dtype=np.bool8)
    
    nanchors = len(anchorinlocalgrid)
    ndetections = detections.shape[0]
    ntrees = btsplits.shape[0]
    
    pts2suppress_range = 2+localgridlen*anchorstep[:2]/2.
    centerpoint_grid = np.zeros(2, dtype=np.float64)
    
    roughgridshape = (localgridlen[0]//3+1, localgridlen[1]//3+1, anchorlen[2]//3+1)
    roughgrid = np.zeros(roughgridshape, dtype=np.bool8)
    
    for grnd2checkgrididx in range(grnd2checkgrid.shape[0]):
        centerx, centery = grnd2checkgrid[grnd2checkgrididx]
        # determine which suppress points are important
        centerpoint_grid[0] = grndstep[0]*(grndstart[0]+centerx+.5)
        centerpoint_grid[1] = grndstep[1]*(grndstart[1]+centery+.5)
        pts2suppressidxs  = np.abs(pts2suppress[:,0]-centerpoint_grid[0]) < pts2suppress_range[0]
        pts2suppressidxs &= np.abs(pts2suppress[:,1]-centerpoint_grid[1]) < pts2suppress_range[1]
        pts2suppress_local = pts2suppress[pts2suppressidxs].copy()
        pts2suppress_local[:,:2] -= centerpoint_grid
        npts2suppress = pts2suppress_local.shape[0]
        
        groundT = groundTs[centerx, centery]
        fillLocalGrid(pts, tileidxs, centerx, centery, groundT, grid)          
        
        for angle in range(anchornangles):
            angcos, angsin = anchorcossins[angle]
            thisgrid = grid[angle]
            prepRough(thisgrid, roughgrid)
            
            for anchoridx in range(nanchors):
                anchorx, anchory = anchorinlocalgrid[anchoridx]
                anchorcenterptx, anchorcenterpty = anchorcenterpoints[angle,anchoridx]
                suppressed = False
                for pt2suppressidx in range(npts2suppress):
                    ptx,pty,ptcos,ptsin = pts2suppress_local[pt2suppressidx]
                    if (np.hypot(ptx-anchorcenterptx, pty-anchorcenterpty) < 2. and
                             abs(ptcos*angsin - ptsin*angcos) < .8):
                        suppressed = True
                suppressed |= ((anchorcenterptx+centerpoint_grid[0])*.866 - 1.3 <
                                abs(anchorcenterpty+centerpoint_grid[1]))
                if suppressed: continue
                score1 = useBoostedTree2(thisgrid, roughgrid, anchorx, anchory, 0,
                                        btsplits, btleaves, ntrees)
                score2 = useBoostedTree2(thisgrid, roughgrid, anchorx, anchory, 1,
                                        btsplits, btleaves, ntrees)
                if score1 > score2:
                    score = score1
                    direction = 0
                else:
                    score = score2
                    direction = 1
                if score > -30: # otherwise, consider culled
                    if detectioncount < ndetections:
                        detectionidx = detectioncount
                    else:
                        detectionidx = np.random.randint(detectioncount+1)
                    if detectionidx < ndetections:
                        sample = grid[angle, anchorx:anchorx+anchorlen[0],
                                             anchory:anchory+anchorlen[1], :]
                        if direction:
                            sample = sample[::-1,::-1]
                        detections[detectionidx] = sample
                    detectioncount += 1
    return detectioncount


def prepForPredicting(fileidx, objects_to_suppress):
    data = np.fromfile(lidar_files.format(fileidx),
                       dtype=np.float32).reshape((-1,4))[:,:3]
    calib_extrinsic = calib_extrinsics[calib_map[fileidx]].copy()
    calib_extrinsic[2,3] += 1.65
    data = data.dot(calib_extrinsic[:3,:3].T) + calib_extrinsic[:3,3]
    
    # get ground
    ground = np.load(ground_planes_by_file.format(fileidx))
    pts, tileidxs = tilePoints(data, grndstart, grndstep, grndlen)
    groundTs = planes2Transforms(ground)
    
    # get suppressed objects
    suppress_start, suppress_end = np.searchsorted(objects_to_suppress[:,0],
                                                   [fileidx, fileidx+1])
    pts2suppress = objects_to_suppress[suppress_start:suppress_end, 1:3].copy()
    pts2suppress = np.zeros((suppress_end-suppress_start, 4))
    pts2suppress[:,:2] = objects_to_suppress[suppress_start:suppress_end, 1:3]
    pts2suppress[:,2] = np.cos(objects_to_suppress[suppress_start:suppress_end,0])
    pts2suppress[:,2] = np.sin(objects_to_suppress[suppress_start:suppress_end,0])
    return pts, tileidxs, pts2suppress, groundTs


if __name__ == '__main__':
    from config import training_file_start, training_file_end
    from time import time
    
    starttime = time()
    
    BT_load_file = '../dataApril19/BT29.npz'
    #np.random.seed(200)
    nnegatives = 7150
    nfilesfornegatives = 60
    
    BTstruct = np.load(BT_load_file)
    btsplits = BTstruct['splits']
    btleaves = BTstruct['leaves']
    
    files2use = np.random.choice(np.arange(training_file_start, training_file_end),
                                 nfilesfornegatives, replace=False)
    objects_to_suppress = np.load(present_boxes_file)
    
    anchoroverlaps = prepLocalNms()

    globaldetections = np.zeros(
            (nnegatives, anchorlen[0], anchorlen[1], anchorlen[2]), dtype=bool)
    detectioncount = 0
    
    for file_idx in files2use:
        # load relevant data
        data = np.fromfile(lidar_files.format(file_idx),
                           dtype=np.float32).reshape((-1,4))[:,:3]
        calib_extrinsic = calib_extrinsics[calib_map[file_idx]].copy()
        calib_extrinsic[2,3] += 1.65
        data = data.dot(calib_extrinsic[:3,:3].T) + calib_extrinsic[:3,3]
        
        # get ground
        ground = np.load(ground_planes_by_file.format(file_idx))
        pts, tileidxs = tilePoints(data, grndstart, grndstep, grndlen)
        groundTs = planes2Transforms(ground)
        
        # get suppressed objects
        suppress_start, suppress_end = np.searchsorted(objects_to_suppress[:,0],
                                                       [file_idx, file_idx+1])
        pts2suppress = objects_to_suppress[suppress_start:suppress_end, 1:3].copy()
        pts2suppress = np.zeros((suppress_end-suppress_start, 4))
        pts2suppress[:,:2] = objects_to_suppress[suppress_start:suppress_end, 1:3]
        pts2suppress[:,2] = np.cos(objects_to_suppress[suppress_start:suppress_end,0])
        pts2suppress[:,2] = np.sin(objects_to_suppress[suppress_start:suppress_end,0])
        
        detectioncount = predictNegs(pts, tileidxs, groundTs, btsplits, btleaves, pts2suppress,
                    globaldetections, detectioncount, anchoroverlaps)
        print(detectioncount)
    np.save('negs9.npy', globaldetections)
    
    print('time in minutes {:.0f}'.format((time() - starttime) / 60.))
