# -*- coding: utf-8 -*-
"""
last mod 7/16/19

much of this code is highly optimized to the point of being pretty unreadable
The algorithm has this general layout
- modify boosted tree for fast use, generate lookup table of detection overlaps
- for each detection
    - transform lidar points to local ground coordinates & order by 3mx3m tile
    - for each 3mx3m tile (where kitti has annotations) and angle
        - fill in binary voxel array with lidar presence
        - take cumulative sum in all directions (integral image)
        - for subgrids of 4 voxel offsets (1/8m) in this tile and 4 cardinal directions
            - run initial boosted tree on integral image
            - if initial score is high enough, for each voxel offset
                - run rest of boosted tree
            - store highest-scoring location and cardinal direction
        - keep small number of highest-scoring detections from this tile
            - using nms lookups
    - use normal nms to maintain sorted list of the top detections
"""
import numpy as np
import numba as nb
from math import floor as mathfloor
from time import time
from ctypes import CDLL, RTLD_GLOBAL, c_void_p

from config import grndstart, grndstep, grndlen
from config import anchorstart, anchorstep, anchorlen
from config import grnd2checkgrid
from config import anchorinlocalgrid_strided, anchorinlocalgrid
from config import anchorangles as anchorangles
from config import anchornangles as anchornangles
from ground import planes2Transforms

localgridlen = (72,72)
anchorinlocalgrid = anchorinlocalgrid + (0,8)
anchorinlocalgrid_strided = anchorinlocalgrid_strided + (0,8)

anchorcossins = np.column_stack((np.cos(anchorangles), np.sin(anchorangles)))

anchorcenterpoints = (anchorinlocalgrid + (24,16) - 36)*anchorstep[:2]
anchorcenter2 = np.einsum(anchorcenterpoints, [0,1], anchorcossins[:,0], [2], [2,0,1])
anchorcenter2[:,:,0] -= np.outer(anchorcossins[:,1], anchorcenterpoints[:,1])
anchorcenter2[:,:,1] += np.outer(anchorcossins[:,1], anchorcenterpoints[:,0])
anchorcenterpoints = anchorcenter2

"""
transforms pts to ground plane
pts outside of ground region are kept in array, but not indexed by tileidxs
"""
@nb.njit(nb.void(nb.f8[:,::1], nb.f8[:,::1], nb.i8[:], nb.f8[:,:,:,::1]))
def tilePoints(pts, newpts, tileidxs, groundTs):
    npts = pts.shape[0]
    maxtileval = grndlen[0]*grndlen[1] # 1 more than maximum location
    grndidx = np.full(pts.shape[0], maxtileval, dtype=np.int64)
    groundTs[:,:,2,:] *= 8
    for ptidx in range(npts):
        pt = pts[ptidx]
        tilex = int(mathfloor(pt[0]/grndstep[0])) - grndstart[0]
        tiley = int(mathfloor(pt[1]/grndstep[1])) - grndstart[1]
        if tilex >= 0 and tiley >= 0 and tilex < grndlen[0] and tiley < grndlen[1]:
            groundT = groundTs[tilex,tiley]
            #pt = groundTs[tilex,tiley,:3,:3].dot(pt) + groundTs[tilex,tiley,:3,3]
            ptz = mathfloor(groundT[2,:3].dot(pt) + groundT[2,3])-1
            # go ahead and transform ptz
            #ptz = mathfloor(pt[2]*8)-1
            if ptz >= 0 and ptz < anchorlen[2]:
                grndidx[ptidx] = tilex*grndlen[1]+tiley
                pt[:2] += groundT[:2,3]
                pt[2] = ptz
    grndorder = np.argsort(grndidx)
    # tileidxs = np.searchsorted(grndidx[grndorder], range(grndlen[0]*grndlen[1]+1))
    # newpts[:tileidxs[-1]] = pts[grndorder[:tileidxs[-1]]]
    # below is single loop version of the above
    currenttile = 0
    tileidxs[0] = 0
    for ptidx in range(npts):
        oldptidx = grndorder[ptidx]
        newpts[ptidx] = pts[oldptidx]
        tileval = grndidx[oldptidx]
        while tileval > currenttile:
            currenttile += 1
            tileidxs[currenttile] = ptidx
        if currenttile == maxtileval:
            break

""" takes pre-transformed points """
@nb.njit(nb.void(nb.f8[:,::1], nb.i8[:], nb.i8, nb.i8, nb.f8,
                 nb.f8, nb.u1[:,:,::1]))
def fillLocalGrid(pts, tileidxs, tilex, tiley, cos, sin, grid):
    grid[:] = 0
    cos8 = cos*8
    sin8 = sin*8
    #offz = 1 - anchorstart[2]
    for tilex2, tiley2 in np.ndindex(5,5):
        tile = (tilex+tilex2-2)*grndlen[1] + tiley+tiley2-2
        pts_idxstart = tileidxs[tile]
        pts_idxend = tileidxs[tile+1]
        offx = (cos8*(tilex2-2) + sin8*(tiley2-2))*3. + 36
        offy = (cos8*(tiley2-2) - sin8*(tilex2-2))*3. + 36
        for ptsidx in range(pts_idxstart, pts_idxend):
            pt = pts[ptsidx]
            #z = int(mathfloor(pt[2]*8)) - 1 # taken care of in tilePoints
            #if z >= 0 and z < 20:
            x = int(mathfloor(cos8*pt[0] + sin8*pt[1] + offx))
            y = int(mathfloor(cos8*pt[1] - sin8*pt[0] + offy))
            if x >= 0 and x < 72 and y >= 0 and y < 72:
                z = int(pt[2])
                grid[x,y,z] = 1

lib = CDLL('./cumsum3d.so', RTLD_GLOBAL)
makeIntegralGrid = lib.cumsum3d
makeIntegralGrid.argtypes = [c_void_p, c_void_p]

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

@nb.njit(nb.b1[:,:,:,:,:,::1]())
def prepLocalNms():
    nanchors = anchorinlocalgrid.shape[0]
    overlaps = np.zeros((anchornangles//2,nanchors,2, anchornangles//2,nanchors,2),
                        dtype=np.bool8)
    obj_len = 2.
    obj_wid = 1.#.9
    overlap_buffer = .4#.3
    obj_hypot = np.hypot(obj_len, obj_wid)
    for angleidx1, angleidx2, anchoridx1, anchoridx2, dir1, dir2 in np.ndindex(
                    anchornangles//2, anchornangles//2, nanchors, nanchors,2,2):
        if angleidx2 < angleidx1:
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
            if dir1:
                cos1, sin1 = -sin1, cos1
            if dir2:
                cos2, sin2 = -sin2, cos2
            overlap = rectOverlap(x1,y1,cos1,sin1,obj_len,obj_wid,
                                  x2,y2,cos2,sin2,obj_len,obj_wid, .5)#.45)
        if overlap:
            overlaps[angleidx1, anchoridx1, dir1, angleidx2, anchoridx2, dir2] = True
            overlaps[angleidx2, anchoridx2, dir2, angleidx1, anchoridx1, dir1] = True
    return overlaps
    
@nb.njit(nb.f8(nb.u4[:], nb.i8, nb.i8[:,:,::1], nb.f8[:,::1], nb.i8, nb.i8, nb.f8))
def useBoostedTree(grid, anchoroffset, btsplits, btleaves, tree1,tree2, score):
    for tree in range(tree1,tree2):
        splitidx = 0
        for depth in range(3):
            startp, xmove, ymove, zmove = btsplits[tree, splitidx]
            startp = startp + anchoroffset
            npoints = (grid[startp+zmove] - grid[startp] + grid[startp+ymove] -
                       grid[startp+ymove+zmove] + grid[startp+xmove] - 
                       grid[startp+xmove+zmove] - grid[startp+xmove+ymove] +
                       grid[startp+xmove+ymove+zmove])
            splitidx <<= 1
            splitidx += 1 if npoints else 2
        score += btleaves[tree, splitidx - 7]
        if score < btleaves[tree,8]:
            return -50. + tree
    return score



@nb.njit(nb.void(nb.f8[:,::1], nb.f8[:,:,:,::1], nb.i8[:,:,:,::1], nb.f8[:,::1],
                 nb.i8[:,:], nb.f8[:], nb.b1[:,:,:,:,:,::1]))
def predict(pts, groundTs, btsplits, btleaves, globaldetections, globalscores,
            anchoroverlaps):
    newpts = np.zeros(pts.shape, dtype=np.float64)
    tileidxs = np.zeros(grndlen[0]*grndlen[1]+1, dtype=np.int64)
    tilePoints(pts, newpts, tileidxs, groundTs)
    pts = newpts # name change for convenience
    
    gridshape = (73,73,21)
    #grid = np.empty(gridshape, dtype=np.int16)
    grid = np.empty((72,72,20), dtype=np.uint8)
    cumgrid = np.zeros(gridshape, dtype=np.uint32)
    gridflat = cumgrid.reshape((-1,))
    
    nanchors_strided = anchorinlocalgrid_strided.shape[0]
    ntrees = btsplits.shape[1]
    guysforotherangle = np.zeros((36,4), dtype=np.float64)
    
    nlocaldetections = 4
    localdetections = np.zeros((nlocaldetections, 3), dtype=np.int64)
    localscores = np.zeros(nlocaldetections, dtype=np.float64)
    nglobaldetections = globaldetections.shape[0]
    
    for grnd2checkgrididx in range(grnd2checkgrid.shape[0]):
        centerx, centery = grnd2checkgrid[grnd2checkgrididx]
        localscores[:] = -100.
        
        for angle in range(0,anchornangles//2):
            cos, sin = anchorcossins[angle]
            fillLocalGrid(pts, tileidxs, centerx, centery, cos, sin, grid)
            makeIntegralGrid(grid.ctypes.data, cumgrid.ctypes.data)
            altangle = angle%2 > 0
            
            for anchoridx in range(nanchors_strided):
                anchorx, anchory = anchorinlocalgrid_strided[anchoridx]
                bestlocalscore = -30
                bestlocalanchorx = 0
                bestlocalanchory = 8
                bestlocaldirection = 0
                anchoroffset = anchorx*gridshape[1]*gridshape[2]+anchory*gridshape[2] # TEMP 7/12
                
                for direction in range(4):
                    if altangle:
                        score = guysforotherangle[anchoridx,direction]
                    else:
                        score = useBoostedTree(gridflat, anchoroffset,
                                               btsplits[direction], btleaves,
                                               0,10, 0.)
                        guysforotherangle[anchoridx,direction] = score
                    if score > -30:
                        for anchorx2 in range(anchorx-1, anchorx+3):
                            for anchory2 in range(anchory-1, anchory+3):
                                anchor2off = (anchorx2*gridshape[1]+anchory2)*gridshape[2]
                                score2 = useBoostedTree(gridflat, anchor2off,
                                                        btsplits[direction],btleaves,
                                                        10,ntrees,score)
                                if score2 > bestlocalscore:
                                    bestlocalscore = score2
                                    bestlocalanchorx = anchorx2
                                    bestlocalanchory = anchory2
                                    bestlocaldirection = direction
                
                #keep top local detections, while also performing nms
                anchoridx2 = bestlocalanchorx*24 + bestlocalanchory-8
                dirover2 = bestlocaldirection//2
                notsuppressed = True
                for detectionidx in range(nlocaldetections):
                    otherangle, otheranchor, otherdir = localdetections[detectionidx]
                    if anchoroverlaps[angle, anchoridx2, dirover2,
                                      otherangle, otheranchor, otherdir//2]:
                        notsuppressed &= localscores[detectionidx] < bestlocalscore
                if notsuppressed:
                    for detectionidx in range(nlocaldetections):
                        otherangle, otheranchor, otherdir = localdetections[detectionidx]
                        if anchoroverlaps[angle, anchoridx2, dirover2,
                                          otherangle, otheranchor, otherdir//2]:
                            localscores[detectionidx] = -100.
                    detectiontoreplace = np.argmin(localscores)
                    if bestlocalscore > localscores[detectiontoreplace]:
                        localdetections[detectiontoreplace] = (angle, anchoridx2,
                                                               bestlocaldirection)
                        localscores[detectiontoreplace] = bestlocalscore
                    
        # add top local detections to global sorted detections
        # basically inplace merge sort step
        localorder = np.argsort(localscores)
        fromglobalidx = 0
        toglobalidx = -nlocaldetections
        for detectionidx in range(nlocaldetections):
            score = localscores[localorder[detectionidx]]
            while fromglobalidx < nglobaldetections and score > globalscores[fromglobalidx]:
                if toglobalidx >= 0:
                    globalscores[toglobalidx] = globalscores[fromglobalidx]
                    globaldetections[toglobalidx] = globaldetections[fromglobalidx]
                toglobalidx += 1
                fromglobalidx += 1
            if toglobalidx >= 0:
                angle, anchoridx, direction = localdetections[localorder[detectionidx]]
                globalscores[toglobalidx] = score
                globaldetections[toglobalidx] = (centerx, centery, angle,
                                                 anchoridx, direction)
            toglobalidx += 1
            

def grayer(img): return ((img.astype(float)-128)*.75 + 128).astype(np.uint8)



if __name__ == '__main__':
    from imageio import imread
    from cv2 import imshow, waitKey, destroyWindow
    from evaluate import MetricAvgPrec
    from calibs import calib_extrinsics, calib_map, view_by_day, calib_projections
    from plotStuff import base_image, plotRectangle, drawLine, plotPoints
    from plotStuff import reference as plotReference
    from kittiGT import readGroundTruthFile, formatForKittiScore
    
    from myio import lidar_files, gt_files, img_files, output_files
    from myio import ground_planes_by_file, model_file, files2use
    
    ndetections = 32
    
    # alter tree format for fast lookups
    # four models are created, one for each cardinal direction
    # each model is "flattened" to look up the integral image in a flattened array
    # (the array's size must be known)
    # the nice thing about this representation is that the positional offset for
    # each detection requires a single add, and only 4 numbers define a split
    BTstruct = np.load(model_file)
    btsplits = BTstruct['splits'][:].copy()
    btleaves = BTstruct['leaves'][:].copy()
    ntrees = btsplits.shape[0]
    btsplitsall = np.zeros((4,ntrees,7,6), dtype=np.int64)
    btsplitsall[0] = btsplits[:,:7] # forward
    btsplitsall[1] = btsplits[:,:7] # backward
    btsplitsall[1,:,:,0] = 48 - btsplits[:,:7,3]
    btsplitsall[1,:,:,1] = 32 - btsplits[:,:7,4]
    btsplitsall[1,:,:,3] = 48 - btsplits[:,:7,0]
    btsplitsall[1,:,:,4] = 32 - btsplits[:,:7,1]
    btsplitsall[2] = btsplits[:,:7] # left
    btsplitsall[2,:,:,0] =  8 + btsplits[:,:7,1]
    btsplitsall[2,:,:,1] = 40 - btsplits[:,:7,3]
    btsplitsall[2,:,:,3] =  8 + btsplits[:,:7,4]
    btsplitsall[2,:,:,4] = 40 - btsplits[:,:7,0]
    btsplitsall[3] = btsplits[:,:7] # right
    btsplitsall[3,:,:,0] = 40 - btsplits[:,:7,4]
    btsplitsall[3,:,:,1] = -8 + btsplits[:,:7,0]
    btsplitsall[3,:,:,3] = 40 - btsplits[:,:7,1]
    btsplitsall[3,:,:,4] = -8 + btsplits[:,:7,3]
    btsplitsflat = np.zeros((4,ntrees,7,4), dtype=np.int64)
    btsplitsflat[:,:,:,0] = (btsplitsall[:,:,:,0] * 73*21 +
                             btsplitsall[:,:,:,1] * 21 +
                             btsplitsall[:,:,:,2])
    btsplitsflat[:,:,:,1] = (btsplitsall[:,:,:,3]-btsplitsall[:,:,:,0])*73*21
    btsplitsflat[:,:,:,2] = (btsplitsall[:,:,:,4]-btsplitsall[:,:,:,1])*21
    btsplitsflat[:,:,:,3] = (btsplitsall[:,:,:,5]-btsplitsall[:,:,:,2])
    
    
    predicttime = 0.
    
    anchoroverlaps = prepLocalNms()
    
    globaldetections = np.zeros((ndetections, 5), dtype=int)
    globalscores = np.zeros(ndetections)
    
    metricAvgPrec = MetricAvgPrec()

    for file_idx in files2use:
        globalscores[:] = -100.
        
        data = np.fromfile(lidar_files.format(file_idx),
                           dtype=np.float32).reshape((-1,4))[:,:3]
        calib_extrinsic = calib_extrinsics[calib_map[file_idx]].copy()
        calib_extrinsic[2,3] += 1.65
        calib_projection = calib_projections[calib_map[file_idx]]
        calib_projection = calib_projection.dot(np.linalg.inv(calib_extrinsic))
        view_angle = view_by_day[calib_map[file_idx]]
        data = data.dot(calib_extrinsic[:3,:3].T) + calib_extrinsic[:3,3]
        img = imread(img_files.format(file_idx))[:,:,::-1]
        img = grayer(img)
    
        ground = np.load(ground_planes_by_file.format(file_idx))
        groundTs = planes2Transforms(ground)
        # just use ground elevation, no tilt
        groundTs[:,:,:3,:3] == np.eye(3)
        groundTs[:,:,0,3] = -(np.arange(grndlen[0])[:,None]+grndstart[0]+.5)*grndstep[0]
        groundTs[:,:,1,3] = -(np.arange(grndlen[1])[None,:]+grndstart[1]+.5)*grndstep[1]
        groundTs[:,:,2,0] = ground[:,:,0]
        groundTs[:,:,2,1] = ground[:,:,1]
        groundTs[:,:,2,3] = -ground[:,:,3]
    
        data2 = data.copy()
        starttime = time()
        predict(data2, groundTs, btsplitsflat, btleaves,
                     globaldetections, globalscores, anchoroverlaps)
        if file_idx != files2use[0]:
            predicttime += time() - starttime
        
        # convert detections to global positions
        detections = np.zeros((ndetections, 5))
        detections[:,3] = 2.
        detections[:,4] = .88
        for detectionidx in range(ndetections):
            centerx, centery, angle, anchoridx, direction = globaldetections[detectionidx]
            groundT = groundTs[centerx, centery]
            anchorx, anchory = anchorinlocalgrid[anchoridx]
            anchorptx = (anchorx+anchorlen[0]*.5 - localgridlen[0]//2 )*anchorstep[0]
            anchorpty = (anchory+anchorlen[1]*.5 - localgridlen[1]//2 )*anchorstep[1]
            anchorcos, anchorsin = anchorcossins[angle]
            localx = anchorcos*anchorptx - anchorsin*anchorpty
            localy = anchorcos*anchorpty + anchorsin*anchorptx
            globalpt = np.linalg.solve(groundT, (localx, localy, 0., 1.))
            globalangle = anchorangles[angle] +\
                          (direction%2)*np.pi + (direction//2)*np.pi/2
            detections[detectionidx,:3] = (globalpt[0], globalpt[1], globalangle)
            
        # perform nms on top detections
        notsuppress = np.ones(ndetections, dtype=bool)
        #scoreorder = np.argsort(globalscores) globalscores is ordered increasing
        for detectionid1 in range(ndetections-1,-1,-1):
            if notsuppress[detectionid1]:
                x1,y1,angle,l1,w1 = detections[detectionid1]
                c1,s1 = (np.cos(angle), np.sin(angle))
                for detectionid2 in range(detectionid1-1,-1,-1):
                    if notsuppress[detectionid2]:
                        x2,y2,angle,l2,w2 = detections[detectionid2]
                        c2,s2 = (np.cos(angle), np.sin(angle))
                        notsuppress[detectionid2] = not rectOverlap(x1,y1,c1,s1,l1,w1,
                                                             x2,y2,c2,s2,l2,w2, .3)
        detections = detections[notsuppress]
        globalscoresS = globalscores[notsuppress]
        globaldetectionsS = globaldetections[notsuppress]
        
        # output to file in kitti scoring format
        detections4output = np.zeros((detections.shape[0], 8))
        for detectionid1 in range(detections.shape[0]):
            x1,y1,angle,l1,w1 = detections[detectionid1,:5]
            detections4output[detectionid1,:5] = detections[detectionid1,:5]
            centerx, centery = globaldetectionsS[detectionid1, :2]
            groundtile = ground[centerx, centery]
            height = groundtile[3] - x1*groundtile[0] - y1*groundtile[1]
            detections4output[detectionid1,5:] = (height,1.7, globalscoresS[detectionid1])
        detections4output = detections4output[detections4output[:,7]>-20]
        outputstr = formatForKittiScore(detections4output, calib_projection, img.shape)
        if output_files is not None:
            outputfname = output_files.format(file_idx)
            with open(outputfname, 'w') as outputfile: outputfile.write(outputstr)
        
        
        with open(gt_files.format(file_idx), 'r') as fd: gtstr = fd.read()
        gtstuff = readGroundTruthFile(gtstr, ('Car', 'Van'))
        
        estsformetric = []; scoresformetric = []
        for outputline in outputstr.split('\n'):
            if outputline=='': continue
            outputline = outputline.split(' ')
            gtang = 4.7124 - float(outputline[14])
            gtang = gtang - 6.2832 if gtang > 3.1416 else gtang
            gtbox = (float(outputline[13]), -float(outputline[11]),
                     gtang, float(outputline[10])/2, float(outputline[9])/2)
            scoresformetric.append(float(outputline[15]))
            estsformetric.append(gtbox)
        metricAvgPrec.add(np.array(gtstuff[1]), np.array(gtstuff[4]), np.array(gtstuff[3]),
                          np.array(estsformetric), np.array(scoresformetric))
        
#        # plot measurements and compare to ground truth boxes
#        plot_img = base_image.copy()
#        # draw lines to show visualized part of map
#        linestart = (639, 320)
#        lineend = plotReference(int(.95/view_angle), .95)
#        drawx, drawy = drawLine(linestart[0], linestart[1], lineend[0], lineend[1])
#        lineend = plotReference(int(.95/view_angle), -.95)
#        drawx2, drawy2 = drawLine(linestart[0], linestart[1], lineend[0], lineend[1])
#        plot_img[drawx, drawy] = [240,240,240]
#        plot_img[drawx2, drawy2] = [240,240,240]
#        plot_img2 = plot_img.copy()
#        # add ground truth boxes, if available:
#        for gtbox, gtscored in zip(gtstuff[1], gtstuff[4]):
#            box = np.array(gtbox)
#            box[[0,1,3,4]] /= 30.
#            box[:2] = plotReference(*box[:2])
#            box[3:] *= 320
#            if gtscored:
#                plotRectangle(plot_img2, box, [0,0,210])
#            else:
#                plotRectangle(plot_img2, box, [30, 80, 255])
#        
#        # plot lidar points post ground removal
#        pts = np.zeros(data.shape, dtype=np.float64)
#        tileidxs = np.zeros(grndlen[0]*grndlen[1]+1, dtype=np.int64)
#        tilePoints(data, pts, tileidxs, groundTs)
#        heights = np.zeros(pts.shape[0])
#        for grndxidx, grndyidx in np.ndindex(*grndlen):
#            ptsintilestart = tileidxs[grndxidx*grndlen[1]+grndyidx]
#            ptsintileend = tileidxs[grndxidx*grndlen[1]+grndyidx+1]
#            groundtile = ground[grndxidx, grndyidx]
#            ptsintile = pts[ptsintilestart:ptsintileend]
#            heights[ptsintilestart:ptsintileend] = ptsintile.dot(groundtile[:3])-groundtile[3]
#        plotpoints_x = pts[heights>.2,0] / 30.
#        plotpoints_y = pts[heights>.2,1] / 30.
#        include_scatter = ((plotpoints_x > .01) & (plotpoints_x < 1.99) &
#                           (plotpoints_y > -.99) & (plotpoints_y < .99))
#        plotpoints_x, plotpoints_y = plotReference(plotpoints_x[include_scatter],
#                                                   plotpoints_y[include_scatter])
#        plotpoints_x = plotpoints_x.astype(int)
#        plotpoints_y = plotpoints_y.astype(int)
#        plotPoints(plot_img,  plotpoints_x, plotpoints_y, ((0,0),), (0.,0.,0.))
#        plotPoints(plot_img2, plotpoints_x, plotpoints_y, ((0,0),), (0.,0.,0.))
#                
#        for detection, score in zip(detections, globalscoresS):
#            box = detection.copy()
#            box[[0,1,3,4]] /= 30.
#            box[:2] = plotReference(*box[:2])
#            box[3:] *= 320
#            if score < -20: continue
#            score = 1./(1+np.exp(-score))
#            colorb = int(200 + 55*score**.5)
#            colorg = 255 - int(55*score**.5)
#            plotRectangle(plot_img, box, (colorb, colorg, 0))
#            
#        eeee = plot_img.astype(int)+plot_img2.astype(int)
#        eeee += np.maximum(plot_img, plot_img2) - np.max(eeee, axis=2)[:,:,None]
#        eeee = np.maximum(eeee, 0)
#        plot_img = eeee.astype(np.uint8)
#        # put the plot on top of the camera image to view, display for 3 seconds      
#        display_img = np.zeros((plot_img.shape[0]+img.shape[0], img.shape[1], 3),
#                               dtype=np.uint8)
#        display_img[:plot_img.shape[0], (img.shape[1]-plot_img.shape[1])//2:
#                    (img.shape[1]+plot_img.shape[1])//2] = plot_img
#        display_img[plot_img.shape[0]:] = img
#        imshow('a', display_img);
#        qkey = waitKey(100)
#        if qkey == ord('q'): break
#    destroyWindow('a')
    print("predict time {:.2f}".format(predicttime/(len(files2use)-1)))
    print("avg precision (not kitti's) {:.3f},{:.3f},{:.3f}".format(
                                                    *metricAvgPrec.calc()[:3]))