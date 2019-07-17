#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 2/4/19
"""

import numpy as np


lidar_files = '../object/training/velodyne/{:06d}.bin'
img_files = '../object/training/image_2/{:06d}.png'
gt_files = '../object/training_labels_orig/{:06d}.txt'
files_to_use = range(1000)

truncated_cutoffs = np.array((.15, .3, .5))
occluded_cutoffs = np.array((0, 1, 2))
height_cutoffs = np.array((40, 25, 25))
scored_classes = ('Car', 'Pedestrian', 'Cyclist')

def readGroundTruthFile(gtstr, classes_include = ('Car',)):
    gtstr = gtstr.split('\n')
    if gtstr[-1] == '': gtstr.pop()
    classes = []
    rectangles = []
    imbbs = []
    difficulties = []
    scored = []
    elevation = []
    for gtrow in gtstr:
        gtrow = gtrow.split(' ')
        this_class = gtrow[0]
        if this_class not in classes_include:
            continue
        classes.append(classes_include.index(this_class))
        # 2D rectangle on ground, in xyalw form
        gtang = 4.7124 - float(gtrow[14])
        gtang = gtang - 6.2832 if gtang > 3.1416 else gtang
        gtbox = (float(gtrow[13]), -float(gtrow[11]),
                 gtang, float(gtrow[10])/2, float(gtrow[9])/2)
        rectangles.append(gtbox)
        # 2D bounding box in image, top-bottom-left-right
        imbbs.append((float(gtrow[5]),float(gtrow[7]),float(gtrow[4]),float(gtrow[6])))
        # elevation as meters up from car bottom
        elevation.append(1.65-float(gtrow[12]))
        # copy kitti's scoring-or-ignoring strategy
        truncation = float(gtrow[1])
        occlusion = int(gtrow[2])
        height = float(gtrow[7]) - float(gtrow[5]) # image bb height
        difficulty = 0
        for dd in range(3):
            not_met = truncation > truncated_cutoffs[dd]
            not_met |= occlusion > occluded_cutoffs[dd]
            not_met |= height < height_cutoffs[dd]
            if not_met: difficulty = dd + 1
        difficulties.append(difficulty)
        scored.append(difficulty < 3 and this_class in scored_classes)
    return classes, rectangles, imbbs, difficulties, scored, elevation
        
def readGroundTruthFileTracking(gtstr, nfiles, classes_include = ('Car',)):
    gtstr = gtstr.split('\n')
    if gtstr[-1] == '': gtstr.pop()
    outputs = [[] for file_idx in xrange(nfiles)]
    base_output = {'class':None,'box':None,'imbb':None,'difficulty':None,
                   'scored':None,'elevation':None,'id':None}
    for gtrow in gtstr:
        gtrow = gtrow.split(' ')
        file_idx = int(gtrow[0])
        # track id = int(gtrow[1])
        row_output = base_output.copy()
        row_output['id'] = int(gtrow[1])
        gtrow = gtrow[2:]
        this_class = gtrow[0]
        if this_class not in classes_include:
            continue
        row_output['class'] = classes_include.index(this_class)
        # 2D rectangle on ground, in xyalw form
        gtang = 4.7124 - float(gtrow[14])
        gtang = gtang - 6.2832 if gtang > 3.1416 else gtang
        gtbox = (float(gtrow[13]), -float(gtrow[11]), gtang,
                 float(gtrow[10])/2, float(gtrow[9])/2)
        row_output['box'] = gtbox
        # 2D bounding box in image, top-bottom-left-right
        row_output['imbb'] = (float(gtrow[5]),float(gtrow[7]),
                              float(gtrow[4]),float(gtrow[6]))
        # elevation as meters up from car bottom
        row_output['elevation'] = 1.65-float(gtrow[12])
        # copy kitti's scoring-or-ignoring strategy
        truncation = float(gtrow[1])
        occlusion = int(gtrow[2])
        height = float(gtrow[7]) - float(gtrow[5]) # image bb height
        difficulty = 0
        for dd in range(3):
            not_met = truncation > truncated_cutoffs[dd]
            not_met |= occlusion > occluded_cutoffs[dd]
            not_met |= height < height_cutoffs[dd]
            if not_met: difficulty = dd + 1
        row_output['difficulty'] = difficulty
        row_output['scored'] = difficulty < 3 and this_class in scored_classes
        outputs[file_idx].append(row_output)
    return outputs

"""
    prunes estimates based on whether kitti will actually have gt
    also formats in standard kitti output
    input: 2d array of boxes, (x,y,a (radians),l,w,z (bottom),h,score)
    output: string version
"""
def formatForKittiScore(ests, calib_project, imgshape):
    output_text_format = '{:s} {:.2f} {:d}' + ' {:.2f}'*13
    outputstr = []
    for msmt in ests:
        cos,sin = np.cos(msmt[2]), np.sin(msmt[2])
        corners = np.zeros((8,3))
        corners[0,:2] = msmt[:2] + (cos*msmt[3]+sin*msmt[4], sin*msmt[3]-cos*msmt[4])
        corners[1,:2] = msmt[:2] + (cos*msmt[3]-sin*msmt[4], sin*msmt[3]+cos*msmt[4])
        corners[2,:2] = msmt[:2] - (cos*msmt[3]+sin*msmt[4], sin*msmt[3]-cos*msmt[4])
        corners[3,:2] = msmt[:2] - (cos*msmt[3]-sin*msmt[4], sin*msmt[3]+cos*msmt[4])
        corners[:4,2] = msmt[5]
        corners[4:,:2] = corners[:4,:2]
        corners[4:,2] = msmt[5]+msmt[6]
        msmt_corners = corners.dot(calib_project[:3,:3].T) + calib_project[:3,3]
        msmt_corners = msmt_corners[:,:2] / msmt_corners[:,2:]
        topfull, leftfull = np.min(msmt_corners, axis=0)
        bottomfull, rightfull = np.max(msmt_corners, axis=0)
        top, bottom, left, right = (max(topfull, 0), min(bottomfull, imgshape[0]),
                                    max(leftfull, 0), min(rightfull, imgshape[1]))
        imbb_area = (bottom - top) * (right - left)
        full_imbb_area = (bottomfull - topfull) * (rightfull - leftfull)
        truncation_level = 1 - imbb_area / full_imbb_area
        if truncation_level > .4:
            continue
        if bottom-top < 22:
            continue
        observation_angle = np.pi/2. - np.arctan2(msmt[1], msmt[0])
        rotation_angle = np.pi/2. - msmt[2]
        output = ('Car',0.,0,observation_angle,left,top,right,bottom,
                  msmt[6],msmt[4]*2,msmt[3]*2,-msmt[1],-msmt[5]+1.65,msmt[0],
                  rotation_angle, msmt[7])
        outputstr.append(output_text_format.format(*output))
    return '\n'.join(outputstr)
        



## check out 3D dimensions of objects by class
if __name__ == '__main__' and False:
    boxes = {}
    for file_idx in files_to_use:
        if gt_files is not None:
            with open(gt_files.format(file_idx), 'r') as fd: gtstr = fd.read()
            gtstr = gtstr.split('\n')
            if gtstr[-1] == '': gtstr = gtstr[:-1]
            gtboxes = []
            for gtrow in gtstr:
                gtrow = gtrow.split(' ')
                if gtrow[0] == 'DontCare' or gtrow[0]=='Misc': continue
                boxlen = float(gtrow[10])
                boxwid = float(gtrow[9])
                boxheight = float(gtrow[8])
                boxclass = gtrow[0]
                if boxclass in boxes.keys():
                    boxes[boxclass].append((boxlen, boxwid, boxheight, file_idx))
                else:
                    boxes[boxclass] = [(boxlen, boxwid, boxheight, file_idx)]
    
    cats = boxes.keys()
    for cat in cats:
        dims = np.array(boxes[cat])
        print cat
        print np.min(dims[:,:3], axis=0)
        print np.max(dims[:,:3], axis=0)
    
    
## check out position distribution
if __name__ == '__main__' and False:
    boxes = []
    for file_idx in files_to_use:
        with open(gt_files.format(file_idx), 'r') as fd: gtstr = fd.read()
        gt = readGroundTruthFile(gtstr, classes_include = ('Car',))
        boxes = boxes + [gt[1][idx] for idx in range(len(gt[0])) if gt[4][idx]]
    boxes = np.array(boxes)
    

## check out characteristics that decide object difficulty
if __name__ == '__main__' and False:
    data = []
    
    for file_idx in files_to_use:
        if gt_files is not None:
            with open(gt_files.format(file_idx), 'r') as fd: gtstr = fd.read()
            gtstr = gtstr.split('\n')
            if gtstr[-1] == '': gtstr = gtstr[:-1]
            for gtrow in gtstr:
                gtrow = gtrow.split(' ')
                if gtrow[0] == 'Car' or gtrow[0] == 'Van':
                    truncation = float(gtrow[1])
                    occlusion = int(gtrow[2])
                    height = float(gtrow[7]) - float(gtrow[5])
                    distance = float(gtrow[13])
                    data.append((truncation, occlusion, height, gtrow[0]=='Van', distance))
    data = np.array(data)
    difficulties = np.zeros(len(data), dtype=np.uint8)
    for difficulty in range(3):
        this_hard = data[:,0] > truncated_cutoffs[difficulty]
        this_hard |= data[:,1] > occluded_cutoffs[difficulty] + .001
        this_hard |= data[:,2] < height_cutoffs[difficulty]
        difficulties[this_hard] = difficulty+1
    difficulties[data[:,3] > .1] = 3

    
    
    
## plot objects w/ ground truth
if __name__ == '__main__' and False:
    from cv2 import imshow, waitKey, destroyWindow
    from plotStuff import base_image, plotRectangle, drawLine, grayer
    from plotStuff import reference as plotReference
    from calibs import calib_map, view_by_day
    from imageio import imread
    
    def clear(): destroyWindow('a') # for convenience
    
    view_distance = 30. # 2*x ahead, x to either side, meters
    cutoffs = np.array([[.15,0,-40],[.3,1,-25],[.5,2,-25]])
    colors_by_difficulty = np.array([[255,0,0],[200,200,0],[0,200,200],[0,0,255]]).astype(np.uint8)
    classes = ['Car', 'Van']
    
    for file_idx in files_to_use:
        view_angle = view_by_day[calib_map[file_idx]]
        img = imread(img_files.format(file_idx))[:,:,::-1]
        img = grayer(img)
        
        plot_img = base_image.copy()
        # draw lines to show visualized part of map
        linestart = (639, 320)
        lineend = plotReference(int(.95/view_angle), .95)
        drawx, drawy = drawLine(linestart[0], linestart[1], lineend[0], lineend[1])
        lineend = plotReference(int(.95/view_angle), -.95)
        drawx2, drawy2 = drawLine(linestart[0], linestart[1], lineend[0], lineend[1])
        plot_img[drawx, drawy] = [240,240,240]
        plot_img[drawx2, drawy2] = [240,240,240]
    
        with open(gt_files.format(file_idx), 'r') as fd: gtstr = fd.read()
        gtstr = gtstr.split('\n')
        if gtstr[-1] == '': gtstr = gtstr[:-1]
        for gtrow in gtstr:
            gtrow = gtrow.split(' ')
            if gtrow[0] not in classes: continue
            
            gtbox = (float(gtrow[13]), -float(gtrow[11]),
                1.5708-float(gtrow[14]), float(gtrow[10])/2, float(gtrow[9])/2)
            box = np.array(gtbox)
            box[[0,1,3,4]] /= view_distance
            box[:2] = plotReference(*box[:2])
            box[3:] *= 320
            
            monitor_vals = (float(gtrow[1]), int(gtrow[2]), float(gtrow[5])-float(gtrow[7]))
            difficulty = 0
            for ddd in range(3):
                if any(monitor_vals > cutoffs[ddd]):
                    difficulty = ddd + 1
            color = colors_by_difficulty[difficulty]
            
            plotRectangle(plot_img, box, color)
            
        display_img = np.zeros((plot_img.shape[0]+img.shape[0], img.shape[1], 3),
                               dtype=np.uint8)
        display_img[:plot_img.shape[0], (img.shape[1]-plot_img.shape[1])//2:
                    (img.shape[1]+plot_img.shape[1])//2] = plot_img
        display_img[plot_img.shape[0]:] = img
        imshow('a', display_img);
        if waitKey(5000) == ord('q'):
            break