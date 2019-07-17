#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 6/4/19
"""

import numpy as np
from scipy.optimize import linear_sum_assignment 
from sklearn.metrics import average_precision_score

overlapres = 50
overlapbox = np.mgrid[:float(overlapres), :float(overlapres)]
overlapbox += .5
overlapbox *= 2./overlapres
overlapbox -= 1
overlapbox = overlapbox.transpose((1,2,0))
#def soMetricIoU(boxa, boxb):
#    boxarel = boxa[:3] - boxb[:3]
#    cb,sb = np.cos(boxb[2]), np.sin(boxb[2])
#    ca,sa = np.cos(boxarel[2]), np.sin(boxarel[2])
#    boxarelx = boxarel[0]*cb + boxarel[1]*sb
#    boxarely = boxarel[1]*cb - boxarel[0]*sb
#    grid = overlapbox * boxa[3:5] + (boxarelx, boxarely)
#    gridx = grid[:,:,0] * ca - grid[:,:,1] * sa
#    gridy = grid[:,:,1] * ca + grid[:,:,1] * sa
#    intersection = np.sum((abs(gridx) < boxb[3]) & (abs(gridy) < boxb[4]))
#    return .7 - float(intersection) / overlapres**2
    
def soMetricIoU(boxa, boxb):
    relx = boxa[0]-boxb[0]
    rely = boxa[1]-boxb[1]
    ca, sa = np.cos(boxa[2]), np.sin(boxa[2])
    cb, sb = np.cos(boxb[2]), np.sin(boxb[2])
    la,wa = boxa[3:5]
    lb,wb = boxb[3:5]
    R = np.array([[la/lb*(ca*cb+sa*sb), wa/lb*(ca*sb-cb*sa)],
                  [la/wb*(cb*sa-ca*sb), wa/wb*(ca*cb+sa*sb)]])
    t = np.array([(cb*relx + sb*rely)/lb, (cb*rely - sb*relx)/wb])
    grid = np.einsum(R, [0,1], overlapbox, [2,3,1], [2,3,0]) + t
    intersection = np.sum(np.all(abs(grid) < 1, axis=2))
    ioa = float(intersection) / overlapres**2
    iou = ioa / (1 - ioa + lb*wb/la/wa)
    return .7 - iou
    

def soMetricEuc(boxa, boxb):
    closeness = np.hypot(*(boxa[:2]-boxb[:2])) * .4
    closeness += ((boxa[2]-boxb[2]+np.pi)%(2*np.pi)-np.pi) * 1.
    closeness += np.hypot(*(boxa[3:]-boxb[3:5])) * .3
    return closeness - 1

class MetricAvgPrec():
    """
    builds histogram of scores for true and false samples
    calculates average precision based on histogram (slight approximation)
    """
    def __init__(self, resolution=391, soMetric=soMetricIoU):
        self.cutoffs = np.append(np.linspace(-29, 10, resolution-1), [1000.])
        self.counts = np.zeros((resolution, 5), dtype=int)
        self.nmissed = np.zeros(4, dtype=int)
        self.soMetric = soMetric
        
        self.faketru = np.zeros(resolution*2 + 1, dtype=bool)
        self.faketru[:resolution] = True
        self.faketru[-1] = True
        self.fakeest = np.concatenate((np.arange(resolution),
                                       np.arange(resolution), [-1.]))
    
    def add(self, gt, gtscored, gtdifficulty, ests, scores):
        ngt = gt.shape[0]
        assert gtscored.shape[0] == ngt
        assert gtdifficulty.shape[0] == ngt
        nests = ests.shape[0]
        assert scores.shape[0] == nests
        matches = np.zeros((ngt, nests))
        for gtidx, estidx in np.ndindex(ngt, nests):
            score = self.soMetric(gt[gtidx], ests[estidx])
            matches[gtidx, estidx] = min(score, 0)
        matchesnonmiss = matches < 0
        #matches[:] = 0
        #matches[matchesnonmiss] = -1
        #matches *= scores
        rowpairs, colpairs = linear_sum_assignment(matches)
        gtmisses = np.ones(ngt, dtype=bool)
        estmisses = np.ones(nests, dtype=bool)
        for gtidx, estidx in zip(rowpairs, colpairs):
            if matchesnonmiss[gtidx, estidx]:
                gtmisses[gtidx] = False
                estmisses[estidx] = False
                gtdiffidx = gtdifficulty[gtidx] if gtscored[gtidx] else 3
                scoreidx = np.searchsorted(self.cutoffs, scores[estidx])
                self.counts[scoreidx, gtdiffidx:4] += 1
        for gtidx in range(ngt):
            if gtmisses[gtidx]:
                gtdiffidx = gtdifficulty[gtidx] if gtscored[gtidx] else 3
                self.nmissed[gtdiffidx:] += 1
        for estidx in range(nests):
            if estmisses[estidx]:
                scoreidx = np.searchsorted(self.cutoffs, scores[estidx])
                self.counts[scoreidx, 4] += 1
        
    def calc(self):
        avgprec = np.zeros(4)
        for difficulty in range(4):
            fakeweights = np.concatenate((self.counts[:,difficulty],
                                          self.counts[:,4],
                                          [self.nmissed[difficulty]]))
            fakeweights = np.maximum(fakeweights, 1e-8)
            avgprec[difficulty] = average_precision_score(self.faketru,
                                   self.fakeest, sample_weight = fakeweights)
        return avgprec
    
    def calcKitti(self):
        avgprecs = np.zeros(4)
        for difficulty in range(4):
            totalhitcount = np.sum(self.counts[:,difficulty])
            RR = np.cumsum(self.counts[:,difficulty])
            RR = totalhitcount - RR + self.counts[:,difficulty]
            totalcount = totalhitcount + self.nmissed[difficulty]
            recallsteps = np.linspace(0, totalcount, 41)[::4]
            recallsteps = recallsteps[recallsteps <= totalhitcount]
            steps = RR.shape[0]-1-np.searchsorted(RR[::-1], recallsteps)
            area = 0.
            for cutoffcount in steps:
                tp = float(RR[cutoffcount])
                area += tp / (tp + np.sum(self.counts[cutoffcount:,4]) + 1e-8)
            avgprecs[difficulty] = area / 11.
        return avgprecs
                
    

class MetricPrecRec():
    def __init__(self, cutoff = .5, soMetric = soMetricEuc):
        self.cutoff = cutoff
        self.tp = np.zeros(4, dtype=int)
        self.t = np.zeros(4, dtype=int)
        self.p = 0
        self.soMetric = soMetric
        
    def add(self, gt, gtscored, gtdiff, ests, scores):
        for gtidx in range(gt.shape[0]):
            gtbox = gt[gtidx]
            difficultyidx = gtdiff[gtidx] if gtscored[gtidx] else 3
            if gtscored: self.t += 1
            matches = False
            for estidx, est in enumerate(ests):
                if self.soMetric(gtbox, est):
                    assert not matches
                    self.tp[difficultyidx:] += 1
                    matches = True
        self.p += ests.shape[0]
        
    def calc(self):
        tp = self.tp.astype(float)
        return np.column_stack(tp/self.t, tp/self.p)

#img = imread(img_files.format(file_idx))[:,:,::-1]
#
#gtboxes = []
#with open(gt_files.format(file_idx), 'r') as fd: gtstr = fd.read()
#gtstr = gtstr.split('\n')
#if gtstr[-1] == '': gtstr = gtstr[:-1]
#for gtrow in gtstr:
#    gtrow = gtrow.split(' ')
#    if gtrow[0] == 'DontCare' or gtrow[0]=='Misc': continue
#    gtboxes.append((float(gtrow[13]), -float(gtrow[11]),
#        1.5708-float(gtrow[14]), float(gtrow[10])/2, float(gtrow[9])/2))
#        
#estboxes = []
#with open(output_files.format(file_idx), 'r') as fd: gtstr = fd.read()
#gtstr = gtstr.split('\n')
#if gtstr[-1] == '': gtstr = gtstr[:-1]
#for gtrow in gtstr:
#    gtrow = gtrow.split(' ')
#    if gtrow[0] == 'DontCare' or gtrow[0]=='Misc': continue
#    estboxes.append((float(gtrow[13]), -float(gtrow[11]),
#        1.5708-float(gtrow[14]), float(gtrow[10])/2, float(gtrow[9])/2))
#    
#ngt = len(gtboxes)
#nest = len(estboxes)
#match_mtx = np.zeros((ngt, nest))
#for gtidx, gtbox in enumerate(gtboxes):
#    gtbox_uv = xy2uv(gtbox)
#    area_gt = gtbox[3]*gtbox[4]*4
#    for estidx, estbox in enumerate(estboxes):
#        estbox_uv = xy2uv(estbox)
#        area_est = estbox[3]*estbox[4]*4
#        iou = overlap(gtbox_uv, estbox_uv)
#        iou /= (area_gt + area_est - iou)
#        match_mtx[gtidx, estidx] = 1.-iou
#ff = assignment(match_mtx, .3)




def MAPfromfile(filename):
    with open(filename, 'r') as fd:
        all_results = fd.read()
    results = [[float(score) for score in result.split(' ') if score != '']
                    for result in all_results.split('\n')]
    print([sum(result[::4])/11. for result in results])
#MAPfromfile('../object/estimates/b/stats_car_detection_ground.txt')