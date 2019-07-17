# -*- coding: utf-8 -*-
"""
last mod 6/11/19
"""
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from time import sleep # might help not kill computer, idk
#from sklearn.neighbors import KernelDensity

from config import floor, anchorstart, anchorstep, anchorlen
from config import present_boxes_file
from config import training_file_start, training_file_end
from config import positive_points_file, positive_points_index_file
from getNegatives2 import prepForPredicting, predictNegs
from lidartree import trainTree, useBoostedTree


BT_file = '../dataApril19/BT5{:02d}.npz'
error_plot_file = '../dataApril19/debugplot5{:02d}.png'
starting_tree = 24
ntrees = 28
nnegatives = 30000
noiseposition = np.array((.275, .275, .055))#(.33, .33, .055))
noiseangle = np.pi/16. * 1.1 #np.pi/16. * 1.1
boost_bias = 2. # was 4.
boost_regularization = .1 # was .05
linesearch_regularization = .1 # was .01
linesearch_regularization_scaleless = .1 # was .05
nfilesfornegatives_min = 40
nnegativesamplerate_min = 20. # was 30
max_percent_positives_to_cull = .5 # out of 100
nhypotheses_per_split_training = 10 # was 12


@nb.njit(nb.f8(nb.b1[:,:,:], nb.i8[:,:,:], nb.f8[:,:]))
def useBoostedTree2(grid, btsplits, btleaves):
    tscore = 0.
    for tree in range(10):
        splitidx = 0
        for depth in range(3):
            split = btsplits[tree, splitidx]
            splitidx = splitidx*2+2
            if np.any(grid[split[0]:split[3], split[1]:split[4], split[2]:split[5]]):
                splitidx -= 1
        tscore += btleaves[tree, splitidx - 7]
        if tscore < btleaves[tree, 8]:
            tscore = -50.
            break
    return tscore
@nb.njit(nb.f8(nb.b1[:,:,:], nb.i8[:,:,:], nb.f8[:,:], nb.f8))
def useBoostedTree3(grid, btsplits, btleaves, tscore):
    ntrees = btsplits.shape[0]
    for tree in range(10, ntrees):
        splitidx = 0
        for depth in range(3):
            split = btsplits[tree, splitidx]
            splitidx = splitidx*2+2
            if np.any(grid[split[0]:split[3], split[1]:split[4], split[2]:split[5]]):
                splitidx -= 1
        tscore += btleaves[tree, splitidx - 7]
        if tscore < btleaves[tree, 8]:
            tscore = -50.
            break
    return tscore

posidxs = np.load(positive_points_index_file)
npositivestotal = posidxs.shape[0]
objects_to_suppress = np.load(present_boxes_file)
if starting_tree == 0:
    btsplits = np.zeros((0,8,6), dtype=int)
    btleaves = np.zeros((0,9))
else:
    BTstruct = np.load(BT_file.format(starting_tree))
    btsplits = BTstruct['splits']
    btleaves = BTstruct['leaves']
    
# ~1e5 for 0
estimated_negatives_per_file = 200
positivesample = np.zeros(anchorlen, dtype=bool)

for tree in range(starting_tree+1, ntrees+1):
    print("learning tree {:d}".format(tree))
    
    # get negative samples
    nfilesfornegatives = nnegativesamplerate_min*nnegatives/estimated_negatives_per_file
    nfilesfornegatives = max(nfilesfornegatives_min, int(nfilesfornegatives))
    nfilesfornegatives = min(nfilesfornegatives, training_file_end-training_file_start)
    files2use = np.random.choice(np.arange(training_file_start, training_file_end),
                                    nfilesfornegatives, replace=False)
    negs = np.zeros((nnegatives, anchorlen[0], anchorlen[1], anchorlen[2]), dtype=bool)
    negscores = np.zeros(nnegatives)
    totalsamples = 0
    for fileidx in files2use:
        pts, tileidxs, pts2suppress, groundTs = prepForPredicting(fileidx,
                                                                objects_to_suppress)
        totalsamples = predictNegs(pts, tileidxs, groundTs, btsplits, btleaves,
                                     pts2suppress, negs, negscores, totalsamples)
        if fileidx%100 == 0:
            sleep(20)
    print("found {:d} negatives in {:d} files".format(totalsamples,
                                                      nfilesfornegatives))
    estimated_negatives_per_file = float(totalsamples) / nfilesfornegatives
    assert all(useBoostedTree3(sample, btsplits, btleaves, tscore) > -20 for
               sample,tscore in zip(negs,negscores))
    #np.save('negs.npy', negs)

    sleep(10)

    # get positive samples
    pospoints = np.load(positive_points_file)
    # randomly perturb positive samples
    poss = np.zeros((npositivestotal, anchorlen[0], anchorlen[1], anchorlen[2]),
                    dtype=bool)
    posscores = np.zeros(npositivestotal)
    noises = np.random.uniform(-noiseposition, noiseposition, size=(npositivestotal,3))
    noiseangles = np.random.uniform(-noiseangle, noiseangle, size=npositivestotal)
    npositives = 0
    for posidx in xrange(npositivestotal):
        pts = pospoints[posidxs[posidx,0]:posidxs[posidx,1]].copy()# + noises[posidx]
        if posidxs[posidx,2]:
            noisecos = np.cos(noiseangles[posidx])
            noisesin = np.sin(noiseangles[posidx])
            ptsx = pts[:,0]*noisecos - pts[:,1]*noisesin
            pts[:,1] = pts[:,0]*noisesin + pts[:,1]*noisecos
            pts[:,0] = ptsx
            pts += noises[posidx]
        pts = floor(pts/anchorstep) - anchorstart
        includepts = np.all(pts >= 0, axis=1) & np.all(pts < anchorlen, axis=1)
        pts = pts[includepts]
        positivesample[:] = False
        positivesample[pts[:,0], pts[:,1], pts[:,2]] = True
        tscore = useBoostedTree2(positivesample, btsplits, btleaves)
        if tscore < -30:
            continue
        if posidxs[posidx,2]:
            noiseanchor = (noises[posidx] + .0625) % .125 - .0625
            noisecos = np.cos((noiseangles[posidx]+np.pi/32)%(np.pi/16)-np.pi/32)
            noisesin = np.sin((noiseangles[posidx]+np.pi/32)%(np.pi/16)-np.pi/32)
            pts = pospoints[posidxs[posidx,0]:posidxs[posidx,1]].copy()
            ptsx = pts[:,0]*noisecos - pts[:,1]*noisesin
            pts[:,1] = pts[:,0]*noisesin + pts[:,1]*noisecos
            pts[:,0] = ptsx
            pts += noiseanchor
            pts = floor(pts/anchorstep) - anchorstart
            includepts = np.all(pts >= 0, axis=1) & np.all(pts < anchorlen, axis=1)
            pts = pts[includepts]
            positivesample[:] = False
            positivesample[pts[:,0], pts[:,1], pts[:,2]] = True
        if useBoostedTree3(positivesample, btsplits, btleaves, tscore) > -30:
            poss[npositives] = positivesample
            posscores[npositives] = tscore
            npositives += 1
    del pospoints
    print("kept {:d} out of {:d} positives".format(npositives, npositivestotal))

    # compile and score samples
    X = np.append(poss[:npositives], negs, axis=0)
    del poss, negs
    scoreold = np.append(posscores[:npositives], negscores, axis=0)
    del posscores, negscores
    score = np.array([useBoostedTree3(sample, btsplits, btleaves, tscore)
                        for sample,tscore in zip(X, scoreold)])
    assert all(score > -20), "{:d} {:d} {:d}".format(X.shape[0],npositives,np.argmin(score))

#    # display histograms of score for positive and negative samples
#    minscore = np.percentile(score, 0.1)
#    maxscore = np.percentile(score, 99.9)
#    scoreplot = np.linspace(minscore, maxscore, 100)
#    kde = KernelDensity(kernel='tophat', bandwidth=(maxscore-minscore)/50.)
#    kdepos = np.exp(kde.fit(score[:npositives,None]).score_samples(scoreplot[:,None]))
#    kdeneg = np.exp(kde.fit(score[npositives:,None]).score_samples(scoreplot[:,None]))
#    plt.plot(scoreplot, kdepos, 'g', scoreplot, kdeneg, 'r')
#    plt.title('distribution of log-odds estimates')
#    plt.legend(('positives', 'negatives'))

    # find an intercept
    ff = np.linspace(-2.5,2.5,1001)
    bestfval = (1e10, 0.)
    for intercept in ff:
        loss = sum(np.log(1+np.exp(-score[:npositives]-intercept)))*boost_bias
        loss += sum(np.log(1+np.exp(score[npositives:]+intercept)))
        bestfval = min(bestfval, (loss, intercept))
    intercept = bestfval[1]
    print("using intercept {:.3f}".format(intercept))
    

    # set up gradients
    prob = 1./(1+np.exp(-score-intercept))
    grad = prob.copy()
    grad[:npositives] -= 1.
    grad[:npositives] *= boost_bias
    hess = prob*(1-prob)
    hess[:npositives] *= boost_bias
    hess += boost_regularization
    
    # train next tree
    tsplits, tleaves = trainTree(X, grad, hess,
                         nhypotheses = nhypotheses_per_split_training, depth = 3)

    # check out split demographics
    tleaves2 = np.array([[0.,1,2,3,4,5,6,7,-30]])
    ss = np.array([useBoostedTree(sample, tsplits[None,:], tleaves2) for sample in X])
    count_pos = [np.sum(ss[:npositives]==leafidx) for leafidx in range(8)]
    count_neg = [np.sum(ss[npositives:]==leafidx) for leafidx in range(8)]

    # for each leaf, determine best value
    tleaves3 = np.zeros(9)
    tleaves3[8] = -50
    ff = np.linspace(-5.,5.,1001)
    for leafidx in range(8):
        scorepos = score[:npositives][ss[:npositives]==leafidx].copy()
        scoreneg = score[npositives:][ss[npositives:]==leafidx].copy()
        nss = scorepos.shape[0]+scoreneg.shape[0]
        f = tleaves[leafidx]
        loss = sum(np.log(1+np.exp(-scorepos-(f+intercept))))*boost_bias
        loss += sum(np.log(1+np.exp(scoreneg+(f+intercept))))
        loss += f*f*linesearch_regularization*nss
        loss += f*f*linesearch_regularization_scaleless
        bestleafval = (loss, f+intercept)
        for f in ff:
            loss = sum(np.log(1+np.exp(-scorepos-(f+intercept))))*boost_bias
            loss += sum(np.log(1+np.exp(scoreneg+(f+intercept))))
            loss += f*f*linesearch_regularization*nss
            loss += f*f*linesearch_regularization_scaleless
            bestleafval = min(bestleafval, (loss, f+intercept))
        tleaves3[leafidx] = bestleafval[1]
    print("positive count, negative count, boost val, linesearch val")
    print np.column_stack((count_pos, count_neg, tleaves, tleaves3[:8]))

    # add tree to boosted model
    btsplits = np.append(btsplits, tsplits[None,:,:], axis=0)
    btleaves = np.append(btleaves, tleaves3[None,:], axis=0)
    oldscore = score
    score = np.array([useBoostedTree3(sample, btsplits, btleaves, tscore)
                        for sample,tscore in zip(X, scoreold)])
    
    # determine cutoff for this tree
    scorecutoff = np.percentile(score[:npositives],
                             max_percent_positives_to_cull*npositivestotal/npositives)
    btleaves[-1,8] = scorecutoff
    newcullpositives = float(sum(score[:npositives]<scorecutoff))/npositivestotal
    newcullnegatives = float(sum(score[npositives:]<scorecutoff))/nnegatives
    estimated_negatives_per_file *= (1-newcullnegatives)
    print("estimated cull {:.3f} positives {:.3f} negatives".format(
                                newcullpositives, newcullnegatives))
    
    # plot distribution specifically for cutoff
    minscore = np.percentile(score, 0.1)
    maxcutscore = np.percentile(score[:npositives], 5.*npositivestotal/npositives,
                                interpolation='higher')
    maybecutpos = np.sort(score[:npositives][score[:npositives]<maxcutscore])
    maybecutneg = np.sort(score[npositives:][score[npositives:]<maxcutscore])
    scoreplot = np.linspace(minscore, maxcutscore, 100)
    cdfpos = np.searchsorted(maybecutpos, scoreplot) / .05 / npositivestotal
    cdfneg = np.searchsorted(maybecutneg, scoreplot) * 1. / nnegatives
    plt.plot(scoreplot, cdfpos, 'g', scoreplot, cdfneg, 'r')
    plt.ylim((0., 1.))
    plt.title('% of samples culled')
    plt.vlines(scorecutoff, 0., 1., 'b', linestyles='dashed')
    plt.legend(('positives * .05', 'negatives', 'cutoff'))
    plt.savefig(error_plot_file.format(tree))
    plt.clf()
    
    # save tree so far
    np.savez(BT_file.format(tree), splits=btsplits, leaves=btleaves)
    
    # pause
    sleep(10)