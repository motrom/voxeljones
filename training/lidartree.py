# -*- coding: utf-8 -*-
"""
last mod 5/14/19
"""
import numpy as np
import numba as nb
from time import time

@nb.njit(nb.i8(nb.b1[:,:,:,:], nb.i8[:], nb.i8, nb.i8, nb.i8[:]))
def orderIdxsBySplit(X, idxs, start, end, split):
    """
    partition array[start:end] so that split_directions==True are on left
    """
    j = start
    for i in xrange(start, end):
        xi = idxs[i]
        if np.any(X[xi,split[0]:split[3],split[1]:split[4],split[2]:split[5]]):
            idxs[i] = idxs[j]
            idxs[j] = xi
            j += 1
    return j

@nb.njit(nb.f8(nb.f8,nb.f8,nb.f8,nb.f8))
def calcScore(gradin,hessin,gradsum,hesssum):
    gradout = gradsum - gradin
    hessout = hesssum - hessin
    return gradin*gradin/max(hessin, 1e-10) + gradout*gradout/max(hessout, 1e-10)

#def makeIntegral(X2):
#    X,Y,Z,N = X2.shape
#    for x,y,n in np.ndindex(X,Y,N):
#        count = 0
#        for z in range(Z):
#            count += X2[x,y,z,n]
#            X2[x,y,z,n] = count
#    for x,z,n in np.ndindex(X,Z,N):
#        count = 0
#        for y in range(Y):
#            count += X2[x,y,z,n]
#            X2[x,y,z,n] = count
#    for y,z,n in np.ndindex(Y,Z,N):
#        count = 0
#        for x in range(X):
#            count += X2[x,y,z,n]
#            X2[x,y,z,n] = count

#@nb.njit(nb.void(nb.i2[:,:,:,:], nb.i2[:], nb.f8[:], nb.f8[:], nb.i8[:],
#                 nb.f8[:], nb.i8[:,:]))
#def findSplitIntegral(X, storage, grad, hess, leafidxs, scores, splits):
#    xfull, yfull, zfull, n = X.shape
#    gradsum = np.sum(grad[leafidxs])
#    hesssum = np.sum(hess[leafidxs])
#    K = scores.shape[0]
#    worstbestscore = scores[0]
#    for xmin, ymin, zmin in np.ndindex(xfull, yfull, zfull):
#        xmaxcount = xfull - xmin
#        ymaxcount = yfull - ymin
#        zmaxcount = zfull - zmin
#        for xmax, ymax, zmax in np.ndindex(xmaxcount, ymaxcount, zmaxcount):
#            xmax += xmin+1
#            ymax += ymin+1
#            zmax += zmin+1
#            gradin = 0.
#            hessin = 0.
#            storage[:] = (X[xmax,ymax,zmax] - X[xmax,ymax,zmin] -
#                          X[xmax,ymin,zmax] - X[xmin,ymax,zmax] +
#                          X[xmax,ymin,zmin] + X[xmin,ymax,zmin] +
#                          X[xmin,ymin,zmax] - X[xmin,ymin,zmin])
#            gradin = np.sum(grad[storage>0])
#            hessin = np.sum(hess[storage>0])
#            if hessin > 0 and hessin < hesssum:
#                gradout = gradsum - gradin
#                hessout = hesssum - hessin
#                score = gradin*gradin/hessin + gradout*gradout/hessout
#                if score > worstbestscore:
#                    lastk = 0
#                    for k in range(1,K):
#                        if score < scores[k]:
#                            break
#                        scores[lastk] = scores[k]
#                        splits[lastk] = splits[k]
#                        lastk = k
#                    scores[lastk] = score
#                    splits[lastk] = (xmin,ymin,zmin,xmax,ymax,zmax)
#                    worstbestscore = scores[0]
    

@nb.njit(nb.void(nb.b1[:,:,:,:], nb.b1[:,:,:], nb.f8[:], nb.f8[:],
                  nb.i8[:], nb.f8[:], nb.i8[:,:]))
def findSplitInitial(X, storage, grad, hess, leafidxs, scores, splits):
    """
    X = [LxWxHxn] binary feature grid
    grad, hess = [n] float target values for GBM
    tries all possible integer 3d boxes in LxWxH
    """
    xfull, yfull, zfull, n = X.shape
    
    # variables to perform efficient calculations of the GB split cost
    gradsum = np.sum(grad[leafidxs])
    hesssum = np.sum(hess[leafidxs])
#    worstbestscore = gradsum*gradsum/hesssum
    K = scores.shape[0]
    worstbestscore = scores[0]
    
    for xmin, ymin, zmin in np.ndindex(xfull, yfull, zfull):
        xmaxcount = xfull - xmin
        ymaxcount = yfull - ymin
        zmaxcount = zfull - zmin
        
        storage[ymin:yfull+1,zmin:zfull+1, leafidxs] = False
        
        for xmax, ymax, zmax in np.ndindex(xmaxcount, ymaxcount, zmaxcount):
            xmax += xmin+1
            ymax += ymin+1
            zmax += zmin+1
            gradin = 0.
            hessin = 0.
            for xidx in leafidxs:
                here = (storage[ymax,zmax,xidx] |
                        storage[ymax-1,zmax,xidx] |
                        storage[ymax,zmax-1,xidx] |
                        X[xmax-1,ymax-1,zmax-1,xidx])
                # current storage[ymax,zmax] has info on :xmax-1,:ymax,:zmax
                # will never use this again, can replace with :xmax,:ymax,:zmax
                storage[ymax,zmax,xidx] = here
                if here:
                    gradin += grad[xidx]
                    hessin += hess[xidx]

            score = calcScore(gradin, hessin, gradsum, hesssum)
            if score > worstbestscore:
                # heapless O(k) update of k-best scores
                lastk = 0
                for k in xrange(1,K):
                    if score < scores[k]:
                        break
                    scores[lastk] = scores[k]
                    splits[lastk] = splits[k]
                    lastk = k
                scores[lastk] = score
                splits[lastk] = (xmin,ymin,zmin,xmax,ymax,zmax)
                worstbestscore = scores[0]


@nb.njit(nb.void(nb.b1[:,:,:,:], nb.f8[:], nb.f8[:],
                  nb.i8[:], nb.i8[:,:], nb.f8[:], nb.i8[:,:]))
def findSplitPartial2(X, grad, hess, leafidxs, boxes, scores, splits):
    """
    X = [nxLxWxH] binary feature grid
    grad, hess = [n] float target values for GBM
    tries various intervals
    """
    n, xfull, yfull, zfull = X.shape
    maxxmin, maxymin, maxzmin, minxmax, minymax, minzmax = boxes[0]
    minxmin, minymin, minzmin, maxxmax, maxymax, maxzmax = boxes[1]
    
    leaf_in = 0
    prevgradin = 0.
    prevhessin = 0.
    for leafidx in xrange(leafidxs.shape[0]):
        xidx = leafidxs[leafidx]
        if np.any(X[xidx, maxxmin:minxmax, maxymin:minymax, maxzmin:minzmax]):
            leafidxs[leafidx] = leafidxs[leaf_in]
            leafidxs[leaf_in] = xidx
            leaf_in += 1
            prevgradin += grad[xidx]
            prevhessin += hess[xidx]
    leaf_out = leafidxs.shape[0]
    for leafidx in xrange(leafidxs.shape[0]-1, leaf_in-1, -1):
        xidx = leafidxs[leafidx]
        if not np.any(X[xidx, minxmin:maxxmax, minymin:maxymax, minzmin:maxzmax]):
            leaf_out -= 1
            leafidxs[leafidx] = leafidxs[leaf_out]
            leafidxs[leaf_out] = xidx
    
    gradsum = np.sum(grad[leafidxs])
    hesssum = np.sum(hess[leafidxs])
    K = scores.shape[0]
    worstbestscore = scores[0]
    thissplit = np.zeros(6, dtype=np.int64)
    
    xmin, ymin, zmin, xmax, ymax, zmax = boxes[0]
    bestchangescore = calcScore(prevgradin, prevhessin, gradsum, hesssum)
    for addition in xrange(25):
        bestchange = 0
        if xmin > minxmin:
            gradin = prevgradin
            hessin = prevhessin
            for xidx in leafidxs[leaf_in:leaf_out]:
                if np.any(X[xidx, xmin-1, ymin:ymax, zmin:zmax]):
                    gradin += grad[xidx]
                    hessin += hess[xidx]
            changescore = calcScore(gradin, hessin, gradsum, hesssum)
            if changescore > bestchangescore:
                bestchange = 1
                bestchangescore = changescore
        if ymin > minymin:
            gradin = prevgradin
            hessin = prevhessin
            for xidx in leafidxs[leaf_in:leaf_out]:
                if np.any(X[xidx, xmin:xmax, ymin-1, zmin:zmax]):
                    gradin += grad[xidx]
                    hessin += hess[xidx]
            changescore = calcScore(gradin, hessin, gradsum, hesssum)
            if changescore > bestchangescore:
                bestchange = 2
                bestchangescore = changescore
        if zmin > minzmin:
            gradin = prevgradin
            hessin = prevhessin
            for xidx in leafidxs[leaf_in:leaf_out]:
                if np.any(X[xidx, xmin:xmax, ymin:ymax, zmin-1]):
                    gradin += grad[xidx]
                    hessin += hess[xidx]
            changescore = calcScore(gradin, hessin, gradsum, hesssum)
            if changescore > bestchangescore:
                bestchange = 3
                bestchangescore = changescore
        if xmax < maxxmax:
            gradin = prevgradin
            hessin = prevhessin
            for xidx in leafidxs[leaf_in:leaf_out]:
                if np.any(X[xidx, xmax, ymin:ymax, zmin:zmax]):
                    gradin += grad[xidx]
                    hessin += hess[xidx]
            changescore = calcScore(gradin, hessin, gradsum, hesssum)
            if changescore > bestchangescore:
                bestchange = 4
                bestchangescore = changescore
        if ymax < maxymax:
            gradin = prevgradin
            hessin = prevhessin
            for xidx in leafidxs[leaf_in:leaf_out]:
                if np.any(X[xidx, xmin:xmax, ymax, zmin:zmax]):
                    gradin += grad[xidx]
                    hessin += hess[xidx]
            changescore = calcScore(gradin, hessin, gradsum, hesssum)
            if changescore > bestchangescore:
                bestchange = 5
                bestchangescore = changescore
        if zmax < maxzmax:
            gradin = prevgradin
            hessin = prevhessin
            for xidx in leafidxs[leaf_in:leaf_out]:
                if np.any(X[xidx, xmin:xmax, ymin:ymax, zmax]):
                    gradin += grad[xidx]
                    hessin += hess[xidx]
            changescore = calcScore(gradin, hessin, gradsum, hesssum)
            if changescore > bestchangescore:
                bestchange = 6
                bestchangescore = changescore
        leaf_in2 = leaf_in
        if bestchange == 0:
            break
        elif bestchange == 1:
            xmin -= 1
            for leafidx in xrange(leaf_in, leaf_out):
                xidx = leafidxs[leafidx]
                if np.any(X[xidx, xmin, ymin:ymax, zmin:zmax]):
                    leafidxs[leafidx] = leafidxs[leaf_in2]
                    leafidxs[leaf_in2] = xidx
                    leaf_in2 += 1
                    prevgradin += grad[xidx]
                    prevhessin += hess[xidx]
        elif bestchange == 2:
            ymin -= 1
            for leafidx in xrange(leaf_in, leaf_out):
                xidx = leafidxs[leafidx]
                if np.any(X[xidx, xmin:xmax, ymin, zmin:zmax]):
                    leafidxs[leafidx] = leafidxs[leaf_in2]
                    leafidxs[leaf_in2] = xidx
                    leaf_in2 += 1
                    prevgradin += grad[xidx]
                    prevhessin += hess[xidx]
        elif bestchange == 3:
            zmin -= 1
            for leafidx in xrange(leaf_in, leaf_out):
                xidx = leafidxs[leafidx]
                if np.any(X[xidx, xmin:xmax, ymin:ymax, zmin]):
                    leafidxs[leafidx] = leafidxs[leaf_in2]
                    leafidxs[leaf_in2] = xidx
                    leaf_in2 += 1
                    prevgradin += grad[xidx]
                    prevhessin += hess[xidx]
        elif bestchange == 4:
            for leafidx in xrange(leaf_in, leaf_out):
                xidx = leafidxs[leafidx]
                if np.any(X[xidx, xmax, ymin:ymax, zmin:zmax]):
                    leafidxs[leafidx] = leafidxs[leaf_in2]
                    leafidxs[leaf_in2] = xidx
                    leaf_in2 += 1
                    prevgradin += grad[xidx]
                    prevhessin += hess[xidx]
            xmax += 1
        elif bestchange == 5:
            for leafidx in xrange(leaf_in, leaf_out):
                xidx = leafidxs[leafidx]
                if np.any(X[xidx, xmin:xmax, ymax, zmin:zmax]):
                    leafidxs[leafidx] = leafidxs[leaf_in2]
                    leafidxs[leaf_in2] = xidx
                    leaf_in2 += 1
                    prevgradin += grad[xidx]
                    prevhessin += hess[xidx]
            ymax += 1
        elif bestchange == 6:
            for leafidx in xrange(leaf_in, leaf_out):
                xidx = leafidxs[leafidx]
                if np.any(X[xidx, xmin:xmax, ymin:ymax, zmax]):
                    leafidxs[leafidx] = leafidxs[leaf_in2]
                    leafidxs[leaf_in2] = xidx
                    leaf_in2 += 1
                    prevgradin += grad[xidx]
                    prevhessin += hess[xidx]
            zmax += 1
        leaf_in = leaf_in2
        
#        #ff = np.any(X[:, xmin:xmax, ymin:ymax, zmin:zmax].reshape((n,-1)),axis=1)
#        for xidx in leafidxs[:leaf_in]:
#            assert np.any(X[xidx, xmin:xmax, ymin:ymax, zmin:zmax])
#        for xidx in leafidxs[leaf_in:]:
#            assert not np.any(X[xidx, xmin:xmax, ymin:ymax, zmin:zmax])
#        gradin2 = np.sum(grad[leafidxs[:leaf_in]])
#        hessin2 = np.sum(hess[leafidxs[:leaf_in]])
#        assert abs(gradin2 - prevgradin) < 1e-4
#        assert abs(hessin2 - prevhessin) < 1e-4
        
        # whether or not this is indented decides how often scores are included!
        score = bestchangescore
        if score > worstbestscore:                
            # heapless O(k) update of k-best scores
            # checking for duplicates
            lastk = 0
            for k in xrange(1,K):
                if score > scores[k]:
                    lastk += 1
            thissplit[:] = (xmin, ymin, zmin, xmax, ymax, zmax)
            if np.all(splits[lastk] == thissplit):
                pass
            elif np.all(splits[lastk+1] == thissplit):
                pass
            else:
                for k in xrange(lastk):
                    scores[k] = scores[k+1]
                    splits[k] = splits[k+1]
                scores[lastk] = score
                splits[lastk] = thissplit
                worstbestscore = scores[0]


rough = 3
Krough = 20
#def prepRough(X):
#    n, xc, yc, zc = X.shape
#    xrough = xc//rough
#    yrough = yc//rough
#    zrough = zc//rough
#    
#    roughX = np.zeros((n, xrough, yrough, zrough), dtype=bool)
#    storage = np.zeros((yrough, zrough, n), dtype=bool)
#    for xoff, yoff, zoff in np.ndindex(rough, rough, rough):
#        roughX |= X[:, xoff:xrough*rough:rough,
#                       yoff:yrough*rough:rough, zoff:zrough*rough:rough]
#    return roughX, storage
@nb.njit(nb.b1[:,:,:,:](nb.b1[:,:,:,:]))
def prepRough2(X):
    n, xc, yc, zc = X.shape
    xrough = xc//rough
    yrough = yc//rough
    zrough = zc//rough
    roughX = np.zeros((xrough, yrough, zrough, n), dtype=np.bool8)
    #storage = np.zeros((yrough, zrough, n), dtype=bool)
    for xidx,x,y,z in np.ndindex(n, xrough*rough, yrough*rough, zrough*rough):
        roughX[x//rough,y//rough,z//rough,xidx] |= X[xidx,x,y,z]
    return roughX


def findSplit(X, roughX, roughstorage, leafidxs, leaf_start, leaf_end, grad, hess, K):
    #starttime = time()
    n, xc, yc, zc = X.shape
    leafidxs = leafidxs[leaf_start:leaf_end].copy()
    
    scores = np.zeros(Krough)
    splits = np.zeros((Krough, 6), dtype=int)
    findSplitInitial(roughX, roughstorage, grad, hess, leafidxs, scores, splits)
    roughsplits = splits
    #roughscores = scores
    
    scores = np.zeros(K)
    splits = np.zeros((K, 6), dtype=int)
    for split in roughsplits:
        xmin, ymin, zmin, xmax, ymax, zmax = split
        minxmin = max(xmin*rough - rough + 1, 0)
        minymin = max(ymin*rough - rough + 1, 0)
        minzmin = max(zmin*rough - rough + 1, 0)
        maxxmax = min(xmax*rough + rough - 1, xc)
        maxymax = min(ymax*rough + rough - 1, yc)
        maxzmax = min(zmax*rough + rough - 1, zc)
        maxxmin = xmin*rough + rough - 1
        minxmax = min(xmax*rough - rough + 1, xc)
        if minxmax < maxxmin:
            minxmax = (minxmax + maxxmin)//2
            maxxmin = minxmax
        maxymin = ymin*rough + rough - 1
        minymax = min(ymax*rough - rough + 1, yc)
        if minymax < maxymin:
            minymax = (minymax + maxymin)//2
            maxymin = minymax
        maxzmin = zmin*rough + rough - 1
        minzmax = min(zmax*rough - rough + 1, zc)
        if minzmax < maxzmin:
            minzmax = (minzmax + maxzmin)//2
            maxzmin = minzmax
        boxes = np.array(((maxxmin, maxymin, maxzmin, minxmax, minymax, minzmax),
                          (minxmin, minymin, minzmin, maxxmax, maxymax, maxzmax)))
        
        findSplitPartial2(X, grad, hess, leafidxs, boxes, scores, splits)
    #endtime = time()
    #print(endtime-starttime)
    return splits, scores
    
#    leafin = orderIdxsBySplit(X, leafidxs, leaf_start, leaf_end, splits[-1])
#    allall = leaf_end-leaf_start
#    posall = sum(y)
#    posin = sum(y[leafidxs[:leafin]])
#    print(np.array([[posin, leafin-posin],[posall-posin, allall-leafin-posall+posin]]))
    
    
    
def trainTree(X, grad, hess, nhypotheses, depth):
    n = X.shape[0]
    # stores indices of each leaf's start and end
    leafidxs = np.arange(n)
    leaf_start = 0
    leaf_end = n
    nleaves = 2 ** depth
    splits = np.zeros((nleaves, 6), dtype=int)
    leaves = np.zeros((nleaves*2-1, 2), dtype=int) # start idx, end idx
    leaves[0] = (0, n) # initial leaf includes everything
    
    #roughX, roughstorage = prepRough(X)
    roughX = prepRough2(X)
    roughstorage = np.zeros((roughX.shape[1]+1,roughX.shape[2]+1,roughX.shape[3]),
                            dtype=np.bool8) # 6/7/19
    #roughstorage = roughX[0].copy()
    for parentidx in range(nleaves-1):
        leaf_start, leaf_end = leaves[parentidx]
        if leaf_end - leaf_start < 5:
            split = np.zeros(6, dtype=int)
        else:
            trialsplits, trialscores = findSplit(X, roughX, roughstorage,
                                            leafidxs, leaf_start, leaf_end,
                                            grad, hess, nhypotheses)
            bestsplit = trialsplits[-1]
            bestscore = trialscores[-1]
            if parentidx*2+2 < nleaves:
                # there are splits below this one
                for split in trialsplits:
                    new_leaf_idx = orderIdxsBySplit(X, leafidxs, leaf_start, leaf_end,
                                                    split)
                    trialsplits2, trialscores2 = findSplit(X, roughX, roughstorage,
                                                      leafidxs, leaf_start, new_leaf_idx,
                                                      grad, hess, 1)
                    score = trialscores2[-1]
                    trialsplits2, trialscores2 = findSplit(X, roughX, roughstorage,
                                                      leafidxs, new_leaf_idx, leaf_end,
                                                      grad, hess, 1)
                    score += trialscores2[-1]
                    if score > bestscore:
                        bestscore = score
                        bestsplit = split
                print("best trial {:d}".format(
                        np.where(np.all(trialsplits==bestsplit,axis=1))[0][0]))
            split = bestsplit
        splits[parentidx] = split
        new_leaf_idx = orderIdxsBySplit(X, leafidxs, leaf_start, leaf_end, split)
        leaves[parentidx*2+1] = (leaf_start, new_leaf_idx)
        leaves[parentidx*2+2] = (new_leaf_idx, leaf_end)
        
    leaves = leaves[nleaves-1:]
    leafvals = np.zeros(nleaves)
    for leafidx in range(nleaves):
        leaf_start, leaf_end = leaves[leafidx]
        if leaf_end == leaf_start:
            # set value to value of split one higher up
            if leafidx % 2:
                leaf_start = leaves[leafidx - 1, 0]
            else:
                leaf_end = leaves[leafidx + 1, 1]
        if leaf_start != leaf_end:
            val = -(sum(grad[leafidxs[leaf_start:leaf_end]]) /
                    sum(hess[leafidxs[leaf_start:leaf_end]]))
        else:
            val = 0.
        leafvals[leafidx] = val
    leafvals = np.array(leafvals)
    return splits, leafvals
    #np.save('../dataApril19/splits56.npy', splits)


@nb.njit(nb.f8(nb.b1[:,:,:], nb.i8[:,:], nb.f8[:]))
def useTree(sample, treesplits, treeleaves):
    """
    treesplits = [nleaves 6] int array, index nleaves-1 is unused
    treeleaves = [nleaves] float array
    """
    splitidx = 0
    nsplits = len(treesplits)-1
    while splitidx < nsplits:
        split = treesplits[splitidx]
        splitidx = splitidx*2+1
        if not np.any(sample[split[0]:split[3], split[1]:split[4], split[2]:split[5]]):
            splitidx += 1
    return treeleaves[splitidx - nsplits]

@nb.njit(nb.f8(nb.b1[:,:,:], nb.i8[:,:,:], nb.f8[:,:]))
def useBoostedTree(sample, treessplits, treesleaves):
    """
    assumes all trees have same shape
    treessplits = [ntrees nleaves 6] int array
    treesleaves = [ntrees  nleaves] float array
    """
#    return sum(useTree(sample, treessplits[tree], treesleaves[tree])
#                for tree in xrange(treessplits.shape[0]))
    score = 0.
    nleaves = treessplits.shape[1]
    for tree in xrange(treessplits.shape[0]):
        score += useTree(sample, treessplits[tree], treesleaves[tree])
        if score < treesleaves[tree,nleaves]:
            return -50. # very low
    return score



def analyzeTree(y, leaves, predictions):
    leafcounts = np.zeros((nleaves,2), dtype=int)
    for jj, leaf in enumerate(leaves):
        leaf_start, leaf_end = leaf
        leafcounts[jj,0] = sum(y[leafidxs[leaf_start:leaf_end]])
        leafcounts[jj,1] = leaf_end-leaf_start - leafcounts[jj,0]
        
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    fpr, tpr, thresholds = roc_curve(y, predictions)
    fprr = np.linspace(0., 1., 50)
    tprr = tpr[np.searchsorted(fpr, fprr, side='right')-1]
#    tprs.append(interp(mean_fpr, fpr, tpr))
#    tprs[-1][0] = 0.0
    plt.plot(fprr, tprr, lw=1, alpha=0.3)
        
#    newp = p/(p+(1-p)*np.exp(predictions*-1))
#    crossent = np.sum(np.log(newp*(2*y-1) + (1-y)))
