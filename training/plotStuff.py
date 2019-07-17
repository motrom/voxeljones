#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 2/1/19 adding some common functions
+x is up, +y is left
reference is bottom center, unlike prototype which is center center
"""

from math import pi, tan, cos, sin
import numpy as np
import cv2
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


size = 320

def reference(x, y):
    return size*2-x*size, size-y*size

twopi = 2*pi
## assumes ccw(0, angle1, angle2)
def fillArc(img, d1, d2, angle1, angle2, color):
    angle_diff = (angle2 - angle1) % twopi
    mid_angle = (angle1 + angle_diff/2 + pi) % twopi - pi
    if angle_diff >= pi/2:
        fillArc(img, d1, d2, angle1, mid_angle, color)
        fillArc(img, d1, d2, mid_angle, angle2, color)
    
    if abs(mid_angle) <= pi/4:
        in_x = True
        start = -max(d1*cos(angle1), d2*cos(angle2))
        slope1 = -tan(angle2)
        slope2 = -tan(angle1)
    elif mid_angle >= 3*pi/4 or mid_angle <= -3*pi/4:
        in_x = True
        start = -min(d1*cos(angle1), d2*cos(angle2))
        slope1 = -tan(angle1)
        slope2 = -tan(angle2)
    elif mid_angle > 0:
        in_x = False
        start = -max(d1*sin(angle1), d2*sin(angle2))
        slope1 = -1./tan(angle1)
        slope2 = -1./tan(angle2)
    else:
        in_x = False
        start = -min(d1*sin(angle1), d2*sin(angle2))
        slope1 = -1./tan(angle2)
        slope2 = -1./tan(angle1)
    if abs(start) > size: return
    
    if start > 0:
        if in_x:
            return # this is in the back of the vehicle
        else:
            vrange = np.arange(start+size, size*2, dtype=int)
            xx = np.empty((vrange.shape[0], 3), dtype=int)
            xx[:,0] = vrange
            xx[:,1] = (size-xx[:,0])*slope1 + size*2 + 1
            xx[:,2] = (size-xx[:,0])*slope2 + size*2 + 1
    else:
        if in_x:
            vrange = np.arange(start+size*2, dtype=int)
            xx = np.empty((vrange.shape[0], 3), dtype=int)
            xx[:,0] = vrange
            xx[:,1] = (size*2-xx[:,0])*slope1 + size + 1
            xx[:,2] = (size*2-xx[:,0])*slope2 + size + 1
        else:
            vrange = np.arange(start+size, dtype=int)
            xx = np.empty((vrange.shape[0], 3), dtype=int)
            xx[:,0] = vrange
            xx[:,1] = (size-xx[:,0])*slope1 + size*2 + 1
            xx[:,2] = (size-xx[:,0])*slope2 + size*2 + 1
    xx[:,1] = np.minimum(np.maximum(xx[:,1], 0), size*2)
    xx[:,2] = np.minimum(np.maximum(xx[:,2], 0), size*2)
    if in_x:
        for x in xx:
            img[x[0], x[1]:x[2]] = color
    else:
        for x in xx:
            img[x[1]:x[2], x[0]] = color
        
### from wikipedia
def drawLine(x0,y0,x1,y1):
    outx = []
    outy = []
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0,y0 = y0,x0
        x1,y1 = y1,x1
    if x0 > x1:
        x0,x1 = x1,x0
        y0,y1 = y1,y0
    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / float(dx) if dx != 0 else 1.
    xend = int(x0+.5)
    yend = y0 + gradient * (xend - x0)
    xpxl1 = xend
    ypxl1 = int(yend)
    outx.append(xpxl1)
    outx.append(xpxl1)
    outy.append(ypxl1)
    outy.append(ypxl1+1)
    intery = yend + gradient
    xend = int(x1+.5)
    yend = y1 + gradient * (xend - x1)
    xpxl2 = xend
    ypxl2 = int(yend)
    outx.append(xpxl2)
    outx.append(xpxl2)
    outy.append(ypxl2)
    outy.append(ypxl2+1)
    for x in xrange(xpxl1 + 1, xpxl2):
        ffff = int(intery)
        outx.append(x)
        outx.append(x)
        outy.append(ffff)
        outy.append(ffff+1)
        intery += gradient
    if steep:
        return outy, outx
    else:
        return outx, outy

def plotCircle(img, x, y, radius, color):
    x_min = int(x-radius) - 1
    x_min = max(x_min, 0)
    x_max = int(x+radius) + 1
    x_max = min(x_max+1, size*2)
    y_min = int(y-radius) - 1
    y_min = max(y_min, 0)
    y_max = int(y+radius) + 1
    y_max = min(y_max+1, size*2)
    
    x_off = np.arange(x_min, x_max) - x
    y_off = np.arange(y_min, y_max) - y
    x_off, y_off = np.meshgrid(y_off, x_off)
    include = np.hypot(x_off, y_off) < radius + .5/size
    img[x_min:x_max, y_min:y_max][include] = color
    
    
def plotRectangle(img, rect, color):
    x,y,angle,l,w = rect
    #angle = pi/2-angle
    c = cos(angle)
    s = sin(angle)
    x_span = abs(c)*l + abs(s)*w
    y_span = abs(c)*w + abs(s)*l
    x_min = int(x-x_span) - 1
    x_min = max(x_min, 0)
    x_max = int(x+x_span) + 1
    x_max = min(x_max+1, size*2)
    y_min = int(y-y_span) - 1
    y_min = max(y_min, 0)
    y_max = int(y+y_span) + 1
    y_max = min(y_max+1, size*2)
    
    x_off = np.arange(x_min, x_max) - x
    y_off = np.arange(y_min, y_max) - y
    y_off, x_off = np.meshgrid(y_off, x_off)
    include = abs(c*x_off + s*y_off) < l+.5/size
    include &= abs(s*x_off - c*y_off) < w+.5/size
    img[x_min:x_max, y_min:y_max][include] = color

def plotPoints(img, x, y, pointshape, color):
    for off_x, off_y in pointshape:
        img[x+off_x, y+off_y] = color

def colorList(n_colors):
    cm_obj = ScalarMappable(norm = Normalize(0, n_colors), cmap='jet')
    return (cm_obj.to_rgba(range(n_colors))[:,:3] * 255.9).astype(np.uint8)

    
base_image = np.zeros((size*2, size*2, 3), dtype=np.uint8) + 255
plotCircle(base_image, size*2, size, 1./30.*320, (250,210,190))


class Display():
    def __init__(self): pass
    def __enter__(self):
        #cv2.imshow('lidar side detections', base_image)
        cv2.namedWindow('lidar side detections', flags=cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(5)
        return self
    def __exit__(self, a, b, c): cv2.destroyWindow('lidar side detections')
    def display(self, image):
        cv2.imshow('lidar side detections', image)
        cv2.waitKey(5)
        
        
        
class DisplayNULL():
    def __init__(self): pass
    def __enter__(self): return self
    def __exit__(self, a, b, c): pass
    def display(self, image): pass



## added feb 2019
def grayer(img):
    return ((img.astype(float)-128)*.75 + 128).astype(np.uint8)
