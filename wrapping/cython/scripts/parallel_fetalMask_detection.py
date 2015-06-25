#!/usr/bin/python

import sys
import numpy as np
import irtk

import os
from glob import glob

from lib.BundledSIFT import *
import scipy.ndimage as nd

import argparse

from datetime import datetime

parser = argparse.ArgumentParser(
    description='Parallel slice-by-slice detection of fetal brain MRI (3D).' )
parser.add_argument( "filename", type=str )
parser.add_argument( "ga", type=float )
parser.add_argument( "output_mask", type=str )
parser.add_argument( '--debug', action="store_true", default=False )
parser.add_argument( '--fold', type=str, default='0' )
parser.add_argument( '--classifier', type=str )
parser.add_argument( '--vocabulary', type=str )
parser.add_argument( '--new_sampling', type=float )
args = parser.parse_args()
print args

filename = args.filename
ga = args.ga
output_mask = args.output_mask

output_dir = os.path.dirname( output_mask )
if output_dir != '' and not os.path.exists(output_dir):
    os.makedirs(output_dir)

if output_dir == '':
    output_dir = '.'

raw_folder = "/home/handong/example-motion-correction/data"
mser_detector = args.classifier
vocabulary = args.vocabulary
NEW_SAMPLING = args.new_sampling

detections = []

img = irtk.imread(filename, dtype="float32").saturate().rescale()

print"Detect MSER regions"
tstart = datetime.now()
image_regions = parallel_detect_mser( filename,
                             ga,
                             vocabulary,
                             mser_detector,
                             NEW_SAMPLING,
                             DEBUG=args.debug,
                             output_folder=output_dir,
                             return_image_regions=True)
tend = datetime.now()
tdiff = tend-tstart
#print 'len: ', len(image_regions)
#print image_regions
print 'Time for detect_mser: ', tdiff.seconds, ' seconds'

# flatten list
# http://stackoverflow.com/questions/406121/flattening-a-shallow-list-in-python
import itertools
chain = itertools.chain(*image_regions)
image_regions = list(chain)

print "Fit cube using RANSAC"
def convert_input(image_regions):
    detections = []
    for ellipse_center, c in image_regions:
        x,y,z = map(float, ellipse_center)
        center = [x*NEW_SAMPLING,
                  y*NEW_SAMPLING,
                  z*img.header['pixelSize'][2]]
        region = np.hstack( (c, [[z]]*c.shape[0]) ).astype('float32')
        region[:,0] *= NEW_SAMPLING
        region[:,1] *= NEW_SAMPLING
        region[:,2] *= img.header['pixelSize'][2]
        detections.append((center, region))

    return detections

tstart = datetime.now()
detections = convert_input(image_regions)
tend = datetime.now()
tdiff = tend-tstart
print 'Time for convert_input: ', tdiff.seconds, ' seconds'

#print detections
tstart = datetime.now()
(center, u, ofd), inliers = ransac_ellipses( detections,
                                             ga,
                                             nb_iterations=1000,
                                             model="box",
                                             return_indices=True )
tend = datetime.now()
tdiff = tend-tstart
print 'Time for ransac_ellipses: ', tdiff.seconds, ' seconds'

print "initial mask"
tstart = datetime.now()
mask = irtk.zeros(img.resample2D(NEW_SAMPLING, interpolation='nearest').get_header(),
                  dtype='uint8')

for i in inliers:
    (x,y,z), c = image_regions[i]
    mask[z,c[:,1],c[:,0]] = 1

mask = mask.resample2D(img.header['pixelSize'][0], interpolation='nearest' )
tend = datetime.now()
tdiff = tend-tstart
print 'Time: ', tdiff.seconds, ' seconds'

print ' ellipse mask'
tstart = datetime.now()
ellipse_mask = irtk.zeros(img.resample2D(NEW_SAMPLING, interpolation='nearest').get_header(), dtype='uint8')
for i in inliers:
    (x,y,z), c = image_regions[i]
    ellipse = cv2.fitEllipse(np.reshape(c, (c.shape[0],1,2) ).astype('int32'))
    tmp_img = np.zeros( (ellipse_mask.shape[1],ellipse_mask.shape[2]), dtype='uint8' )
    cv2.ellipse( tmp_img, (ellipse[0],
                           (ellipse[1][0],ellipse[1][1]),
                           ellipse[2]) , 1, thickness=-1)
    ellipse_mask[z][tmp_img > 0] = 1

ellipse_mask = ellipse_mask.resample2D(img.header['pixelSize'][0], interpolation='nearest')
#irtk.imwrite(output_dir + "/ellipse_mask.nii", ellipse_mask )

mask[ellipse_mask == 1] = 1
tend = datetime.now()
tdiff = tend-tstart
print 'Time: ', tdiff.seconds, ' seconds'

print 'fill holes, close and dilate'
tstart = datetime.now()
disk_close = irtk.disk( 5 )
disk_dilate = irtk.disk( 2 )
for z in xrange(mask.shape[0]):
    mask[z] = nd.binary_fill_holes( mask[z] )
    mask[z] = nd.binary_closing( mask[z], disk_close )
    mask[z] = nd.binary_dilation( mask[z], disk_dilate )

neg_mask = np.ones(mask.shape, dtype='uint8')*2

# irtk.imwrite(output_dir + "/mask.nii", mask )
# irtk.imwrite(output_dir + "/mask.vtk", mask )

#x,y,z = img.WorldToImage(center)
x,y,z = center
x = int(round( x / img.header['pixelSize'][0] ))
y = int(round( y / img.header['pixelSize'][1] ))
z = int(round( z / img.header['pixelSize'][2] ))

w = h = int(round( ofd / img.header['pixelSize'][0]))
d = int(round( ofd / img.header['pixelSize'][2]))

#print 'zyx ', z,y,x
#print 'whd ', w,h,d

# cropped = img[max(0,z-d/2):min(img.shape[0],z+d/2+1),
#                max(0,y-h/2):min(img.shape[1],y+h/2+1),
#                max(0,x-w/2):min(img.shape[2],x+w/2+1)]

#irtk.imwrite(output_dir + "/cropped.nii", cropped )
#irtk.imwrite(output_dir + "/cropped.vtk", cropped )

neg_mask[max(0,z-d/2):min(img.shape[0],z+d/2+1),
         max(0,y-h/2):min(img.shape[1],y+h/2+1),
         max(0,x-w/2):min(img.shape[2],x+w/2+1)] = 0

mask[neg_mask>0] = 2

#print mask, mask.max()
irtk.imwrite(output_mask, mask )
tend = datetime.now()
tdiff = tend-tstart
print 'Time: ', tdiff.seconds, ' seconds'
