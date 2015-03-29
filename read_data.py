# -*- coding: utf-8 -*-

import numpy as np
import sys
import cv2

from glob import glob
from os.path import join, isdir, basename

TRAINING_LABELS = [
    'alpha',
    'beta',
    'delta',
    'gamma',
    'epsilon',
    'zeta',
    'heta',
    'theta',
    'iota',
    'kapa',
    'lamda',
    'mi',
    'ni',
    'xi',
    'omikron',
    'pii',
    'ro',
    'sigma',
    'tau',
    'ypsilon',
    'fi',
    'xsi',
    'psi',
    'omega',
]

# Reads images from 'directory', collects them as np array of pixel values,
# and returns a 2D array (#samples, pixel data) and the corresponding label
def read_image_dir(directory):
    if not isdir(directory):
        raise IOError("{} is not a directory".format(directory))

    # Determine training label
    label = ''
    for l in TRAINING_LABELS:
        if l in basename(directory):
            label = l
            break

    # Read in files. Each elem of imgarrays is a numpy.ndarray
    bmpfilenames = glob(join(directory, '*.bmp'))
    imgarrays = [cv2.imread(bmp, cv2.IMREAD_GRAYSCALE) for bmp in bmpfilenames]

    # Find max image width/height from arrays and set that param
    maxheight = sorted(imgarrays, key=lambda x: x.shape[0])[-1].shape[0]
    maxwidth = sorted(imgarrays, key=lambda x: x.shape[1])[-1].shape[1]

    # imgsdata is list of numpy arrays of pixel data
    imgsdata = map(lambda i: i.flatten(), imgarrays)
    return [imgsdata, label]

# Reads images from each directory inside 'toplevel' by calling read_image_dir
# on each one of them and putting results into a numpy array
def read_toplevel_dir(directory, formatstr=""):
    if not isdir(directory):
        raise IOError("{} is not a directory".format(directory))

    # Temporarily store images in a list until we can figure out how big to
    # make the full numpy.array
    dirs = glob( join(directory, '{}*'.format(formatstr)) )
    imglist = []
    train_labels = []
    total_num_samples = 0
    for d in dirs:
        images, label = read_image_dir(d)
        imglist.extend(images)
        # Repeat the training labels for the number of samples there were for
        # that letter
        train_labels.extend( [label for _ in range( len(images) )] )
        total_num_samples += len(images)

    # Find the max dimension across npixels to correctly size the output array
    maxnpixels = sorted(imglist, key=lambda x: x.shape[0])[-1].shape[0]

    # Make new numpy.ndarray with padding to store all vectors of image data
    # Dimensions: (nsamples, npixels)
    # npixels is padded to be the highest amount in dataset
    for i, img in enumerate(imglist):
        tmpimg = np.zeros((1, maxnpixels))
        tmpimg[0, :img.shape[0]] = img
        imglist[i] = tmpimg

    images = np.concatenate(imglist)
    labels = np.array(train_labels)[:, np.newaxis]

    return [images, labels]
