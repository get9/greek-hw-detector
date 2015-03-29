# -*- coding: utf-8 -*-

import numpy as np
import sys
import cv2

from glob import glob
from os.path import join, isdir, basename

TRAINING_LABELS = {
    'alpha': 0,
    'beta': 1,
    'delta': 2,
    'gamma': 3,
    'epsilon': 4,
    'zeta': 5,
    'heta': 6,
    'theta': 7,
    'iota': 8,
    'kapa': 9,
    'lamda': 10,
    'mi': 11,
    'ni': 12,
    'xi': 13,
    'omikron': 14,
    'pii': 15,
    'ro': 16,
    'sigma': 17,
    'tau': 18,
    'ypsilon': 19,
    'fi': 20,
    'xsi': 21,
    'psi': 22,
    'omega': 23,
}

# Reads images from 'directory', collects them as np array of pixel values,
# and returns a 2D array (#samples, pixel data) and the corresponding label
def read_image_dir(directory):
    if not isdir(directory):
        raise IOError("{} is not a directory".format(directory))

    # Determine training label
    label = 0
    for l in TRAINING_LABELS:
        if l.upper() in basename(directory):
            label = TRAINING_LABELS[l]
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

    # Resize each image array to a (1, maxpixels) array
    for i, img in enumerate(imglist):
        tmpimg = np.zeros((1, maxnpixels), dtype=np.float32)
        tmpimg[0, :img.shape[0]] = img
        imglist[i] = tmpimg

    # Concatenate them together into giant list of all samples of all letters
    images = np.concatenate(imglist)
    labels = np.array(train_labels)[:, np.newaxis]

    return [images, labels]
