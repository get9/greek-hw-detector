# -*- coding: utf-8 -*-

import numpy as np
import sys
import cv2

from glob import glob
from os.path import join, isdir, basename, splitext
from itertools import izip_longest, chain

IMG_WIDTH  = 51
IMG_HEIGHT = 49

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
    'ksi': 21,
    'psi': 22,
    'omega': 23,
}

# Reads images from 'directory', collects them as np array of pixel values,
# and returns a 2D array (#samples, pixel data) and the corresponding label
def read_image_dir(directory):
    if not isdir(directory):
        raise IOError("{} is not a directory".format(directory))

    # Determine training label
    dir_label = splitext(directory.rstrip('/'))[1].strip('.').lower()
    label = TRAINING_LABELS[dir_label]
    #print("{} --> {}, {}".format(basename(directory), dir_label, label))

    # Read in files. Each elem of imgarrays is a numpy.ndarray
    bmpfilenames = glob(join(directory, '*.bmp'))
    imgarrays = [cv2.imread(bmp, cv2.IMREAD_GRAYSCALE) for bmp in bmpfilenames]

    # imgsdata is list of numpy arrays of pixel data
    labels = [label for _ in imgarrays]
    return zip(imgarrays, labels)

# Reads images from each directory inside 'toplevel' by calling read_image_dir
# on each one of them and putting results into a numpy array
def read_toplevel_dir(directory, formatstr=""):
    if not isdir(directory):
        raise IOError("{} is not a directory".format(directory))

    # Temporarily store images in a list until we can figure out how big to
    # make the full numpy.array
    dirs = glob( join(directory, '{}*'.format(formatstr)) )
    imglist = []
    total_num_samples = 0
    for d in dirs:
        # images = [(np.array, label), ...]
        images = read_image_dir(d)
        # imglist is [[(np.array, label), ...], ...]
        imglist.append(images)

    # Need to swizzle images together
    swizzled_imgs = list(chain.from_iterable(izip_longest(*imglist)))

    # Remove None elements that were introduced from swizzling
    swizzled_imgs = filter(None, swizzled_imgs)

    # Concatenate them together into giant list of all samples of all letters
    data, labels = zip(*swizzled_imgs)
    images = np.concatenate(data)
    labels = np.array(labels)[:, np.newaxis]

    return [images, labels]
