# -*- coding: utf-8 -*-

import numpy as np
import skimage.io
import sys

from glob import glob
from os.path import join, isdir, basename, splitext
from itertools import zip_longest, chain

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
def read_image_dir(directory, extension):
    if not isdir(directory):
        raise IOError("{} is not a directory".format(directory))

    # Determine training label
    dir_label = splitext(directory.rstrip('/'))[1].strip('.').lower()
    label = TRAINING_LABELS[dir_label]
    #print("{} --> {}, {}".format(basename(directory), dir_label, label))

    # Read in files. Each elem of imgarrays is a numpy.ndarray
    bmpfilenames = glob(join(directory, '*.{}'.format(extension)))
    imgarrays = [skimage.io.imread(bmp, as_grey=True) for bmp in bmpfilenames]

    # imgsdata is list of numpy arrays of pixel data
    labels = [label for _ in imgarrays]
    return zip(imgarrays, labels)

# Reads images from each directory inside 'toplevel' by calling read_image_dir
# on each one of them and putting results into a numpy array
def read_toplevel_dir(directory, extension, formatstr="", flatten=False, concat=False):
    if not isdir(directory):
        raise IOError("{} is not a directory".format(directory))

    dirs = glob( join(directory, '{}*'.format(formatstr)) )
    imglist = []
    for d in dirs:
        imglist.extend(read_image_dir(d, extension))

    # Concatenate them together into giant list of all samples of all letters
    data, labels = zip(*imglist)
    labels = np.array(labels)

    # If we need to ravel the data (flatten it all out so that images are 1-D and
    # labels are 1-D on primary axis), do so
    if flatten:
        data = list(map(lambda i: i.reshape((1, IMG_WIDTH * IMG_HEIGHT)), data))

    # If we have to concat, then we need to put the data arrays together
    if concat:
        data = np.concatenate(data)

    return [data, labels]
