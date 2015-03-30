# -*- coding: utf-8 -*-

import numpy as np
import sys
import cv2

from glob import glob
from os.path import join, isdir, basename, splitext
from itertools import izip_longest, chain

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
    print(directory)
    dir_label = splitext(directory.rstrip('/'))[1].strip('.').lower()
    label = TRAINING_LABELS[dir_label]

    # Read in files. Each elem of imgarrays is a numpy.ndarray
    bmpfilenames = glob(join(directory, '*.bmp'))
    imgarrays = [cv2.imread(bmp, cv2.IMREAD_GRAYSCALE) for bmp in bmpfilenames]

    # Find max image width/height from arrays and set that param
    maxheight = sorted(imgarrays, key=lambda x: x.shape[0])[-1].shape[0]
    maxwidth = sorted(imgarrays, key=lambda x: x.shape[1])[-1].shape[1]

    # imgsdata is list of numpy arrays of pixel data
    imgsdata = map(lambda i: i.flatten(), imgarrays)
    labels = [label for _ in imgsdata]
    data_with_labels = zip(imgsdata, labels)
    return zip(imgsdata, labels)

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
        total_num_samples += len(images)

    # Need to swizzle images together
    swizzled_imgs = list(chain.from_iterable(izip_longest(*imglist)))

    # Remove None elements that were introduced from swizzling
    swizzled_imgs = filter(None, swizzled_imgs)

    # Find the max dimension across npixels to correctly size the output array
    maxnpixels = sorted(swizzled_imgs, key=lambda x: x[0].shape[0])[-1][0].shape[0]

    # Resize each image array to a (1, maxpixels) array
    for i, img in enumerate(swizzled_imgs):
        tmpimg = np.zeros((1, maxnpixels), dtype=np.float32)
        tmpimg[0, :img[0].shape[0]] = img[0]
        swizzled_imgs[i] = (tmpimg, img[1])

    # Concatenate them together into giant list of all samples of all letters
    data, labels = zip(*swizzled_imgs)
    images = np.concatenate(data)
    labels = np.array(labels)[:, np.newaxis]

    return [images, labels]
