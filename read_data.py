import numpy as np
import sys

from glob import glob
from os.path import join, isdir

# Reads images from 'directory', collects them as np array of pixel values,
# and returns as a 4D array of dim [#imgs, #letters, imgwidth, imgheight]
def read_image_dir(directory):
    if not isdir(directory):
        raise IOError("{} is not a directory".format(directory))

    # Read in files. Each elem of imgarrays is a numpy.ndarray
    bmpfilenames = glob(join(directory, '*.bmp'))
    imgarrays = [cv2.imread(bmp, cv2.IMREAD_GRAYSCALE) for bmp in bmpfilenames]

    # Find max image width/height from arrays and set that param
    maxheight = sorted(imgarrays, key=lambda x: x.shape[0])[-1].shape[0]
    maxwidth = sorted(imgarrays, key=lambda x: x.shape[1])[-1].shape[1]

    # Create new numpy ndarray to store all image data
    imgsdata = np.zeros( (len(bmpfilenames), maxheight, maxwidth), dtype=np.uint8 )
    for i, img in enumerate(imgarrays):
        imgsdata[i, :img.shape[0], :img.shape[1]] = img

    return imgsdata

# Reads images from each directory inside 'toplevel' by calling read_image_dir
# on each one of them and putting results into a numpy array
def read_toplevel_dir(directory, formatstr=""):
    if not isdir(directory):
        raise IOError("{} is not a directory".format(directory))

    # Temporarily store images in a list until we can figure out how big to
    # make the full numpy.array
    dirs = glob( join(directory, '{}*'.format(formatstr)) )
    imglist = [read_image_dir(d) for d in dirs]

    # Find the max dimension across #letters, imgwidth, imgheight to correctly
    # size the output array
    samplelen = sorted(imglist, key=lambda x: x.shape[0])[-1].shape[0]
    maxheight = sorted(imglist, key=lambda x: x.shape[1])[-1].shape[1]
    maxwidth = sorted(imglist, key=lambda x: x.shape[2])[-1].shape[2]

    # Make new numpy.ndarray with padding to store all vectors of image data
    # Dimensions are as follows:
    #     # samples (extended to longest # samples)
    #     # letters
    #     height (pixels)
    #     width (pixels)
    images = np.zeros( (len(dirs), samplelen, maxheight, maxwidth), dtype=np.uint8 )
    for i, img in enumerate(imglist):
        images[i, :img.shape[0], :img.shape[1], :img.shape[2]] = img

    return images
