import sys
import cv2
import numpy as np
import math
from read_data import read_toplevel_dir, IMG_WIDTH, IMG_HEIGHT

SZ = 50
bin_n = 16 # Number of bins

svm_params = dict(
    kernel_type = cv2.SVM_LINEAR,
    svm_type    = cv2.SVM_C_SVC,
    C           = 1.67,
    gamma       = 5.383
)

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def deskew(img):
    m = cv2.moments(img, binaryImage=True)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:SZ/2,:SZ/2], bins[SZ/2:,:SZ/2], bins[:SZ/2,SZ/2:], bins[SZ/2:,SZ/2:]
    mag_cells = mag[:SZ/2,:SZ/2], mag[SZ/2:,:SZ/2], mag[:SZ/2,SZ/2:], mag[SZ/2:,SZ/2:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

data, labels = read_toplevel_dir(sys.argv[1])

# First half is trainData, remaining is testData
imgs = np.vsplit(data, data.shape[0]/IMG_HEIGHT)
train_imgs, test_imgs = imgs[:len(imgs)/2], imgs[len(imgs)/2:]
train_labels, test_labels = labels[:labels.shape[0]/2], labels[labels.shape[0]/2:]

######     Now training      ########################
#deskewed = map(deskew, train_imgs)
hogdata = map(hog, train_imgs)
trainData = np.float32(hogdata).reshape(-1,bin_n*4)

svm = cv2.SVM()
svm.train(trainData, train_labels, params=svm_params)
svm.save('svm_data.dat')

######     Now testing      ########################
#deskewed = map(deskew, test_imgs)
hogdata = map(hog, test_imgs)
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict_all(testData)

#######   Check Accuracy   ########################
mask = result==test_labels
correct = np.count_nonzero(mask)
print correct*100.0/result.size
