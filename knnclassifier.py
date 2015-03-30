# -*- coding: utf-8 -*-

import sys
import numpy as np
import math
import cv2

from read_data import read_toplevel_dir
from matplotlib import pyplot as plt

# Image classes are divided into 24/25 letters, with:
#     - Capitals (24 letters)
#           - Normal
#           - Suffix
#     - Lowercase (25 letters)
#           - Normal
#           - Suffix
#
# Because there aren't an equal number of pixels or images in each dir in data,
# Need to pad numpy arrays with zeros in read_data

# XXX Only focus on capital, normal letters for now

def divide_array(array, trainratio=0.5):
    divpoint = int( math.floor(array.shape[0] * trainratio) )
    return [array[:divpoint], array[divpoint:]]

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("    {} data-file-toplevel-dir".format(sys.argv[0]))
        sys.exit(1)

    indir = sys.argv[1]

    # Read in data to giant numpy array
    # raw_data dimensions:
    #     (#letters, #samples, imgheight, imgwidth)
    raw_data, labels = read_toplevel_dir(indir)

    # Set training and testing data/labels
    train_data, test_data = divide_array(raw_data, trainratio=0.5)
    train_labels, test_labels = divide_array(labels, trainratio=0.5)

    # Set up kNN
    knn = cv2.KNearest()
    knn.train(train_data, train_labels)

    # Testing
    k = range(1, 21)
    accuracies = []
    for i in k:
        ret, result, neighbors, dist = knn.find_nearest(test_data, k=i)
        matches = result == test_labels
        correct = np.count_nonzero(matches)
        accuracy = correct*100.0/result.size
        accuracies.append(accuracy)
        print("({}, {})".format(i, accuracy))

    # Plotting
    plt.plot(k, accuracies, 'ro')
    plt.axis('tight')
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.show()

#imgs = read_image_dir(indir)
#
#img = cv2.imread('digits.png')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
## Now we split the image to 5000 cells, each 20x20 size
#cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
#
## Make it into a Numpy array. It size will be (50,100,20,20)
#x = np.array(cells)
#
## Now we prepare train_data and test_data.
#train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
#test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
#
## Create labels for train and test data
#k = np.arange(10)
#train_labels = np.repeat(k,250)[:,np.newaxis]
#test_labels = train_labels.copy()
#
## Initiate kNN, train the data, then test it with test data for k=1
#knn = cv2.KNearest()
#knn.train(train,train_labels)
#ret,result,neighbours,dist = knn.find_nearest(test,k=5)
#
## Now we check the accuracy of classification
## For that, compare the result with test_labels and check which are wrong
#matches = result==test_labels
#correct = np.count_nonzero(matches)
#accuracy = correct*100.0/result.size
#print accuracy

main()
