import sys
import numpy as np
import math
import cv2

from read_data import read_toplevel_dir
from matplotlib import pyplot as plt

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
    raw_data, labels = read_toplevel_dir(indir)

    # Set training and testing data/labels
    train_data, test_data = divide_array(raw_data, trainratio=0.5)
    train_labels, test_labels = divide_array(labels, trainratio=0.5)

    # Set up kNN
    knn = cv2.KNearest()
    knn.train(train_data, train_labels)

    # Testing
    k = range(4, 21)
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

main()
