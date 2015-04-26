from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from read_data import *
import sys


if len(sys.argv) < 5:
    print("Usage:")
    print("    {} valImageDir testImageDir featureVecLen neighbors".format(sys.argv[0]))
    sys.exit(1)

# Args
bmp_data_dir = sys.argv[1]
test_data_dir = sys.argv[2]
feature_vec_len = int(sys.argv[3])
num_neighbors = int(sys.argv[4])

# Read in image bmp_data and split it up into training/validation set
bmp_data, bmp_labels = read_toplevel_dir(bmp_data_dir, 'bmp', flatten=True, concat=True)
test_data, test_labels = read_toplevel_dir(test_data_dir, 'png', flatten=True, concat=True)

# Split into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(bmp_data, bmp_labels, test_size=0.5)

# Reduce dimensionality of image bmp_data via PCA
pca = PCA(n_components=feature_vec_len)
train_data = pca.fit_transform(train_data)
val_data = pca.transform(val_data)

# Scale to make distance metrics easier
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)

# Fit with kNN classifier
knn = KNeighborsClassifier(n_neighbors=num_neighbors, weights='distance')
knn.fit(train_data, train_labels)
val_prediction = knn.predict(val_data)
#print("="*20)
#print(knn)

#print("Validation Confusion matrix")
#print("="*20)
#print(confusion_matrix(val_labels, val_prediction))

# Get validation accuracy
val_correct = np.count_nonzero(val_labels == val_prediction)
print("Validation Accuracy: {}".format(val_correct * 100.0 / val_labels.size))

# Transform test feature vectors
test_data = pca.transform(test_data)

test_prediction = knn.predict(test_data)
print(classification_report(test_prediction, test_labels))
test_correct = np.count_nonzero(test_labels == test_prediction)
print("Test accuracy: {}".format(test_correct * 100.0 / test_labels.size))
