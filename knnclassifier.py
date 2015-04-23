from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from read_data import *
import sys


if len(sys.argv) < 4:
    print("Usage:")
    print("    {} test_img_dir fvec_len n")
    sys.exit(1)

# Read in image data and split it up into training/validation set
data, labels = read_toplevel_dir(sys.argv[1], 'bmp', ravel=True)
data = np.concatenate(data)
labels = np.array(labels).ravel()

# Split into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.5)

# Reduce dimensionality of image data via PCA
pca = RandomizedPCA(n_components=int(sys.argv[2]))
train_data = pca.fit_transform(train_data)
val_data = pca.transform(val_data)

# Scale to make distance metrics easier
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)

# Fit with kNN classifier
knn = KNeighborsClassifier(n_neighbors=int(sys.argv[3]), weights='distance')
knn.fit(train_data, train_labels)
prediction = knn.predict(val_data)
print("="*20)
print(knn)

print("Confusion matrix")
print("="*20)
print(confusion_matrix(val_labels, prediction))

correct = np.count_nonzero(val_labels == prediction)
print("Accuracy: {}".format(correct * 100.0 / val_labels.size))
