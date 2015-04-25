from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from read_data import *
import sys


if len(sys.argv) < 4:
    print("Usage:")
    print("    {} valImageDir testImageDir featureVecLen".format(sys.argv[0]))
    sys.exit(1)

# Args
bmp_data_dir = sys.argv[1]
test_data_dir = sys.argv[2]
feature_vec_len = int(sys.argv[3])

# Read in image bmp_data and split it up into training/validation set
bmp_data, bmp_labels = read_toplevel_dir(bmp_data_dir, 'bmp', ravel=True, concat=True)
test_data, test_labels = read_toplevel_dir(test_data_dir, 'png', ravel=True, concat=True)

# Split into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(bmp_data, bmp_labels, test_size=0.5)

# Reduce dimensionality of image bmp_data via PCA
pca = PCA(n_components=feature_vec_len)
train_data = pca.fit_transform(train_data)
val_data = pca.transform(val_data)
test_data = pca.transform(test_data)

# Scale to make distance metrics easier
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

# Hyperparameters estimated with GridSearchCV
tuned_parameters = [
    {
        'kernel': ['rbf'],
        'gamma':  [1e-4, 1e-3, 1e-2, 1e-1],
        'C':      [1, 10, 100, 1000],
    },
    {
        'kernel': ['linear'],
        'C':      [1, 10, 100, 1000],
    },
]

scores = [
    ('precision', precision_score),
    ('recall',    recall_score),
]

# Fit with SVC classifier
svc = SVC(C=8)
svc.fit(train_data, train_labels)
val_prediction = svc.predict(val_data)
print("="*20)
print(svc)
print(classification_report(val_prediction, val_labels))

#print("Validation Confusion matrix")
#print("="*20)
#print(confusion_matrix(val_labels, val_prediction))

# Get validation accuracy
val_correct = np.count_nonzero(val_labels == val_prediction)
print("Validation Accuracy: {}".format(val_correct * 100.0 / val_labels.size))

# Get testing accuracy
test_prediction = svc.predict(test_data)
test_correct = np.count_nonzero(test_labels == test_prediction)
print("Test accuracy: {}".format(test_correct * 100.0 / test_labels.size))
print(classification_report(test_prediction, test_labels))
