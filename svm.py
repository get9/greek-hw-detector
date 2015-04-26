from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from read_data import *
import sys


if len(sys.argv) < 4:
    print("Usage:")
    print("    {} modernGreekDir ancientGreekDir featureVecLen".format(sys.argv[0]))
    sys.exit(1)

# Args
bmp_data_dir = sys.argv[1]
test_data_dir = sys.argv[2]
feature_vec_len = int(sys.argv[3])

# Read in image bmp_data and split it up into training/validation set
bmp_data, bmp_labels = read_toplevel_dir(bmp_data_dir, 'bmp', flatten=True, concat=True)
test_data, test_labels = read_toplevel_dir(test_data_dir, 'png', flatten=True, concat=True)

# Split into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(bmp_data, bmp_labels, test_size=0.5, random_state=42)

# Reduce dimensionality of image bmp_data via PCA
print("Performing PCA to reduce feature dimensionality")
pca = PCA(n_components=feature_vec_len)
train_data = pca.fit_transform(train_data)
val_data = pca.transform(val_data)
test_data = pca.transform(test_data)

# Scale to make distance metrics easier
print("Scaling features to make mean = 0")
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

# Hyperparameters estimated with GridSearchCV
tuned_parameters = [
    {
        'kernel': ['rbf'],
        'gamma':  [1e-2, 1e-1, 1e-0, 1e1],
        'C':      [0.001, 0.01, 0.1, 1, 10, 100],
    },
    #{
    #    'kernel': ['linear'],
    #    'C':      [0.001, 0.01, 0.1, 1, 10],
    #},
]

scores = ['precision', 'recall']

# Fit with SVC classifier
svc = SVC(kernel='rbf', C=10, gamma=0.01)

# Perform a grid search over all above parameters
#print("Performing grid search of SVC hyperparameters")
#for s in scores:
#    gridsearch = GridSearchCV(svc, tuned_parameters, scoring='{}_weighted'.format(s), n_jobs=-1, verbose=3)
#    gridsearch.fit(train_data, train_labels)
#
#    print("Best parameters set found on training data:")
#    print(gridsearch.best_params_)
#    print()
#    print("Grid scores on training data:")
#    for params, mean_score, scores in gridsearch.grid_scores_:
#        print("{:3} (+/- {:3}) for {}".format(mean_score, scores.std()*2, params))
#    print()
#    print("Detailed classification report:\n")
#    print("The model is trained on the full development set.")
#    print("The scores are computed on the full evaluation set.\n")
#    val_predict = gridsearch.predict(val_data)
#    print(classification_report(val_labels, val_predict))
#    print()


svc.fit(train_data, train_labels)
val_prediction = svc.predict(val_data)
print("="*20)
print(svc)
print(classification_report(val_prediction, val_labels))

#print("Validation Confusion matrix")
#print("="*20)
print(confusion_matrix(val_labels, val_prediction))

# Get validation accuracy
val_correct = np.count_nonzero(val_labels == val_prediction)
print("Validation Accuracy: {}".format(val_correct * 100.0 / val_labels.size))

# Get testing accuracy
print(test_labels)
test_prediction = svc.predict(test_data)
print(test_prediction)
test_correct = np.count_nonzero(test_labels == test_prediction)
print("Test accuracy: {}".format(test_correct * 100.0 / test_labels.size))
print(classification_report(test_prediction, test_labels))
