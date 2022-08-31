"""
Script for finding best parameters to a kNN classifier
"""
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
import util
from time import time
import pickle

CLASSES = ['Dry', 'Feedback Delay', 'Slapback Delay', 'Reverb', 'Chorus', 'Flanger', 'Phaser',
           'Tremolo', 'Vibrato', 'Distortion', 'Overdrive']

dataset = pd.read_csv('/home/alexandre/dataset/dataset.csv')
dataset.drop(columns=['flux_min'])
subset = dataset
target = []
for fx in subset['target_name']:
    target.append(util.idmt_fx2class_number(fx))
data = subset.drop(columns=['target_name'])


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# tuned_parameters = [
#    {'weights': ['uniform'], 'n_neighbors': [1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000]},
#    {'weights': ['distance'], 'n_neighbors': [1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000]}
#]
tuned_parameters = [
    {'weights': ['distance'], 'n_neighbors': [50], }
]

scores = ["precision", "recall"]

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    knn_clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, scoring="%s_macro" % score, n_jobs=-1)
    knn_clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(knn_clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = knn_clf.cv_results_["mean_test_score"]
    stds = knn_clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, knn_clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, knn_clf.predict(X_test)
    print(metrics.classification_report(y_true, y_pred))
    print()