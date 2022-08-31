"""
Classifier using k-Nearest Neighbors algorithm
"""
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
from time import time
import pickle
import sys
sys.path.append('..')

import util

CLASSES = ['Dry', 'Feedback Delay', 'Slapback Delay', 'Reverb', 'Chorus', 'Flanger', 'Phaser',
           'Tremolo', 'Vibrato', 'Distortion', 'Overdrive']


dataset = pd.read_csv('/home/alexandre/dataset/dataset.csv')
dataset.drop(columns=['flux_min'])
subset = dataset
target = []
for fx in subset['target_name']:
    target.append(util.idmt_fx2class_number(fx))
data = subset.drop(columns=['target_name'])


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

knn_clf = KNeighborsClassifier(n_neighbors=50, weights='distance')
print("Training...")
start = time()
knn_clf.fit(X_train, y_train)
end = time()
print("Training took: ", end-start)

X_test = scaler.transform(X_test)
y_pred = knn_clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision: ", metrics.precision_score(y_test, y_pred, average=None))
print("Recall: ", metrics.recall_score(y_test, y_pred, average=None))
print(metrics.confusion_matrix(y_test, y_pred))
print(CLASSES)
with open("classifiers/knn_v0.pkl", 'wb') as f:
    pickle.dump(knn_clf, f)
