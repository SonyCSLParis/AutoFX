"""
Simple neural network classifier for baseline comparison with the SVM.
"""

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import Perceptron

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


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = Perceptron(fit_intercept=True, alpha=1e-4, eta0=0.1, penalty='elasticnet', tol=0.01, l1_ratio=1)

print("Training...")
start = time()
clf.fit(X_train, y_train)
end = time()
print("Training took: ", end-start)

y_pred = clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision: ", metrics.precision_score(y_test, y_pred, average=None))
print("Recall: ", metrics.recall_score(y_test, y_pred, average=None))
print(metrics.confusion_matrix(y_test, y_pred))
print(CLASSES)
with open("perceptron_opti.pkl", 'wb') as f:
    pickle.dump(clf, f)
