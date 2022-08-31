"""
Multi Layer Perceptron (MLP) classifier for baseline comparison with the SVM.
"""

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

import pandas as pd
import source.util as util
from time import time
import pickle
# import pyRAPL

CLASSES = ['Dry', 'Feedback Delay', 'Slapback Delay', 'Reverb', 'Chorus', 'Flanger', 'Phaser',
           'Tremolo', 'Vibrato', 'Distortion', 'Overdrive']


dataset = pd.read_csv('/home/alexandre/dataset/full_dataset.csv', index_col=0)
subset = dataset.drop(columns=['flux_min'])
target = []
for fx in subset['target_name']:
    target.append(util.idmt_fx2class_number(fx))
data = subset.drop(columns=['target_name'])
print(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
print(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier(activation='logistic', solver='adam', max_iter=500)
print(clf.get_params())

# pyRAPL.setup()
# measure = pyRAPL.Measurement('MLP')
# measure.begin()
print("Training...")
start = time()
clf.fit(X_train, y_train)
end = time()
print("Training took: ", end-start)
print("Best loss: ", clf.best_loss_)
print("Loss curve: ", clf.loss_curve_)
print("num iter: ", clf.n_iter_)
print("num layers: ", clf.n_layers_)
print("batch_size: ", clf.batch_size)
print("hidden layer sizes: ", clf.hidden_layer_sizes)
# measure.end()
# print(measure.result)

y_pred = clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision: ", metrics.precision_score(y_test, y_pred, average=None))
print("Recall: ", metrics.recall_score(y_test, y_pred, average=None))
print(metrics.confusion_matrix(y_test, y_pred))
print(CLASSES)
with open("mlp_full_dataset.pkl", 'wb') as f:
    pickle.dump((clf, scaler), f)
