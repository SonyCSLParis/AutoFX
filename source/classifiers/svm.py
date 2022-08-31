from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
import sklearn as skl
from sklearn import metrics
import pandas as pd
import util
from time import time
import pickle
import pyRAPL

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
X_test = scaler.transform(X_test)
clf = svm.SVC(kernel='rbf', C=1000, gamma=0.001)
# clf = svm.LinearSVC()
pyRAPL.setup()

measure = pyRAPL.Measurement('bar')
measure.begin()
print("Training...")
start = time()
clf.fit(X_train, y_train)
end = time()
measure.end()
print(measure.result)
print("Training took: ", end-start)

y_pred = clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Precision: ", metrics.precision_score(y_test, y_pred, average=None))
print("Recall: ", metrics.recall_score(y_test, y_pred, average=None))
print(metrics.confusion_matrix(y_test, y_pred))
print(CLASSES)
with open("svm_full_dataset.pkl", 'wb') as f:
    pickle.dump((clf, scaler), f)
