import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

import source.util as util
import pandas as pd
from source.classifiers.classifier_pytorch import MLPClassifier
import source.classifiers.classifier_pytorch as torch_clf

CKPT_PATH = "/home/alexandre/logs/classif9aout/guitar_mono_aggregated/version_0/checkpoints/epoch=61-step=7998.ckpt"
DATA_PATH = "/home/alexandre/dataset/IDMT_guitar_mono_CUT_22050/out.csv"

AGGREGATE = True

CLASSES = ['Dry', 'Feedback Delay', 'Slapback Delay', 'Reverb',
           'Chorus', 'Flanger', 'Phaser',
           'Tremolo', 'Vibrato', 'Distortion', 'Overdrive']

if AGGREGATE:
    CLASSES = ["Modulation", "Delay", "Distortion", "Reverb", "Tremolo", "Dry"]


dataset = pd.read_csv(DATA_PATH, index_col=0)
subset = dataset.drop(columns=['file'])
if AGGREGATE:
    subset['class'] = subset['class'].apply(util.aggregated_class)
target = subset['class']
data = subset.drop(columns=['class'])
print(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=2,
                                                    stratify=target)


X_test = torch.tensor(X_test.values, dtype=torch.float)
y_test = torch.tensor(y_test.values)

sss = StratifiedShuffleSplit(n_splits=1, test_size=1 / 9, random_state=2)
i = 0
for train_index, valid_index in sss.split(X_train, y_train):
    i += 1
    X_train_cv = X_train.iloc[train_index]
    X_valid_cv = X_train.iloc[valid_index]
    y_train_cv = y_train.iloc[train_index]
    y_valid_cv = y_train.iloc[valid_index]
    X_train_cv = torch.tensor(X_train_cv.values, dtype=torch.float)
    X_valid_cv = torch.tensor(X_valid_cv.values, dtype=torch.float)
    y_train_cv = torch.tensor(y_train_cv.values)
    y_valid_cv = torch.tensor(y_valid_cv.values)

    scaler = torch_clf.TorchStandardScaler()
    scaler.fit(X_train_cv)

X_test = scaler.transform(X_test.clone())

trained_clf = MLPClassifier.load_from_checkpoint(CKPT_PATH, input_size=163, output_size=6,
                                                 hidden_size=100, activation='sigmoid', solver='adam',
                                                 max_iter=200)
trained_clf.to('cpu').eval()

pred = trained_clf(X_test)
pred = torch.argmax(pred, dim=-1)

cm = confusion_matrix(y_test.numpy(), pred.numpy())
print(cm)
fig = util.make_confusion_matrix(cm, group_names=CLASSES, categories=CLASSES, cbar=False, percent=True, count=False,
                                 unbalanced_set=True)
fig.show()
plt.show(block=True)


