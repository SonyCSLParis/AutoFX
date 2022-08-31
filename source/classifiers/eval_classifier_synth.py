import torch
import pytorch_lightning as pl
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

import source.util as util
import pandas as pd
from source.classifiers.classifier_pytorch import MLPClassifier
import source.classifiers.classifier_pytorch as torch_clf

CKPT_PATH = "/home/alexandre/logs/classif9aout/guitar_mono_aggregated/version_0/checkpoints/epoch=61-step=7998.ckpt"
SCALER_PATH = "/home/alexandre/logs/classif9aout/guitar_mono_aggregated/version_0/scaler.pkl"
# DATA_PATH = "/home/alexandre/dataset/IDMT_guitar_mono_CUT_22050/out.csv"
DATA_PATH = "/home/alexandre/dataset/modulation_delay_distortion_guitar_mono_cut/out.csv"
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


X_test = torch.tensor(data.values, dtype=torch.float)
y_test = torch.tensor(target.values)


with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

X_test = scaler.transform(X_test.clone())

trained_clf = MLPClassifier.load_from_checkpoint(CKPT_PATH, input_size=163, output_size=6,
                                                 hidden_size=100, activation='sigmoid', solver='adam',
                                                 max_iter=200)
trained_clf.to('cpu').eval()

pred = trained_clf(X_test)
pred = torch.argmax(pred, dim=-1)

cm = confusion_matrix(y_test.numpy(), pred.numpy())
print(cm)
fig = util.make_confusion_matrix(cm, group_names=CLASSES, categories=CLASSES, cbar=False, percent=False, count=True,
                                 unbalanced_set=False)
fig.show()
plt.show(block=True)


