import pickle

import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import source.util as util
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
import source.classifiers.classifier_pytorch as torch_clf

CLASSES = ['Dry', 'Feedback Delay', 'Slapback Delay', 'Reverb',
           'Chorus', 'Flanger', 'Phaser',
           'Tremolo', 'Vibrato', 'Distortion', 'Overdrive']
AGGREGATE = True
OUT_PATH = "/home/alexandre/logs/classif9aout"
dataset = pd.read_csv('/home/alexandre/dataset/IDMT_guitar_mono_CUT_22050/out.csv', index_col=0)
subset = dataset.drop(columns=['file'])
if AGGREGATE:
    subset['class'] = subset['class'].apply(util.aggregated_class)
target = subset['class']
data = subset.drop(columns=['class'])
print(data)


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=2,
                                                    stratify=target)


sss = StratifiedShuffleSplit(n_splits=1, test_size=1 / 9, random_state=2)
i = 0
for train_index, valid_index in sss.split(X_train, y_train):
    print("Working on fold", i)
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
    X_train_scaled = scaler.transform(X_train_cv.clone())
    X_valid_scaled = scaler.transform(X_valid_cv.clone())

    train_dataset = torch_clf.ClassificationDataset(X_train_scaled, y_train_cv)
    valid_dataset = torch_clf.ClassificationDataset(X_valid_scaled, y_valid_cv)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                                  num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=128, num_workers=4)

    clf = torch_clf.MLPClassifier(len(data.columns), 6 if AGGREGATE else 11, 100, activation='sigmoid', solver='adam',
                                  max_iter=200, learning_rate=0.002)

    logger = TensorBoardLogger(OUT_PATH, name="guitar_mono_aggregated")
    # early_stop_callback = EarlyStopping(monitor="train_loss",
    #                                     min_delta=clf.tol,
    #                                    patience=clf.n_iter_no_change)

    early_stopping = EarlyStopping(monitor='loss/test', patience=10)
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor='loss/test')
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=clf.max_iter,
                         accelerator='ddp',
                         auto_select_gpus=True, log_every_n_steps=10,
                         callbacks=[checkpoint_callback, early_stopping])

    trainer.fit(clf, train_dataloader, valid_dataloader)
    with open(logger.log_dir + '/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
