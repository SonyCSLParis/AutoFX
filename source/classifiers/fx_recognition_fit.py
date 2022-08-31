import argparse
import pathlib
import pickle
import sys

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


def main(parser):
    args = vars(parser.parse_args())
    AGGREGATE = args['aggregate']
    OUT_PATH = pathlib.Path(args['output_dir'])
    dataset = pd.read_csv(args['dataset'], index_col=0)
    subset = dataset.drop(columns=['file'])
    if AGGREGATE:
        subset['class'] = subset['class'].apply(util.aggregated_class)
    target = subset['class']
    data = subset.drop(columns=['class'])
    print("Working on data:\n", data)
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.2, random_state=2,
                                                        stratify=target)
    X_train = torch.tensor(X_train.values, dtype=torch.float)
    X_test = torch.tensor(X_test.values, dtype=torch.float)
    y_train = torch.tensor(y_train.values)
    y_test = torch.tensor(y_test.values)
    scaler = torch_clf.TorchStandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train.clone())
    X_test_scaled = scaler.transform(X_test.clone())

    train_dataset = torch_clf.ClassificationDataset(X_train_scaled, y_train)
    test_dataset = torch_clf.ClassificationDataset(X_test_scaled, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                                  num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=128, num_workers=4)
    clf = torch_clf.MLPClassifier(len(data.columns), 6 if AGGREGATE else 11, 100, activation='sigmoid', solver='adam',
                                  max_iter=200, learning_rate=0.002)
    logger = TensorBoardLogger(OUT_PATH)
    early_stopping = EarlyStopping(monitor='loss/test', patience=10)
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor='loss/test')
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=clf.max_iter,
                         auto_select_gpus=True, log_every_n_steps=10,
                         callbacks=[checkpoint_callback, early_stopping])

    trainer.fit(clf, train_dataloader, test_dataloader)
    with open(logger.log_dir + '/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script of the effect recognition network.")
    parser.add_argument('--output-dir', '-o', type=str,
                        help="Path to store the training logs")
    parser.add_argument('--dataset', '-d', type=str,
                        help="Path to the dataset .csv file")
    parser.add_argument('--aggregate', '-a', action='store_true',
                        help="Flag to use aggregated classes")
    sys.exit(main(parser))
