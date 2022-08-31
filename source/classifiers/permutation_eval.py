import pandas as pd
import sklearn as skl
from sklearn.inspection import permutation_importance
import torch
import numpy as np
from tqdm import tqdm

import source.util
from source.classifiers.classifier_pytorch import MLPClassifier, ClassificationDataset

CKPT_PATH = "/home/alexandre/logs/classif9aout/guitar_mono/version_0/checkpoints/epoch=87-step=11352.ckpt"
LOG_FILE = "/home/alexandre/logs/classif9aout/guitar_mono_aggregated/version_0/permutation_log.txt"
classif = MLPClassifier.load_from_checkpoint(CKPT_PATH,
                                             input_size=163, output_size=11, hidden_size=100,
                                             activation='sigmoid', solver='adam', max_iter=1000)

AGGREGATE = False

dataset = pd.read_csv('/home/alexandre/dataset/IDMT_guitar_mono_CUT_22050/out.csv', index_col=0)
subset = dataset.drop(columns=['file'])
if AGGREGATE:
    subset['class'] = subset['class'].apply(source.util.aggregated_class)
target = subset['class']
data = subset.drop(columns=['class'])

X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(data, target, test_size=0.1, random_state=2,
                                                                        stratify=target)

sss = skl.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=1 / 9, random_state=2)
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

    scaler = skl.preprocessing.StandardScaler()
    scaler.fit(X_train_cv)

X_test = scaler.transform(X_test)
NUM_REPEATS = 10
file = open(LOG_FILE, 'a')

dataset = ClassificationDataset(X_test, y_test)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
iterator = iter(dataloader)
precision = []
accuracy = []
recall = []
classif.prec.reset()
classif.recall.reset()
classif.accuracy.reset()
while True:
    try:
        feat, label = next(iterator)
        pred = classif(feat)
        classif.prec.update((pred, label))
        classif.recall.update((pred, label))
        classif.accuracy.update((pred, label))
    except StopIteration:
        ref_precision = classif.prec.compute()
        ref_recall = classif.recall.compute()
        ref_acc = classif.accuracy.compute()
        # print("Reference Precision: ", classif.prec.compute(), file=file)
        # print("Reference Recall: ", classif.recall.compute(), file=file)
        # print("Reference Accuracy: ", classif.accuracy.compute(), file=file)
        break

avg_precision = torch.empty((163, 11, NUM_REPEATS))
avg_accuracy = torch.empty((163, 1, NUM_REPEATS))
avg_recall = torch.empty((163, 11, NUM_REPEATS))
for n in tqdm(range(NUM_REPEATS)):
    # precision = torch.tensor([])
    # accuracy = []
    # recall = torch.tensor([])
    for c in tqdm(range(X_test.shape[1])):
        shuffled = X_test.copy()
        shuffled[:, c] = np.random.permutation(shuffled[:, c])
        # print(shuffled)
        # print(y_test)
        dataset = ClassificationDataset(shuffled, y_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
        print("Shuffling column ", c, file=file)
        iterator = iter(dataloader)
        classif.prec.reset()
        classif.recall.reset()
        classif.accuracy.reset()
        while True:
            try:
                feat, label = next(iterator)
                pred = classif(feat)
                classif.prec.update((pred, label))
                classif.recall.update((pred, label))
                classif.accuracy.update((pred, label))
            except StopIteration:
                avg_precision[c, :, n] = classif.prec.compute()
                avg_accuracy[c, :, n] = classif.accuracy.compute()
                avg_recall[c, :, n] = classif.recall.compute()
                # precision = torch.cat([precision, classif.prec.compute()], dim=0)
                # accuracy.append(classif.accuracy.compute())
                # recall = torch.cat([recall, classif.recall.compute()], dim=0)
                # print("Precision: ", classif.prec.compute(), file=file)
                # print("Recall: ", classif.recall.compute(), file=file)
                # print("Accuracy: ", classif.accuracy.compute(), file=file)
                break

avg_precision = torch.mean(avg_precision, dim=-1)
avg_accuracy = torch.mean(avg_accuracy, dim=-1)
avg_recall = torch.mean(avg_recall, dim=-1)

loss_acc = torch.tensor([ref_acc - acc for acc in avg_accuracy.flatten()])
loss_prec = torch.vstack([ref_precision - precision for precision in avg_precision])
loss_recall = torch.vstack([ref_recall - recall for recall in avg_recall])

print("Mean accuracy loss", torch.mean(loss_acc))
print("Mean Precision loss", torch.mean(loss_prec, dim=0))
print("Mean Recall loss", torch.mean(loss_recall, dim=0))
print("Top 3 accuracy feat", torch.topk(loss_acc, 3))
print("Top 3 Precision feat", torch.topk(loss_prec, 3, dim=0))
print("Top 3 Recall feat", torch.topk(loss_recall, 3, dim=0))
