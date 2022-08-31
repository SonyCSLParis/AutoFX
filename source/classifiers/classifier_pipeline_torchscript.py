import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from source.classifiers.classifier_pytorch import MLPClassifier

import source.classifiers.classifier_pytorch as torch_clf

# CHECKPOINT = "/home/alexandre/logs/classif4july/torch/version_0/checkpoints/epoch=999-step=151000.ckpt"

# clf = MLPClassifier.load_from_checkpoint(CHECKPOINT, input_size=143,
#                                          output_size=11, hidden_size=100,
#                                          activation='sigmoid', solver='adam',
#                                         max_iter=1000)


class ClassifierPipeline(nn.Module):
    def __init__(self, classifier, feat_extractor, scaler, remover):
        super(ClassifierPipeline, self).__init__()
        self.clf = classifier
        # self.feat_extractor = torch.jit.trace(feat_extractor, example)
        self.feat_extractor = feat_extractor
        self.scaler = scaler
        self.remover = remover
        self.pipeline = nn.Sequential(
            self.remover,
            self.feat_extractor,
            self.scaler,
            self.clf
        )

    def forward(self, audio):
        # cut_audio = self.remover(audio)
        # feat = self.feat_extractor(cut_audio)
        # scaled_feat = self.scaler(feat)
        # pred = self.clf(scaled_feat)
        out = self.pipeline(audio)
        # return torch.argmax(out)
        return out


dataset = pd.read_csv('/home/alexandre/dataset/IDMT_FULL_CUT_22050/out.csv', index_col=0)
subset = dataset.drop(columns=['file'])
target = subset['class']
data = subset.drop(columns=['class'])
print(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=2)
X_train, X_test = torch.tensor(X_train.values, dtype=torch.float), torch.tensor(X_test.values, dtype=torch.float)
y_train, y_test = torch.tensor(y_train.values), torch.tensor(y_test.values)

scaler = torch_clf.TorchStandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train.clone())
X_test_scaled = scaler.transform(X_test.clone())

train_dataset = torch_clf.ClassificationDataset(X_train_scaled, y_train)
test_dataset = torch_clf.ClassificationDataset(X_test_scaled, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=4)


