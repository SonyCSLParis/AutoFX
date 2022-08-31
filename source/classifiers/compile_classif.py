import torch
import pickle
import source.classifiers.classifier_pytorch as torch_clf
from source.classifiers.classifier_pipeline_torchscript import ClassifierPipeline

CHECKPOINT = "/home/alexandre/logs/classif4july/guitar_Mono/version_1/checkpoints/epoch=66-step=20770.ckpt"

clf = torch_clf.MLPClassifier.load_from_checkpoint(CHECKPOINT, input_size=163, output_size=11, hidden_size=100, activation="sigmoid", solver="adam", max_iter=1000)
extractor = torch_clf.FeatureExtractor()
cutter = None
with open("/home/alexandre/logs/classif4july/guitar_Mono/version_1/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

scaler_mod = torch_clf.ScalerModule()
scaler_mod.mean = scaler.mean
scaler_mod.std = scaler.std

pipe = ClassifierPipeline(clf, extractor, scaler_mod, cutter)
p = torch.jit.script(pipe)
p.save("/home/alexandre/logs/classif4july/guitar_Mono/version_1/no_cutter_no_argmax.pt")