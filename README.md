# AutoFX
## Automatic Effect Recognition and Configuration for Timbre Reproduction

__Intern:__ [Alexandre D'Hooge](https://github.com/adhooge)

__Supervisor:__ [Gaëtan Hadjeres](https://github.com/Ghadjeres)

This work was conducted as an internship in the Music Team of [SonyCSL Paris](https://csl.sony.fr/) during the summer 2022.

The report of the internship is available in `docs/AutoFX_report.pdf`.

### Abstract

Many musicians use audio effects to shape their sound to the point that these effects become part of
their sound identity. However, configuring audio effects often requires expert knowledge to find the
correct setting to reach a desired sound. During this internship, we studied a novel method to automatically
recognize the effect present in a reference sound and find parameters that allow to reproduce its timbre.
This tool aims at helping artists during their creative process to quickly configure effects to reproduce
a chosen sound, making it easier to explore similar sounds afterwards, similarly to what presets offer
but in a much more flexible manner.
We implement a classification algorithm to recognize the audio effect used on a guitar sound and reach
up to 95 % accuracy on the test set. This classifier is also compiled and can be used as a standalone
plugin to analyze a sound and automatically instanciate the correct effect. We also propose a pipeline
to generate a synthetic dataset of guitar sounds processed with randomly configured effects at speeds
unreachable before. We use that dataset to train different neural networks to automatically retrieve
effect’s parameters. We demonstrate that a feature-based approach with typical Music Information
Retrieval (MIR) features compete with a larger Convolutional Neural Network (CNN) trained on
audio spectrograms while being faster to train and requiring far less parameters. Contrary to the
existing literature, making the effects we use differentiable does not allow to improve the performance
of our networks which already propose fair reproduction of unseen audio effects when trained only
in a supervised manner on a loss on the parameters. We further complete our results with an online
perceptual experiment that shows that the proposed approach yields sound matches that are much
better than using random parameters, suggesting that this technique is indeed promising and that
any audio effect could be reproduced by a correctly configured generic effect.

___

### Install

0. Clone the repo:
```commandline
git clone https://github.com/adhooge/AutoFX
```

#### Using Conda

1. Create the environment using the `environment.yml` file:
```commandline
conda env create -f environment.yml
```

2. Activate the environment:
 ```commandline
conda activate autofx
 ```

#### Using Pip

Install `requirements.txt`. Note that this is a system-wide installation.
```commandline
pip install -r requirements.txt
```

#### Dataset

The dataset used for this work can be downloaded freely from [here](https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html).
Installation cannot be automated since a request form has to be submitted.
The compressed dataset weighs 6.5GB.

### Audio effects recognition

#### Computation of classification features

It is easier if all processed sounds are in the same folder. To put all Monophonic Guitar Sounds in the same folder, you can do it manually or run the following script:
```commandline
python3 -m data.copy_guitar_mono -i "PATH_TO_DATASET/Gitarre monophon" -o "PATH_TO_STORE_GUITAR_MONO"
python3 -m data.copy_guitar_mono -i "PATH_TO_DATASET/Gitarre monophon2" -o "PATH_TO_STORE_GUITAR_MONO"
```
This will recursively copy the `.wav` files into a new folder. If you do not need to keep the original files, you can add the `--cut` or `-c` argument at the end of each command.

To obtain the features required for classification, you then have to run the `FeatureExtractor` on the containing folder:
```commandline
python3 -m data.feat_extractor -i PATH_TO_GUITAR_MONO
```
This will generate an `out.csv` file containing the features for all input sounds. _Note: This may take a while._

#### Training the model
To train the classification model on the dataset, you can run:
```commandline
python3 -m source.classifiers.fx_recognition_fit -o PATH/TO/LOG/DIR -d PATH/TO/dataset.csv 
```
If you want to train a classifier network on aggregated effect classes, add the `--aggregate` or `-a` argument to the previous command.
You can monitor the model's performance during/after training using Tensorboard:
```commandline
tensorboard --logdir PATH/TO/LOG/DIR
```
The trained model is then available in the `checkpoints` directory of the log directory and can be loaded in Python using:
```python
from source.classifiers.classifier_pytorch import MLPClassifier
checkpoint = "PATH/TO/CHECKPOINT"
model = MLPClassifier.load_from_checkpoint(checkpoint)
```
the corresponding scaler can also be loaded using:
```python
import pickle
with open("/PATH/TO/LOG/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
```

### Parameters estimation

#### Data generation

The synthetic data for training parameters estimation models is obtained from clean guitar sounds. To put those sounds in a separate folder, you can do it manually or run:
```commandline
python3 -m data.get_clean_sounds -i "PATH/TO/IDMT/DATASET" -o "FOLDER/TO/STORE/CLEAN"
```
Add `--cut` or `-c` at the end of the previous command to move instead of copying.


____
UNDER CONSTRUCTION, please come back later