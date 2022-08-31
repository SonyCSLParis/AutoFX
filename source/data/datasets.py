import torch
import pathlib
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, Subset


class TorchStandardScaler:
    """
    from    https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576/8
    """

    def __init__(self):
        self.mean = torch.tensor(0)
        self.std = torch.tensor(1)

    def fit(self, x):
        self.mean = x.mean(0, keepdim=False)
        self.std = x.std(0, unbiased=False, keepdim=False)

    def transform(self, x):
        if x.device != self.mean.device or x.device != self.std.device:
            # print('YO', x.device, self.mean.device)
            x.to(self.mean.device)
        # print(x.shape)
        # print(self.mean.shape)
        x -= self.mean
        x /= (self.std + 1e-7)
        return x


class FeatureInDomainDataset(Dataset):
    def __init__(self, data_path: str = None, validation: bool = False,
                 clean_path: str = None, processed_path: str = None,
                 pad_length: int = None, reverb: bool = False,
                 conditioning: bool = False, classes2keep: list = None,
                 return_file_name: bool = False, csv_name: str = "data.csv",
                 aggregated_classes: bool = False, fx_feat: bool = True, clf_feat: bool = False):
        """

        :param data_path:
        :param validation: Deprecated, to remove.
        :param clean_path:
        :param processed_path:
        :param pad_length:
        :param reverb:
        :param conditioning:
        :param classes2keep: Classes that should be kept for training.
        :param return_file_name: flag to also return filename
        :param csv_name: Filename of the dataframe in .csv format. Default is 'data.csv'
        :param aggregated_classes: is the dataset for aggregated classes?
        :param fx_feat: Set to True if the Features for fx regression are needed (Onsets etc)
        :param clf_feat: Set to True if the features used for classification are needed.
        """
        if validation and (clean_path is None or processed_path is None):
            raise ValueError("Clean and Processed required for validation dataset.")
        self.data_path = pathlib.Path(data_path)
        self.validation = validation
        self.clean_path = pathlib.Path(clean_path) if clean_path is not None else None
        self.processed_path = pathlib.Path(processed_path) if processed_path is not None else None
        self.data = pd.read_csv(self.data_path / csv_name, index_col=0)
        columns = list(self.data.columns)
        num_features = 0
        num_param = 0
        num_cond = 0
        features_columns = []
        param_columns = []
        cond_columns = []
        for (i, c) in enumerate(columns):
            if 'f-' in c:
                num_features += 1
                features_columns.append(i)
            elif 'p-' in c:
                num_param += 1
                param_columns.append(i)
            elif 'c-' in c:
                num_cond += 1
                cond_columns.append(i)
        if fx_feat and not clf_feat:
            features_columns = features_columns[:-163]
        elif clf_feat and not fx_feat:
            features_columns = features_columns[-163:]
        self.num_features = num_features
        self.feat_columns = features_columns
        self.cond_columns = cond_columns
        self.num_param = num_param
        self.param_columns = param_columns
        if aggregated_classes and num_cond != 6:
            raise ValueError(f"6 classes should be in the dataset. {num_cond} have been found.")
        self.num_cond = num_cond
        self.scaler = TorchStandardScaler()
        if pad_length is None:
            if reverb:
                self.pad_length = 2 ** 17
            else:
                self.pad_length = 35000
        else:
            self.pad_length = pad_length
        self.reverb = reverb
        self.conditioning = conditioning
        # Keep only relevant classes
        self.classes2keep = classes2keep
        self.return_file_name = return_file_name
        if classes2keep is not None:
            self.data = self.data[self.data["fx_class"].isin(classes2keep)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        filename = self.data.iloc[item, 0]
        if self.validation:
            cln_snd_path = self.clean_path / (filename.split('_')[0] + '.wav')
            prc_snd_path = self.processed_path / (filename + '.wav')
            cln_sound, rate = torchaudio.load(cln_snd_path, normalize=True)
            prc_sound, rate = torchaudio.load(prc_snd_path, normalize=True)
            cln_sound = cln_sound[0]
            prc_sound = prc_sound[0]
            cln_sound = cln_sound / torch.max(torch.abs(cln_sound))
            prc_sound = prc_sound / torch.max(torch.abs(prc_sound))
            cln_pad = torch.zeros((1, self.pad_length))
            cln_pad[0, :len(cln_sound)] = cln_sound
            cln_pad[0, len(cln_sound):] = torch.randn(self.pad_length - len(cln_sound)) / 1e9
            prc_pad = torch.zeros((1, self.pad_length))
            prc_pad[0, :len(prc_sound)] = prc_sound
            prc_pad[0, len(prc_sound):] = torch.randn(self.pad_length - len(prc_sound)) / 1e9
        # print(self.data.iloc[item])
        params = self.data.iloc[item, self.param_columns]
        params = torch.Tensor(params)
        if self.reverb:
            params = torch.hstack([params, torch.zeros(1)])
        # print(params)
        # print(self.feat_columns)
        features = self.data.iloc[item, self.feat_columns]
        if self.conditioning:
            conditioning = self.data.iloc[item, self.cond_columns]
            conditioning = torch.tensor(conditioning, dtype=torch.float)
            fx_class = self.data.iloc[item, -1]
            fx_class = torch.tensor(fx_class, dtype=torch.int)
        else:
            conditioning = 'None'
            # fx_class = None
            fx_class = self.data.iloc[item, -1]
            fx_class = torch.tensor(fx_class, dtype=torch.int)
        features = torch.Tensor(features)
        # print("DHJSHDHSDHS", features.shape)
        features = self.scaler.transform(features)
        # print(features)
        if self.validation:         # TODO: remove that
            if self.conditioning:
                if self.return_file_name:
                    return cln_pad, prc_pad, features, params, conditioning, fx_class, filename
                else:
                    return cln_pad, prc_pad, features, params, conditioning, fx_class
            else:
                # return cln_pad, prc_pad, features, params
                return cln_pad, prc_pad, features, params, conditioning, fx_class
        else:
            return features, params

    @property
    def target_classes(self):
        if not self.conditioning:
            return None
        else:
            return torch.tensor(self.data["conditioning"])

    def target_classes_subset(self, indices):
        if not self.conditioning:
            return None
        else:
            out = torch.tensor(self.data[["conditioning"]].iloc[indices].values)
            return out.float()


class FeatureOutDomainDataset(Dataset):
    # TODO: Update for conditioning
    def __init__(self, data_path: str,
                 clean_path: str = None, processed_path: str = None,
                 pad_length: int = 35000, index_col: int = None,
                 conditioning: bool = False, classes2keep: list = None,
                 return_file_name: bool = False, csv_name: str = "data.csv",
                 fx_feat: bool = True, clf_feat: bool = False
                 ):
        """

        :param data_path:
        :param clean_path:
        :param processed_path:
        :param pad_length:
        :param index_col:
        :param conditioning:
        :param classes2keep:
        :param return_file_name: flag to also return filename. Defaults to False.
        :param fx_feat: Set to True if the Features for fx regression are needed (Onsets etc)
        :param clf_feat: Set to True if the features used for classification are needed.
        """
        self.data_path = pathlib.Path(data_path)
        self.clean_path = pathlib.Path(clean_path) if clean_path is not None else clean_path
        self.processed_path = pathlib.Path(processed_path) if processed_path is not None else processed_path
        self.data = pd.read_csv(self.data_path / csv_name, index_col=index_col)
        columns = self.data.columns
        num_features = 0
        num_param = 0
        num_cond = 0
        features_columns = []
        param_columns = []
        cond_columns = []
        for (i, c) in enumerate(columns):
            if 'f-' in c:
                num_features += 1
                features_columns.append(i)
            elif 'p-' in c:
                num_param += 1
                param_columns.append(i)
            elif 'c-' in c:
                num_cond += 1
                cond_columns.append(i)
        if fx_feat and not clf_feat:
            features_columns = features_columns[:-163]
        elif clf_feat and not fx_feat:
            features_columns = features_columns[-163:]
        self.num_features = num_features
        self.feat_columns = features_columns
        self.cond_columns = cond_columns
        self.num_param = num_param
        self.param_columns = param_columns
        self.num_cond = num_cond
        if "conditioning" in self.data.columns:
            self.data = self.data.drop(columns=["conditioning"])
        self.fx2clean = pd.read_csv(self.data_path / "fx2clean.csv", index_col=0)
        self.pad_length = pad_length
        self.scaler = TorchStandardScaler()
        self.conditioning = conditioning
        self.return_file_name = return_file_name
        self.classes2keep = classes2keep
        if classes2keep is not None:
            self.data = self.data[self.data["fx_class"].isin(classes2keep)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        filename = self.data.iloc[item, 0]
        if isinstance(filename, pd.Series):
            for (i, f) in enumerate(filename):
                cln_snd_path = self.clean_path / (self.fx2clean.iloc[i, 1] + '.wav')
                fx_snd_path = self.processed_path / (self.fx2clean.iloc[i, 0] + '.wav')
                cln_sound, rate = torchaudio.load(cln_snd_path, normalize=True)
                prc_sound, rate = torchaudio.load(fx_snd_path, normalize=True)
                cln_sound = cln_sound[0]
                prc_sound = prc_sound[0]
                cln_sound = cln_sound / torch.max(torch.abs(cln_sound))
                prc_sound = prc_sound / torch.max(torch.abs(prc_sound))
                cln_pad = torch.zeros((1, self.pad_length))
                cln_pad[0, :len(cln_sound)] = cln_sound
                cln_pad[0, len(cln_sound):] = torch.randn(self.pad_length - len(cln_sound)) / 1e9
                prc_pad = torch.zeros((1, self.pad_length))
                prc_pad[0, :len(prc_sound)] = prc_sound
                prc_pad[0, len(prc_sound):] = torch.randn(self.pad_length - len(prc_sound)) / 1e9
                features = self.data.iloc[i, :]
                if self.conditioning:
                    features = features[:-1]
                    conditioning = torch.tensor(features[-1], dtype=torch.float)
                else:
                    conditioning = None
                features = torch.Tensor(features)
                features = self.scaler.transform(features)
                if self.conditioning:
                    return cln_pad, prc_pad, features, conditioning
                else:
                    return cln_pad, prc_pad, features
        else:
            # cln_snd_path = self.clean_path / (self.fx2clean.iloc[item, 1] + '.wav')
            # fx_snd_path = self.processed_path / (self.fx2clean.iloc[item, 0] + '.wav')
            # print(filename)
            # print(self.fx2clean.loc[[filename]].values[0])
            cln_snd_path = self.clean_path / (self.fx2clean.loc[[filename]].values[0] + '.wav')
            fx_snd_path = self.processed_path / (filename + ".wav")
            cln_sound, rate = torchaudio.load(cln_snd_path[0], normalize=True)
            prc_sound, rate = torchaudio.load(fx_snd_path, normalize=True)
            cln_sound = cln_sound[0]
            prc_sound = prc_sound[0]
            cln_sound = cln_sound / torch.max(torch.abs(cln_sound))
            prc_sound = prc_sound / torch.max(torch.abs(prc_sound))
            cln_pad = torch.zeros((1, self.pad_length))
            cln_pad[0, :len(cln_sound)] = cln_sound
            cln_pad[0, len(cln_sound):] = torch.randn(self.pad_length - len(cln_sound)) / 1e9
            prc_pad = torch.zeros((1, self.pad_length))
            if len(prc_sound > 35000):
                prc_sound = prc_sound[:35000]
            prc_pad[0, :len(prc_sound)] = prc_sound
            prc_pad[0, len(prc_sound):] = torch.randn(self.pad_length - len(prc_sound)) / 1e9
            # print("DHCBNHJQSK", self.feat_columns)
            features = self.data.iloc[item, self.feat_columns]
            # print(features.shape)
            if self.conditioning:
                conditioning = self.data.iloc[item, self.cond_columns]
                conditioning = torch.tensor(conditioning, dtype=torch.float)
                fx_class = self.data.iloc[item, -1]
                fx_class = torch.tensor(fx_class, dtype=torch.int)
            else:
                conditioning = None
                fx_class = None
            features = torch.Tensor(features)
            features = self.scaler.transform(features)
            if self.conditioning:
                if self.return_file_name:
                    return cln_pad, prc_pad, features, conditioning, fx_class, filename
                else:
                    return cln_pad, prc_pad, features, conditioning, fx_class
            else:
                return cln_pad, prc_pad, features

    @property
    def target_classes(self):
        if not self.conditioning:
            return None
        else:
            return torch.tensor(self.data["conditioning"])

    def target_classes_subset(self, indices):
        if not self.conditioning:
            return None
        else:
            out = torch.tensor(self.data[["conditioning"]].iloc[indices].values)
            return out.float()
