import pathlib

import pytorch_lightning as pl
import sklearn.model_selection
import torch.utils.data
from torch.utils.data import DataLoader, Subset
from cfgv import Optional
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from source.data.datasets import FeatureInDomainDataset, FeatureOutDomainDataset


class FeaturesDataModule(pl.LightningDataModule):
    def __init__(self, clean_dir: str, processed_dir: str,
                 out_of_domain_dir: str, batch_size: int = 32, num_workers: int = 4,
                 in_scaler_mean: list = None, in_scaler_std: list = None,
                 out_scaler_mean: list = None, out_scaler_std: list = None,
                 out_of_domain: bool = False, seed: int = None, reverb: bool = False,
                 conditioning: bool = False, classes2keep: list = None, return_file_name: bool = False,
                 csv_name: str = "data.csv", out_csv_name: str = "data.csv",
                 aggregated_classes: bool = False, fx_feat: bool = True, clf_feat: bool = False,
                 *args, **kwargs):
        super(FeaturesDataModule, self).__init__()
        self.clean_dir = clean_dir
        self.processed_dir = processed_dir
        self.out_of_domain_dir = out_of_domain_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.out_of_domain = out_of_domain
        if seed is None:
            seed = torch.randint(100000, (1, 1))
        self.seed = seed
        self.reverb = reverb
        self.in_scaler_mean = in_scaler_mean
        self.in_scaler_std = in_scaler_std
        self.out_scaler_mean = out_scaler_mean
        self.out_scaler_std = out_scaler_std
        self.conditioning = conditioning
        self.classes2keep = classes2keep
        self.return_file_name = return_file_name
        self.csv_name = csv_name
        self.out_csv_name = out_csv_name
        self.aggregated_classes = aggregated_classes
        self.fx_feat = fx_feat
        self.clf_feat = clf_feat
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        in_domain_full = FeatureInDomainDataset(self.processed_dir, validation=True,
                                                clean_path=self.clean_dir, processed_path=self.processed_dir,
                                                reverb=self.reverb, conditioning=self.conditioning,
                                                classes2keep=self.classes2keep, return_file_name=self.return_file_name,
                                                csv_name=self.csv_name, aggregated_classes=self.aggregated_classes,
                                                fx_feat=self.fx_feat, clf_feat=self.clf_feat)
        out_domain_full = FeatureOutDomainDataset(self.out_of_domain_dir, self.clean_dir, self.out_of_domain_dir,
                                                  index_col=0, conditioning=self.conditioning,
                                                  classes2keep=self.classes2keep,
                                                  return_file_name=self.return_file_name,
                                                  csv_name=self.out_csv_name,
                                                  fx_feat=self.fx_feat, clf_feat=self.clf_feat)
        if self.classes2keep is None:
            # split can be random if balanced classes is irrelevant
            self.in_train, self.in_val = torch.utils.data.random_split(in_domain_full,
                                                                       [len(in_domain_full) - len(in_domain_full) // 5,
                                                                        len(in_domain_full) // 5],
                                                                       generator=torch.Generator().manual_seed(
                                                                           self.seed))
            self.out_train, self.out_val = torch.utils.data.random_split(out_domain_full,
                                                                         [len(out_domain_full) - len(
                                                                             out_domain_full) // 5,
                                                                          len(out_domain_full) // 5],
                                                                         generator=torch.Generator().manual_seed(
                                                                             self.seed))
        else:
            # otherwise we keep balance between train and validation
            sss_in = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
            y_in = in_domain_full.data["fx_class"]
            X_in = in_domain_full.data.iloc[:, :-1]
            train_index, val_index = next(iter(sss_in.split(X_in, y_in)))
            self.in_train = Subset(in_domain_full, train_index)
            self.in_val = Subset(in_domain_full, val_index)
            self.out_train, self.out_val = torch.utils.data.random_split(out_domain_full,
                                                                         [len(out_domain_full) - len(
                                                                             out_domain_full) // 5,
                                                                          len(out_domain_full) // 5],
                                                                         generator=torch.Generator().manual_seed(
                                                                             self.seed))
            # sss_out = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
            # y_out = out_domain_full.data["fx_class"]
            # X_out = out_domain_full.data[:, :-1]
            # train_index, val_index = next(iter(sss_out.split(X_out, y_out)))
            # self.out_train = out_domain_full.data.iloc[train_index]
            # self.out_val = out_domain_full.data.iloc[val_index]
        if self.in_scaler_mean is None or self.in_scaler_std is None:
            tmp_dataloader = DataLoader(self.in_train, batch_size=len(self.in_train),
                                        num_workers=self.num_workers)
            in_domain_full.scaler.fit(next(iter(tmp_dataloader))[:][2])
        else:
            in_domain_full.scaler.mean = torch.tensor(self.in_scaler_mean)
            in_domain_full.scaler.std = torch.tensor(self.in_scaler_std)
        print("Scaler mean: ", in_domain_full.scaler.mean)
        print("Scaler std: ", in_domain_full.scaler.std)
        if self.out_scaler_std is None or self.out_scaler_mean is None:
            tmp_dataloader = DataLoader(self.out_train, batch_size=len(self.out_train),
                                        num_workers=self.num_workers)
            out_domain_full.scaler.fit(next(iter(tmp_dataloader))[:][2])
        else:
            out_domain_full.scaler.mean = torch.tensor(self.out_scaler_mean)
            out_domain_full.scaler.std = torch.tensor(self.out_scaler_std)
        print("Out Scaler mean: ", out_domain_full.scaler.mean)
        print("Out Scaler std: ", out_domain_full.scaler.std)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        in_dataloader = DataLoader(self.in_train, self.batch_size, num_workers=self.num_workers,
                                   shuffle=True)
        out_dataloader = DataLoader(self.out_train, self.batch_size, num_workers=self.num_workers,
                                    shuffle=True)
        if self.out_of_domain:
            return out_dataloader
        else:
            return in_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # print("CONDITIONING: ", self.conditioning)
        in_dataloader = DataLoader(self.in_val, self.batch_size, num_workers=self.num_workers,
                                   shuffle=True)
        out_dataloader = DataLoader(self.out_val, self.batch_size, num_workers=self.num_workers,
                                    shuffle=False)
        if self.out_of_domain:
            return out_dataloader
        else:
            return in_dataloader
