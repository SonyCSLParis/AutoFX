from typing import List, Tuple

import numpy as np
import pedalboard as pdb
import torch

from source import util


def _settings_list2dict(settings_list, fx: pdb.Plugin):
    settings_dict = {}
    items = list(fx.__dict__.items())
    cnt = 0
    for item in items:
        if isinstance(item[1], property):
            settings_dict[item[0]] = settings_list[cnt]
            cnt += 1
    return settings_dict


class CustomDistortion:
    def __call__(self, audio, rate, same: bool = True, *args, **kwargs):
        return self.process(audio, rate, same, args, kwargs)

    def __init__(self):
        lo_filter = pdb.LowShelfFilter()
        hi_filter = pdb.HighShelfFilter()
        disto = pdb.Distortion()
        self.fx = pdb.Pedalboard([disto, lo_filter, hi_filter])

    def process(self, audio, rate, same: bool = True, *args, **kwargs):
        """
        :param audio:
        :param rate:
        :param args:
        :param kwargs:
        :param same: Should the output have the same shape as the input? Default is True.
        :return:
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        out = self.fx(audio, rate)
        if same and (out.shape != audio.shape):
            tmp = torch.zeros_like(audio)
            tmp[:, :, :out.shape[-1]] = out
            out = tmp
        return torch.tensor(out)

    def set_fx_params(self, settings: list[dict] or dict or list, flat: bool = False,
                      param_range: List[Tuple] = None) -> None:
        params = torch.clone(torch.Tensor(settings))
        if param_range is None:
            param_range = [(0, 1)] * len(params)
        # print(param_range)
        # print(params)
        for i in range(len(params)):
            params[i] = params[i] * (param_range[i][1] - param_range[i][0]) + \
                        param_range[i][0]
        params = torch.Tensor(params)
        # if flat:
        #     params = torch.reshape(params, (num_bands, self.num_fx, -1))
        # if params.ndim < 2 or (params.ndim == 2 and not isinstance(params[0, 0], dict)):
        #     raise NotImplementedError(params)
        # else:
        #    for b in range(num_bands):
        lo_filt = _settings_list2dict(params[1:4], pdb.HighShelfFilter)
        disto = _settings_list2dict([params[0]], pdb.Distortion)
        hi_filt = _settings_list2dict(params[4:], pdb.LowShelfFilter)
        self.fx = util.set_fx_params(self.fx, [disto, lo_filt, hi_filt])

    def add_perturbation_to_fx_params(self, perturb, param_range):
        settings_list = self.settings_list
        param_range = [[param_range[0]], param_range[1:4], param_range[4:]]
        num_params = [1, 3, 3]
        for f in range(3):
            for p in range(num_params[f]):
                eps = perturb[num_params[f] + p]
                scaled_eps = eps * (param_range[f][p][1] - param_range[f][p][0])
                settings_list[f][p] += scaled_eps
                settings_list[f][p] = min(param_range[f][p][1], settings_list[f][p])
                settings_list[f][p] = max(param_range[f][p][0], settings_list[f][p])
        settings_list = [item for sublist in settings_list for item in sublist]
        settings_list = torch.tensor(settings_list).flatten()
        self.set_fx_params(settings_list.tolist(), flat=True)

    def fake_add_perturbation_to_fx_params(self, perturb, param_range, fake_num_bands):
        self.add_perturbation_to_fx_params(perturb, param_range)

    @property
    def settings(self):
        settings = [util.get_fx_params(self.fx[0])]
        settings.append(util.get_params_iirfilter(self.fx[1], 'lo'))
        settings.append(util.get_params_iirfilter(self.fx[2], 'hi'))
        return settings

    @property
    def settings_list(self):
        settings_dict = self.settings
        settings_list = []
        for fx in settings_dict:
            tmp = []
            for dico in fx:
                tmp.append(list(dico.values()))
            settings_list.append(tmp[0])
        return settings_list

    @property
    def total_num_params_per_band(self):
        return 7
