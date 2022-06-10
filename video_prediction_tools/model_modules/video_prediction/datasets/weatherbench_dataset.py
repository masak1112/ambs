# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__email__ = "s.grasse@fz-juelich.de"
__author__ = "Simon Grasse"
__date__ = "2022-05-03"
from pathlib import Path

import xarray as xr
import tensorflow as tf

from .base_dataset import BaseDataset

class WeatherBenchDataset(BaseDataset):
    """Weatherbench dataset as presented in https://arxiv.org/abs/2002.00469."""

    def __init__(
        self,
        inputdir: str,
        datasplit_config: str,
        hparams_dict_config: str,
        mode: str = "train",
        seed: int = None,
        nsamples_ref: int = None,
    ):
        super.__init__(
            input_dir, datasplit_config, hparams_dict_config, mode, seed, nsamples_ref
        )
    
    def filenames(self, mode):
        self.data_mode = self.data_dict[self.mode]
        files = []
        for year in self.data_mode:
            files.append(self.input_dir / f"{variable}_{year}_{resolution}.nc")
        
        return files


    def load_data(self, filenames):
        ds = xr.open_mfdataset(filenames, coords="minimal", compat="override")
        ds = ds.drop_vars("level")

        da = ds.to_array(dim="variables").squeeze()

        dims = ["time", "lat", "lon", "variables"]
        return da.transpose(*dims)


    def normalize(self, x):
        # mean or min-max normalization ?
        def normalize_var(x, mean, std):
            return (x - mean) / std

        x = x.copy()  # assure no inplace operation

        # normalize each variable seperatly
        for i in range(x.shape[-1]):
            x[:, :, :, :, i] = normalize_var(
                x[:, :, :, :, i], mean[i], std[i]
            )

        return x

    def num_examples_per_epoch(self):
        # total number of samples if the shift is one
        return self.n_ts - self.sequence_length + 1
