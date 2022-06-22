# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-03-17"

from .base_dataset import BaseDataset
import tensorflow as tf
import xarray as xr
import numpy as np
import os
from functools import partial


class ERA5Dataset(BaseDataset):
    def __init__(
        self,
        input_dir: str,
        datasplit_config: str,
        hparams_dict_config: str,
        mode: str = "train",
        seed: int = None,
        nsamples_ref: int = None,
    ):
        super().__init__(
            input_dir, datasplit_config, hparams_dict_config, mode, seed, nsamples_ref
        )
        self.load_data()

    def filnames(self, mode):
        """
        Get absolute .nc path names based on the data splits patterns
        """
        filenames = []
        data_mode = self.data_dict[mode]
        for year, months in data_mode.items():
            for month in months:
                tf_files = os.path.join(
                    self.input_dir, f"era5_vars4ambs_{year}{month:02d}.nc"
                )
                filenames.append(tf_files)

        return filenames

    def specific_hparams(self):
        return ["shift"]

    def load_data(self):
        """
        Obtain the data and meta information from the netcdf files, including the len of the samples, mim and max values
        return: np.array
        """
        ds = xr.open_mfdataset(self.filenames)

        da = ds.to_array(dim="variables").squeeze()
        dims = ["time", "lat", "lon"]

        data_arr = np.squeeze(da.values)
        data_arr = np.transpose(
            data_arr, (1, 2, 3, 0)
        )  # swap positions to [n_samples, lat, lon, channels]

        # obtain the meta information
        self.n_ts = len(ds["time"].values)
        self.nlat = len(ds["lat"].values)
        self.nlon = len(ds["lon"].values)
        self.n_samples = data_arr.shape[0]
        self.variables = list(ds.keys())
        self.n_vars = len(self.variables)
        return data_arr

    def normalize(self, x: tf.Tensor, mode: str):
        """
        x: tensor with shape [batch_size,seq_length,lat, lon, nvars]
        return: normalised data (tf.Tensor), the shape is the same as input "x"
        """

        def normalize_var(x, min, max):
            return tf.divide(tf.subtract(x, min), tf.subtract(max, min))
            # maybe evene return (x-min) / (max-min)

        x = x.copy()  # assure no inplace operation
        stats = self._stats_lookup[mode]

        for i in range(x.shape[-1]):
            x[:, :, :, :, i] = normalize_var(x[:, :, :, :, i], stats.minimum[i], stats.maximum[i])

        return x

    def num_examples_per_epoch(self):
        # total number of samples if the shift is one
        return self.n_ts - self.sequence_length + 1
