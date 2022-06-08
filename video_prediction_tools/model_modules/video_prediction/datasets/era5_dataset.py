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

    def __init__(self, input_dir: str, datasplit_config: str, hparams_dict_config: str, mode: str = "train", seed: int = None, nsamples_ref: int = None):
        super().__init__(input_dir, datasplit_config, hparams_dict_config, mode, seed, nsamples_ref)
        self.load_data()

    def get_filenames_from_datasplit(self):
        """
        Get  absolute .nc path names based on the data splits patterns
        """
        filenames = []
        data_mode = self.data_dict[self.mode]
        for year, months in self.data_mode.items():
            for month in months:
                tf_files = os.path.join(self.input_dir,"era5_vars4ambs_{}{:02d}.nc".format(year, month))
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

        da = ds.to_array(dim = "variables").squeeze()
        dims = ["time", "lat", "lon"]

        max_vars, min_vars = da.max(dim = dims).values, da.min(dim = dims).values
        self.min_max_values = np.array([min_vars, max_vars])

        data_arr = np.squeeze(da.values)
        data_arr = np.transpose(data_arr, (1, 2, 3, 0)) #swap positions to [n_samples, lat, lon, channels]

        #obtain the meta information
        self.n_ts = len(ds["time"].values)
        self.nlat = len(ds["lat"].values)
        self.nlon = len(ds["lon"].values)
        self.n_samples = data_arr.shape[0]
        self.variables = list(ds.keys())
        self.n_vars = len(self.variables)
        return data_arr


    def make_dataset(self):
        """
        Prepare batch_size dataset fed into to the models.
        If the data are from training dataset,then the data is shuffled;
        If the data are from val dataset, the shuffle var will be decided by the hparams.shuffled_on_val;
        if the data are from test dataset, the data will not be shuffled
        """
        method = ERA5Dataset.make_dataset.__name__

        shuffle = self.mode == 'train' or (self.mode == 'val' and self.shuffle_on_val)
        filenames = self.filenames
        data_arr = self.load_data()

        def normalize(x:tf.Tensor=None):
            """
            x: tensor with shape [batch_size,seq_length,lat, lon, nvars]
            norm_dim: dimension (var) to be normalised
            return: normalised data (tf.Tensor), the shape is the same as input "x"
            """

            def normalize_var(x, min, max):
                return tf.divide(tf.subtract(x, min), tf.subtract(max, min))
                # maybe evene return (x-min) / (max-min)

            x = x.copy() # assure no inplace operation
            mins, maxs = self.min_max_values

            for i in range(x.shape[-1]):
                x[:,:,:,:,i] = normalize_var(x[:, :, :, :, i], mins[i], maxs[i])
        
             
            return x


        def data_generator():
            for d in data_arr:
                yield d

        if len(filenames) == 0:
            raise ("The filenames list is empty for {} dataset, please make sure your data_split dictionary is configured correctly".format(self.mode))
        else:
            #group the data into sequenceds
            dataset = tf.data.Dataset.from_generator(data_generator,output_types=tf.float32, output_shapes=[self.nlat,self.nlon,self.n_vars])
            dataset = dataset.window(self.sequence_length, shift = self.shift, drop_remainder=True)
            dataset = dataset.flat_map(lambda window: window.batch(self.sequence_length))
            # shuffle the data
            if shuffle:
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=self.max_epochs, seed=self.seed)) #seed ?
            else:
                dataset = dataset.repeat(self.max_epochs)
            dataset = dataset.batch(self.batch_size) #obtain data with batch size
            dataset = dataset.map(normalize) #normalise
            return dataset

    def num_examples_per_epoch(self):
        #total number of samples if the shift is one
        return self.n_ts - self.sequence_length + 1

