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
class ERA5Dataset(BaseDataset):

    def __init__(self, input_dir: str, datasplit_config: str, hparams_dict_config: str, mode: str = "train", seed: int = None, nsamples_ref: int = None):
        super().__init__(input_dir, datasplit_config, hparams_dict_config, mode, seed, nsamples_ref)
        self.get_filenames_from_datasplit()
        self.get_datasplit()
        self.load_data_from_nc()



    def get_hparams(self):
        """
        obtain the hparams from the dict to the class variables
        """
        method = ERA5Dataset.get_hparams.__name__

        try:
            self.context_frames = self.hparams['context_frames']
            self.max_epochs = self.hparams['max_epochs']
            self.batch_size = self.hparams['batch_size']
            self.shuffle_on_val = self.hparams['shuffle_on_val']

        except Exception as error:
           print("Method %{}: error: {}".format(method,error))
           raise("Method %{}: the hparameter dictionary must include 'context_frames','max_epochs','batch_size','shuffle_on_val'".format(method))

    def get_filenames_from_datasplit(self):
        """
        Get  absolute .nc path names based on the data splits patterns
        """
        self.filenames = []
        self.data_mode = self.data_dict[self.mode]
        for year, months in self.data_mode.items():
            for month in months:
                tf_files = os.path.join(self.input_dir,"era5_vars4ambs_{}{:02d}.nc".format(year, month))
                self.filenames.append(tf_files)

        return self.filenames


    def load_data_from_nc(self):
        """
        Obtain the data and meta information from the netcdf files, including the len of the samples, mim and max values
        :return: None
        """
        ds = xr.open_mfdataset(self.filenames)
        da = ds.to_array(dim = "variables").squeeze()
        dims = ["time", "lat", "lon"]

        max_vars, min_vars = da.max(dim = dims).values, da.min(dim = dims).values
        min_max_vars = np.array([min_vars, max_vars])
        self.min_max_values = np.transpose(min_max_vars, (1, 0))

        data_arr = np.squeeze(da.values)
        data_arr = np.transpose(data_arr, (1, 0, 2, 3)) #swap n_samples and variables position

        #obtain the meta information
        self.n_ts = len(ds["time"].values)
        self.lat = ds["lat"].values
        self.lon = ds["lon"].values
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

        data_arr = ERA5Dataset.load_data_from_nc()

        fixed_range = self.min_max_values


        def normalize_fn(x, min_value, max_value):

            return tf.divide(tf.subtract(x, min_value), tf.subtract(max_value, min_value))

        def normalize_fixed(x, current_range, n_vars=self.n_vars):
            current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
            x_norm = []
            for i in range(n_vars):
                dt_norm = normalize_fn(x[:, :, i, :, :], current_min[i], current_max[i])
                x_norm.append(dt_norm)

            x_normed = tf.stack(x_norm, axis=2)
            return x_normed

        def parse_example(line_batch):
            # features = tf.transpose(line_batch)
            features = normalize_fixed(line_batch, fixed_range)
            return features

        def data_generator():
            for d in data_arr:
                yield d

        if len(filenames) == 0:
            raise (
                "The filenames list is empty for {} dataset, please make sure your data_split dictionary is configured correctly".format(
                    self.mode))
        else:
            #group the data into sequenceds
            dataset = tf.data.Dataset.from_generator(data_generator, tf.float64)
            dataset = dataset.window(self.sequence_length, shift = 1, drop_remainder = True)
            dataset = dataset.flat_map(lambda window: window.batch(self.sequence_length))
            # shuffle the data
            if shuffle:
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size = 1024, count = self.max_epochs))
            else:
                dataset = dataset.repeat(self.max_epochs)
            dataset = dataset.shuffle()
            dataset = dataset.batch(self.batch_size) #obtain data with batch size
            dataset = dataset.map(parse_example) #normalise
            return dataset



# if __name__ == '__main__':
#     dataset = ERA5Dataset(input_dir: str = None, datasplit_config: str = None, hparams_dict_config: str = None,
#                  mode: str = "train", seed: int = None, nsamples_ref: int = None)
#     for next_element in dataset.take(2):
#         # time_s = time.time()
#         # tf.print(next_element.shape)
#         pass

