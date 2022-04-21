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
        self.get_hparams()
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
            self.sequence_length = self.hparams["sequence_length"]
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
        self.min_max_values = np.array([min_vars, max_vars])

        data_arr = np.squeeze(da.values)
        data_arr = np.transpose(data_arr, (1, 2, 3, 0)) #swap positions to [n_samples, lat, lon, channels]

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
        data_arr = self.load_data_from_nc()
        
        def normalize_fn(x:tf.Tensor, min_value:float, max_value:float):
            return tf.divide(tf.subtract(x, min_value), tf.subtract(max_value, min_value))

        def normalize_fixed(x:tf.Tensor=None, min_max_values:list=None,norm_dim:int=2):
            """
            x is the tensor with the shape of [batch_size,seq_length, n_vars, lat, lon]
            min_max_values is a list contains min and max values of variables. the first element of list is the min values for variables, and the second is the max values
            norm_dim is a int that indicate the dim (var) to be normalised
            return: normalised data (tf.Tensor), the shape is the same as input "x"
            """
             
            current_min, current_max = min_max_values[0], min_max_values[1]
            n_vars = len(current_min)
            if not len(current_min) == len(current_max):
                raise ("The length of min and max values should be equal to the number of variables in normalized_fixed function!")

            x_norm = []
            for i in range(n_vars):
                dt_norm = normalize_fn(x[:, :, :, :, i], current_min[i], current_max[i])
                x_norm.append(dt_norm)
            x_norm = tf.stack(x_norm,axis=norm_dim) #[batch_size,sequence_length,nvar, lon,lat]
            #x_norm = tf.transpose(x_norm,perm=[1,2,0,3,4])
            return x_norm

        def parse_example(line_batch):
            # features = tf.transpose(line_batch)
            features = normalize_fixed(line_batch, self.min_max_values)
            return features

        def data_generator():
            for d in data_arr:
                yield d

        if len(filenames) == 0:
            raise ("The filenames list is empty for {} dataset, please make sure your data_split dictionary is configured correctly".format(self.mode))
        else:
            #group the data into sequenceds
            dataset = tf.data.Dataset.from_generator(data_generator,output_types=tf.float32,output_shapes=[len(self.lat),len(self.lon),self.n_vars])
            dataset = dataset.window(self.sequence_length, shift = 1, drop_remainder = True)
            dataset = dataset.flat_map(lambda window: window.batch(self.sequence_length))
            # shuffle the data
            if shuffle:
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size = 1024, count = self.max_epochs))
            else:
                dataset = dataset.repeat(self.max_epochs)
            dataset = dataset.batch(self.batch_size) #obtain data with batch size
            dataset = dataset.map(parse_example) #normalise
            return dataset

    def num_examples_per_epoch(self):
        #total number of samples if the shift is one
        return self.n_ts - self.sequence_length + 1



# if __name__ == '__main__':
#     dataset = ERA5Dataset(input_dir: str = None, datasplit_config: str = None, hparams_dict_config: str = None,
#                  mode: str = "train", seed: int = None, nsamples_ref: int = None)
#     for next_element in dataset.take(2):
#         # time_s = time.time()
#         # tf.print(next_element.shape)
#         pass

